"""Self-contained file-descriptor monitor for the SkyRL RL driver.

This is a minimal, dependency-free port of the FileDescriptorMonitor used in
the OT-Agent datagen path (`hpc/local_runner_utils.py`). It is duplicated here
on purpose so it does NOT need to import anything from OT-Agent — the RL conda
env may not have OT-Agent on its path.

Goal: log file-descriptor usage of the *driver* process (the one that
FD-aborts with `uv__epoll_ctl_prep` SIGABRT on long a3 RL chains) every
`interval` seconds on a daemon thread. Output uses the same `[fd-monitor]`
prefix/format as the datagen monitor so existing greps keep working, and uses
`print(..., flush=True)` so it lands in the SLURM `.out`.

Only start this on the driver / main entrypoint process (not every Ray
worker) to avoid log spam.
"""
from __future__ import annotations

import os
import resource
import threading
import time
from pathlib import Path

DEFAULT_FD_MONITOR_INTERVAL = 120  # 2 minutes


def _get_fd_usage() -> tuple:
    """Get current file descriptor usage.

    Returns:
        Tuple of (current_open_fds, soft_limit, hard_limit, percent_used).
        Returns (-1, -1, -1, 0.0) on any failure (e.g. /proc unavailable).
    """
    try:
        pid = os.getpid()
        fd_dir = Path(f"/proc/{pid}/fd")
        if fd_dir.exists():
            current_fds = len(list(fd_dir.iterdir()))
        else:
            # Fallback for non-Linux systems (no /proc) — count via fstat.
            current_fds = 0
            for fd in range(1024):
                try:
                    os.fstat(fd)
                    current_fds += 1
                except OSError:
                    pass

        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        percent_used = (current_fds / soft_limit * 100) if soft_limit > 0 else 0
        return current_fds, soft_limit, hard_limit, percent_used
    except Exception:
        return -1, -1, -1, 0.0


def _get_mem_usage() -> tuple:
    """Get this process's RSS + system available memory (Linux /proc).

    Added 2026-05-28 to test the OOM hypothesis: the driver aborts in
    `uv__epoll_ctl_prep` with plenty of FD headroom, so the suspected cause is
    host memory exhaustion (ENOMEM), not file descriptors. Self-contained
    (no psutil) — reads /proc directly.

    Returns:
        Tuple of (rss_kb, mem_avail_kb, mem_total_kb, system_percent_used).
        Returns (-1, -1, -1, 0.0) on any failure (e.g. /proc unavailable).
    """
    try:
        rss_kb = -1
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    break
        mem_avail_kb = mem_total_kb = -1
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    mem_avail_kb = int(line.split()[1])
                elif line.startswith("MemTotal:"):
                    mem_total_kb = int(line.split()[1])
        system_pct = (
            (mem_total_kb - mem_avail_kb) / mem_total_kb * 100
            if mem_total_kb > 0
            else 0.0
        )
        return rss_kb, mem_avail_kb, mem_total_kb, system_pct
    except Exception:
        return -1, -1, -1, 0.0


def _get_cgroup_mem() -> tuple:
    """Get this container's CGROUP memory current + limit (bytes).

    Added 2026-07-11 for the 80B naive-map policy-node host-RAM OOM: the pod is
    OOM-killed at the ``--memory`` CGROUP cap (~1303 GiB for ``--memory 1400GB``),
    which is SMALLER than the node's physical RAM (~2014 GiB). So the
    ``/proc/meminfo`` node view (``_get_mem_usage``) reads "healthy" (~65% of the
    node) right up to the kernel cgroup-OOM SIGKILL — it never sees the binding
    cap. This reads the cgroup's OWN accounting (v2 ``memory.current`` /
    ``memory.max``, falling back to v1 ``memory.usage_in_bytes`` /
    ``memory.limit_in_bytes``) so the log shows RSS/usage vs the ACTUAL cap.

    Returns:
        Tuple of (cur_bytes, max_bytes). ``max_bytes`` is -1 when unlimited
        ("max" on v2, or a sentinel-huge value on v1). (-1, -1) on any failure.
    """
    try:
        # cgroup v2 (unified hierarchy) — the container sees its own cgroup root.
        v2_cur = "/sys/fs/cgroup/memory.current"
        v2_max = "/sys/fs/cgroup/memory.max"
        if os.path.exists(v2_cur):
            with open(v2_cur) as f:
                cur = int(f.read().strip())
            mx = -1
            try:
                with open(v2_max) as f:
                    raw = f.read().strip()
                mx = -1 if raw == "max" else int(raw)
            except Exception:
                mx = -1
            return cur, mx
        # cgroup v1 fallback.
        v1_cur = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
        v1_max = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        if os.path.exists(v1_cur):
            with open(v1_cur) as f:
                cur = int(f.read().strip())
            mx = -1
            try:
                with open(v1_max) as f:
                    lim = int(f.read().strip())
                # v1 "unlimited" is a near-INT64 sentinel; treat >~1 PiB as no cap.
                mx = -1 if lim > (1 << 50) else lim
            except Exception:
                mx = -1
            return cur, mx
        return -1, -1
    except Exception:
        return -1, -1


def _log_status(peaks: dict | None = None) -> None:
    """Log current file descriptor status with the [fd-monitor] prefix."""
    current, soft, hard, percent = _get_fd_usage()

    if current < 0:
        print("[fd-monitor] Unable to read file descriptor usage", flush=True)
        return

    if percent >= 90:
        level = "CRITICAL"
    elif percent >= 75:
        level = "WARNING"
    elif percent >= 50:
        level = "INFO"
    else:
        level = "OK"

    timestamp = time.strftime("%H:%M:%S")
    print(
        f"[fd-monitor] [{timestamp}] {level}: {current:,} / {soft:,} FDs open "
        f"({percent:.1f}% of soft limit, hard limit: {hard:,})",
        flush=True,
    )

    if percent >= 75:
        print(
            "[fd-monitor] Consider reducing --n_concurrent or increasing ulimit -n",
            flush=True,
        )

    # Memory telemetry (RSS + system available) — the suspected cause of the
    # driver SIGABRT once FDs were ruled out (see
    # agent_logs/2026-05-28_fresh_a3_chain_crash_not_fd.md).
    rss_kb, avail_kb, total_kb, sys_pct = _get_mem_usage()
    if rss_kb >= 0 and total_kb > 0:
        if sys_pct >= 95:
            mlevel = "CRITICAL"
        elif sys_pct >= 85:
            mlevel = "WARNING"
        elif sys_pct >= 70:
            mlevel = "INFO"
        else:
            mlevel = "OK"
        gib = 1048576.0  # KiB per GiB
        # Track the peak RSS across the process lifetime (the number that matters
        # for a cgroup-OOM post-mortem — the instantaneous sample rarely catches
        # the exact pre-kill peak).
        peak_str = ""
        if peaks is not None:
            peaks["rss_kb"] = max(peaks.get("rss_kb", 0), rss_kb)
            peak_str = f", peak RSS {peaks['rss_kb'] / gib:.2f} GiB"
        print(
            f"[fd-monitor] [{timestamp}] {mlevel}: RSS {rss_kb / gib:.2f} GiB{peak_str} | "
            f"node mem {(total_kb - avail_kb) / gib:.1f}/{total_kb / gib:.1f} GiB used "
            f"({sys_pct:.1f}%), avail {avail_kb / gib:.1f} GiB",
            flush=True,
        )
        if sys_pct >= 85:
            print(
                "[fd-monitor] Node memory pressure HIGH — consider reducing "
                "n_concurrent_trials / num_parallel_generation_workers",
                flush=True,
            )

    # CGROUP telemetry — the ACTUAL binding cap on CoreWeave (the `--memory`
    # limit is smaller than the node's physical RAM, so `node mem` above never
    # sees the wall that OOM-kills the pod). This is the decisive host-RAM signal
    # for the 80B GDN-scan policy-node OOM.
    gib_b = 1073741824.0  # bytes per GiB
    cg_cur, cg_max = _get_cgroup_mem()
    if cg_cur >= 0:
        cg_peak_str = ""
        if peaks is not None:
            peaks["cg_cur"] = max(peaks.get("cg_cur", 0), cg_cur)
            cg_peak_str = f", peak {peaks['cg_cur'] / gib_b:.2f} GiB"
        if cg_max > 0:
            cg_pct = cg_cur / cg_max * 100
            if cg_pct >= 95:
                clevel = "CRITICAL"
            elif cg_pct >= 85:
                clevel = "WARNING"
            elif cg_pct >= 70:
                clevel = "INFO"
            else:
                clevel = "OK"
            print(
                f"[fd-monitor] [{timestamp}] {clevel}: cgroup mem "
                f"{cg_cur / gib_b:.2f}/{cg_max / gib_b:.2f} GiB ({cg_pct:.1f}% of cap)"
                f"{cg_peak_str}",
                flush=True,
            )
            if cg_pct >= 85:
                print(
                    "[fd-monitor] CGROUP memory pressure HIGH — approaching the "
                    "--memory cap; OOM-kill imminent (reduce policy ranks/node, "
                    "GDN scan working set, or n_concurrent_trials)",
                    flush=True,
                )
        else:
            print(
                f"[fd-monitor] [{timestamp}] OK: cgroup mem {cg_cur / gib_b:.2f} GiB "
                f"(no cap){cg_peak_str}",
                flush=True,
            )


def _run(stop_event: threading.Event, interval: int) -> None:
    """Background thread loop: log immediately, then every `interval` seconds.

    Maintains a per-thread ``peaks`` dict so each sample can report the running
    peak RSS + peak cgroup usage (the numbers that matter for a cgroup-OOM
    post-mortem — the instantaneous sample rarely lands on the pre-kill peak).
    """
    peaks: dict = {}
    _log_status(peaks)
    while not stop_event.is_set():
        stop_event.wait(interval)
        if not stop_event.is_set():
            _log_status(peaks)


def start_fd_monitor(interval_seconds: int = DEFAULT_FD_MONITOR_INTERVAL) -> threading.Event:
    """Start a daemon thread that periodically logs FD usage of this process.

    Self-contained and best-effort: never raises into the caller. Intended to
    be started once in the RL driver entrypoint (skyrl_entrypoint), not in
    every Ray worker.

    Args:
        interval_seconds: How often to log FD usage (default: 120s). A value
            <= 0 disables the monitor (no-op).

    Returns:
        The threading.Event used to stop the loop. Set it to stop early; the
        thread is a daemon so it does not need to be joined for shutdown.
    """
    stop_event = threading.Event()
    if interval_seconds <= 0:
        print("[fd-monitor] Disabled (interval <= 0)", flush=True)
        return stop_event

    thread = threading.Thread(
        target=_run,
        args=(stop_event, interval_seconds),
        daemon=True,
        name="fd-monitor",
    )
    thread.start()
    print(f"[fd-monitor] Started monitoring (every {interval_seconds}s)", flush=True)
    return stop_event
