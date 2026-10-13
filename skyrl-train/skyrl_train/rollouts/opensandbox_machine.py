"""Run the Shellbox machine contract on an OpenSandbox sandbox.

The factory mirrors ``shellbox.backends.daytona.machine.DaytonaMachineFactory``:
one sandbox per machine, a startup deadline around create and readiness, a
server-side TTL, and teardown through the machine's ``close``.

Requests follow Harbor 0.17.1's OpenSandbox provider (the copy overfit_tbench
vendors as ``sandbox/opensandbox_upstream.py``) together with the fixes in its
``patch_opensandbox_compat.py``: the API key is also sent as a header so the
execd health check authenticates through the server proxy, the execd protocol
comes from the domain's scheme, and readiness fails fast on a Failed or
Terminated sandbox with the reason the server reported.

The SDK is the optional ``opensandbox`` extra and is imported only when a
factory first creates a sandbox.
"""

import asyncio
import logging
import math
import shlex
import stat
import uuid
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from rigging.timing import ExponentialBackoff
from shellbox.backends.docker.machine import INTERRUPT_TIMEOUT
from shellbox.image import RegistryImage
from shellbox.machine import (
    Command,
    DockerImage,
    ExitReason,
    MachineSpec,
    NetworkPolicy,
    Result,
    UnsupportedMachineSpec,
)

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_TTL_MINUTES = 360
API_KEY_HEADER = "OPEN-SANDBOX-API-KEY"
CREATE_ATTEMPTS = 3
# Provisioning errors worth another create, matched by name so the SDK stays optional.
TRANSIENT_SDK_EXCEPTIONS = frozenset(
    {
        "SandboxApiException",
        "SandboxInternalException",
        "SandboxReadyTimeoutException",
        "SandboxUnhealthyException",
        "PoolAcquireFailedException",
        "PoolEmptyException",
        "PoolStateStoreUnavailableException",
        "PoolStateStoreContentionException",
    }
)
TERMINAL_STATES = ("Failed", "Terminated")


class OpenSandboxBackend(StrEnum):
    """Shellbox's Backend enum has no OpenSandbox member; this one reports the same way."""

    OPENSANDBOX = "opensandbox"


@dataclass(frozen=True)
class OpenSandboxSDK:
    """The SDK names this backend uses, injectable so tests need no server or package."""

    Sandbox: Any
    ConnectionConfig: Any
    RunCommandOpts: Any
    ExecutionHandlers: Any
    SearchEntry: Any
    WriteEntry: Any
    NetworkPolicy: Any
    NetworkRule: Any
    SandboxImageAuth: Any
    SandboxImageSpec: Any


def load_sdk() -> OpenSandboxSDK:
    try:
        from opensandbox import Sandbox
        from opensandbox.config import ConnectionConfig
        from opensandbox.models.execd import ExecutionHandlers, RunCommandOpts
        from opensandbox.models.filesystem import SearchEntry, WriteEntry
        from opensandbox.models.sandboxes import NetworkPolicy, NetworkRule, SandboxImageAuth, SandboxImageSpec
    except ImportError as error:
        raise ImportError(
            "The OpenSandbox backend requires the opensandbox extra (uv sync --extra opensandbox)"
        ) from error
    return OpenSandboxSDK(
        Sandbox,
        ConnectionConfig,
        RunCommandOpts,
        ExecutionHandlers,
        SearchEntry,
        WriteEntry,
        NetworkPolicy,
        NetworkRule,
        SandboxImageAuth,
        SandboxImageSpec,
    )


class OpenSandboxNetworkMode(StrEnum):
    BLOCK_ALL = "block_all"
    UNRESTRICTED = "unrestricted"
    DOMAIN_ALLOW_LIST = "domain_allow_list"


@dataclass(frozen=True)
class OpenSandboxNetworkPolicy:
    """Daytona's policy modes that OpenSandbox can express; its egress rules match domains, not CIDRs."""

    mode: OpenSandboxNetworkMode
    value: str | None = None

    def __post_init__(self) -> None:
        if self.mode == "network_allow_list":
            raise ValueError("OpenSandbox egress rules match domains; network_allow_list is not supported")
        mode = OpenSandboxNetworkMode(self.mode)
        if mode is OpenSandboxNetworkMode.DOMAIN_ALLOW_LIST and not self.value:
            raise ValueError(f"{mode} requires an allow list")
        if mode is not OpenSandboxNetworkMode.DOMAIN_ALLOW_LIST and self.value is not None:
            raise ValueError(f"{mode} does not accept an allow list")

    def build(self, sdk: OpenSandboxSDK) -> Any | None:
        """Return the egress policy, or None when the sandbox keeps the server default."""
        mode = OpenSandboxNetworkMode(self.mode)
        if mode is OpenSandboxNetworkMode.UNRESTRICTED:
            return None
        rules = []
        if mode is OpenSandboxNetworkMode.DOMAIN_ALLOW_LIST:
            assert self.value is not None
            rules = [
                sdk.NetworkRule(action="allow", target=domain.strip())
                for domain in self.value.split(",")
                if domain.strip()
            ]
        return sdk.NetworkPolicy(defaultAction="deny", egress=rules)


def connection_config(
    sdk: OpenSandboxSDK,
    *,
    domain: str,
    api_key: str,
    use_server_proxy: bool,
    request_timeout_sec: float | None,
) -> Any:
    # The execd endpoint is a bare host:port, so a protocol that disagrees with the
    # domain's scheme speaks TLS to a plaintext port and every health ping fails.
    protocol = "http" if domain.startswith("http://") else "https"
    kwargs: dict[str, Any] = {
        "domain": domain,
        "api_key": api_key,
        "protocol": protocol,
        "use_server_proxy": use_server_proxy,
        # The SDK sends no auth on execd calls, which the server proxy rejects; its
        # health and command clients merge these headers.
        "headers": {API_KEY_HEADER: api_key},
    }
    if request_timeout_sec is not None:
        kwargs["request_timeout"] = timedelta(seconds=request_timeout_sec)
    return sdk.ConnectionConfig(**kwargs)


def _octal_mode(path: Path) -> int:
    # WriteEntry.mode spells the octal permission bits in decimal digits (755 means rwxr-xr-x).
    return int(format(stat.S_IMODE(path.stat().st_mode), "o"))


def _clip(text: str, limit: int) -> tuple[bytes, bool]:
    data = text.encode()
    return data[:limit], len(data) > limit


class OpenSandboxMachine:
    """One OpenSandbox sandbox. Each command is a fresh execd process over shared files."""

    def __init__(self, sandbox: Any, spec: MachineSpec, resources: AsyncExitStack, sdk: OpenSandboxSDK):
        self.sandbox = sandbox
        self.spec = spec
        self.resources = resources
        self.sdk = sdk
        self._uids: dict[str, int] = {}
        self._closed = False

    async def _exec(self, script: str, *, uid: int | None = None, cwd: str | None = None, env=None, timeout=None):
        options: dict[str, Any] = {"working_directory": cwd, "envs": env or None, "timeout": timeout}
        if uid is not None:
            options["uid"] = uid
        return await self.sandbox.commands.run(script, opts=self.sdk.RunCommandOpts(**options))

    @staticmethod
    def _exit_code(execution: Any) -> int:
        # The SDK infers the exit code from execd's error value; that value is free-form
        # for kills and interpreter errors, so fall back to a generic failure.
        if execution.exit_code is not None:
            return execution.exit_code
        return 1 if execution.error is not None else 0

    async def _uid(self, user: str | None) -> int | None:
        """execd takes a numeric uid; resolve a name in the guest's account database."""
        if user is None:
            return None
        if user.isdecimal():
            return int(user)
        if user == "root":
            return 0
        if user not in self._uids:
            account = await self._exec(f"id -u {shlex.quote(user)}", uid=0, cwd="/")
            uid = "".join(message.text for message in account.logs.stdout).strip()
            if self._exit_code(account) or not uid.isdecimal():
                raise ValueError(f"Execution user {user} has no guest account")
            self._uids[user] = int(uid)
        return self._uids[user]

    async def run(self, command: Command) -> Result:
        if self._closed:
            raise RuntimeError("Machine is closed")
        if not command.argv:
            raise ValueError("Command argv is empty")
        if command.output_limit_bytes < 0:
            raise ValueError("Output limit must be nonnegative")
        prefix = f"/tmp/.shellbox-{uuid.uuid4().hex}" if command.stdin else None
        execution_id = None
        primary_error = None

        async def started(init: Any) -> None:
            nonlocal execution_id
            execution_id = init.id

        try:
            async with asyncio.timeout(command.timeout) as deadline:
                uid = await self._uid(command.user)
                stdin = "/dev/null"
                if prefix is not None:
                    stdin = f"{prefix}/in"
                    await self.sandbox.files.create_directories([self.sdk.WriteEntry(path=prefix, mode=700)])
                    await self.sandbox.files.write_files(
                        [self.sdk.WriteEntry(path=stdin, data=command.stdin, mode=600)]
                    )
                options: dict[str, Any] = {
                    "working_directory": command.cwd or self.spec.workdir or None,
                    "envs": {**self.spec.env, **command.env} or None,
                    # The client deadline expires first; the server limit only reaps a lost process.
                    "timeout": None if command.timeout is None else timedelta(seconds=math.ceil(command.timeout + 10)),
                }
                if uid is not None:
                    options["uid"] = uid
                execution = await self.sandbox.commands.run(
                    f"{shlex.join(command.argv)} < {shlex.quote(stdin)}",
                    opts=self.sdk.RunCommandOpts(**options),
                    handlers=self.sdk.ExecutionHandlers(on_init=started),
                )
            stdout, stdout_truncated = _clip(
                "".join(message.text for message in execution.logs.stdout), command.output_limit_bytes
            )
            stderr, stderr_truncated = _clip(
                "".join(message.text for message in execution.logs.stderr), command.output_limit_bytes
            )
            return Result(
                self._exit_code(execution), stdout, stderr, stdout_truncated, stderr_truncated, ExitReason.EXITED
            )
        except TimeoutError as error:
            if not deadline.expired():
                primary_error = error
                raise
            if execution_id is not None:
                try:
                    async with asyncio.timeout(INTERRUPT_TIMEOUT):
                        await self.sandbox.commands.interrupt(execution_id)
                except Exception as interrupt_error:
                    await self.close()
                    raise RuntimeError("Cannot interrupt the OpenSandbox command") from interrupt_error
            return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
        except asyncio.CancelledError:
            await self.close()
            raise
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if prefix is not None and not self._closed:
                try:
                    async with asyncio.timeout(INTERRUPT_TIMEOUT):
                        removed = await self._exec(f"rm -rf {shlex.quote(prefix)}", uid=0, cwd="/")
                    if self._exit_code(removed):
                        raise RuntimeError("rm failed")
                except asyncio.CancelledError:
                    await self.close()
                    raise
                except Exception as error:
                    await self.close()
                    if primary_error is None:
                        raise RuntimeError("Cannot remove OpenSandbox command files") from error
                    primary_error.add_note(f"OpenSandbox command cleanup failed: {error}")

    async def upload(self, source: Path, target: str) -> None:
        if self._closed:
            raise RuntimeError("Machine is closed")
        WriteEntry = self.sdk.WriteEntry
        if source.is_dir():
            directories = {target}
            files = []
            for path in source.rglob("*"):
                remote = str(PurePosixPath(target) / path.relative_to(source).as_posix())
                if path.is_dir():
                    directories.add(remote)
                    continue
                directories.add(str(PurePosixPath(remote).parent))
                files.append(WriteEntry(path=remote, data=path.read_bytes(), mode=_octal_mode(path)))
            await self.sandbox.files.create_directories(
                [WriteEntry(path=path, mode=755) for path in sorted(directories)]
            )
            if files:
                await self.sandbox.files.write_files(files)
            return
        parent = str(PurePosixPath(target).parent)
        if parent not in (".", "/"):
            await self.sandbox.files.create_directories([WriteEntry(path=parent, mode=755)])
        await self.sandbox.files.write_files(
            [WriteEntry(path=target, data=source.read_bytes(), mode=_octal_mode(source))]
        )

    async def _is_directory(self, path: str) -> bool:
        probe = await self._exec(f"test -d {shlex.quote(path)}", uid=0, cwd="/")
        return self._exit_code(probe) == 0

    async def download(self, source: str, target: Path) -> None:
        if self._closed:
            raise RuntimeError("Machine is closed")
        if not await self._is_directory(source):
            data = await self.sandbox.files.read_bytes(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return
        target.mkdir(parents=True, exist_ok=True)
        for entry in await self.sandbox.files.search(self.sdk.SearchEntry(path=source, pattern="**")):
            relative = PurePosixPath(entry.path).relative_to(PurePosixPath(source))
            local = target if relative == PurePosixPath(".") else target / relative
            kind = str(getattr(entry, "entry_type", "") or "").lower()
            is_directory = kind.endswith("directory")
            if not is_directory and not kind.endswith("file"):
                is_directory = await self._is_directory(entry.path)
            if is_directory:
                local.mkdir(parents=True, exist_ok=True)
                continue
            try:
                data = await self.sandbox.files.read_bytes(entry.path)
            except Exception as error:
                # Dangling symlinks, sockets, and FIFOs cannot be read; keep the rest of the tree.
                logger.warning("Skipping unreadable OpenSandbox entry %s: %s", entry.path, error)
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.resources.aclose()


class OpenSandboxMachineFactory:
    """Create an OpenSandbox sandbox from a prebuilt registry image."""

    backend: OpenSandboxBackend = OpenSandboxBackend.OPENSANDBOX

    def __init__(
        self,
        *,
        domain: str,
        api_key: str,
        use_server_proxy: bool = False,
        request_timeout_sec: float | None = None,
        ttl_minutes: int = DEFAULT_SANDBOX_TTL_MINUTES,
        create_timeout: float = 600,
        network_policy: OpenSandboxNetworkPolicy | None = None,
        image_auth: tuple[str, str] | None = None,
        health_check_interval: float = 2.0,
        retry_interval: float = 1.0,
        sdk_loader: Callable[[], OpenSandboxSDK] = load_sdk,
    ):
        self.domain = domain
        self.api_key = api_key
        self.use_server_proxy = use_server_proxy
        self.request_timeout_sec = request_timeout_sec
        self.ttl_minutes = ttl_minutes
        self.create_timeout = create_timeout
        self.network_policy = network_policy
        self.image_auth = image_auth
        self.health_check_interval = health_check_interval
        self.retry_interval = retry_interval
        self.sdk_loader = sdk_loader

    def _network(self, sdk: OpenSandboxSDK, spec: MachineSpec) -> Any | None:
        if self.network_policy is not None:
            return self.network_policy.build(sdk)
        if spec.network is NetworkPolicy.DENY:
            return sdk.NetworkPolicy(defaultAction="deny", egress=[])
        return None

    async def _sandbox(self, sdk: OpenSandboxSDK, spec: MachineSpec, image: Any) -> Any:
        resource = {}
        if spec.cpus is not None:
            resource["cpu"] = str(spec.cpus)
        if spec.memory_mb is not None:
            resource["memory"] = f"{spec.memory_mb}Mi"
        if spec.gpus:
            resource["gpu"] = str(spec.gpus)
        # OpenSandbox runtimes expose no per-sandbox disk limit, so storage_mb is not sent.
        backoff = ExponentialBackoff(initial=self.retry_interval, maximum=10 * self.retry_interval)
        for attempt in range(CREATE_ATTEMPTS):
            try:
                return await sdk.Sandbox.create(
                    image,
                    timeout=timedelta(minutes=self.ttl_minutes),
                    env=dict(spec.env),
                    resource=resource,
                    network_policy=self._network(sdk, spec),
                    connection_config=connection_config(
                        sdk,
                        domain=self.domain,
                        api_key=self.api_key,
                        use_server_proxy=self.use_server_proxy,
                        request_timeout_sec=self.request_timeout_sec,
                    ),
                    # Readiness is polled below so a Failed sandbox ends the wait with its reason.
                    skip_health_check=True,
                )
            except Exception as error:
                if type(error).__name__ not in TRANSIENT_SDK_EXCEPTIONS or attempt == CREATE_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(backoff.next_interval())
        raise AssertionError("unreachable")

    async def _wait_until_ready(self, sandbox: Any) -> None:
        while True:
            try:
                info = await sandbox.get_info()
            except Exception:
                info = None
            status = getattr(info, "status", None)
            state = str(getattr(status, "state", "") or "")
            if state in TERMINAL_STATES:
                raise RuntimeError(
                    f"OpenSandbox sandbox {sandbox.id} entered {state} before becoming ready: "
                    f"{getattr(status, 'message', '') or '(no message)'}"
                )
            try:
                if await sandbox.is_healthy():
                    return
            except Exception:
                # A failed probe means not ready yet; the startup deadline bounds the wait.
                pass
            await asyncio.sleep(self.health_check_interval)

    async def create(self, spec: MachineSpec) -> OpenSandboxMachine:
        if not isinstance(spec.source, RegistryImage | DockerImage):
            raise UnsupportedMachineSpec("OpenSandbox requires a prebuilt registry image")
        sdk = self.sdk_loader()
        image: Any = spec.source.reference
        if self.image_auth is not None:
            username, password = self.image_auth
            image = sdk.SandboxImageSpec(image=image, auth=sdk.SandboxImageAuth(username=username, password=password))
        async with AsyncExitStack() as lifetime:
            timeout = self.create_timeout if spec.startup_timeout is None else spec.startup_timeout
            async with asyncio.timeout(timeout):
                sandbox = await self._sandbox(sdk, spec, image)
                # Exit callbacks run in reverse: kill the sandbox, then release its transport.
                lifetime.push_async_callback(sandbox.close)
                lifetime.push_async_callback(sandbox.kill)
                logger.info("OpenSandbox sandbox %s created for %s", sandbox.id, spec.source.reference)
                try:
                    await self._wait_until_ready(sandbox)
                except TimeoutError:
                    logger.warning("OpenSandbox sandbox %s was not ready within %ss", sandbox.id, timeout)
                    raise
            machine = OpenSandboxMachine(sandbox, spec, lifetime, sdk)
            if spec.workdir:
                result = await machine.run(Command(("mkdir", "-p", spec.workdir), cwd="/", user="root"))
                if result.exit_code:
                    raise RuntimeError(f"Failed to create workdir {spec.workdir}: {result.stderr!r}")
            machine.resources = lifetime.pop_all()
            return machine
