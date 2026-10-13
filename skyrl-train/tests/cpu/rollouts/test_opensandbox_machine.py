"""OpenSandbox machine factory against an in-memory SDK; no server or network."""

import asyncio
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError
from shellbox.image import DockerfileSource, RegistryImage
from shellbox.machine import Command, ExitReason, MachineSpec, NetworkPolicy, UnsupportedMachineSpec
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.opensandbox_machine import (
    OpenSandboxMachineFactory,
    OpenSandboxNetworkMode,
    OpenSandboxNetworkPolicy,
    OpenSandboxSDK,
    connection_config,
)

IMAGE = "registry.test/task@sha256:" + "a" * 64
DOMAIN = "http://sandbox.test:8080"


def record(**kwargs):
    return SimpleNamespace(**kwargs)


def output(text):
    return [SimpleNamespace(text=text)] if text else []


def execution(stdout="", stderr="", exit_code=0, error=None):
    return SimpleNamespace(
        exit_code=exit_code, error=error, logs=SimpleNamespace(stdout=output(stdout), stderr=output(stderr))
    )


class FakeCommands:
    def __init__(self, sandbox):
        self.sandbox = sandbox
        self.calls = []
        self.interrupted = []
        self.respond = lambda script, opts: execution()
        self.interrupt_error = None

    async def run(self, script, *, opts, handlers=None):
        self.calls.append((script, opts))
        if script.startswith("mkdir -p") and self.sandbox.workdir_error:
            return execution(stderr="read-only", exit_code=1)
        if handlers is not None:
            await handlers.on_init(SimpleNamespace(id=f"exec-{len(self.calls)}"))
        result = self.respond(script, opts)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    async def interrupt(self, execution_id):
        if self.interrupt_error is not None:
            raise self.interrupt_error
        self.interrupted.append(execution_id)


class FakeFiles:
    def __init__(self):
        self.data = {}
        self.modes = {}
        self.directories = []
        self.entries = []

    async def create_directories(self, entries):
        self.directories.extend((entry.path, entry.mode) for entry in entries)

    async def write_files(self, entries):
        for entry in entries:
            self.data[entry.path] = entry.data
            self.modes[entry.path] = entry.mode

    async def read_bytes(self, path):
        return self.data[path]

    async def search(self, entry):
        return self.entries


class FakeSandbox:
    def __init__(self, states=("Running",), healthy_after=0, workdir_error=False):
        self.id = "sbx-1"
        self.states = list(states)
        self.health_checks = 0
        self.healthy_after = healthy_after
        self.workdir_error = workdir_error
        self.commands = FakeCommands(self)
        self.files = FakeFiles()
        self.events = []

    async def get_info(self):
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return SimpleNamespace(status=SimpleNamespace(state=state, message="image pull 502"))

    async def is_healthy(self):
        self.health_checks += 1
        return self.healthy_after is not None and self.health_checks > self.healthy_after

    async def kill(self):
        self.events.append("kill")

    async def close(self):
        self.events.append("close")


class FakeSandboxClass:
    def __init__(self, sandbox, failures=()):
        self.sandbox = sandbox
        self.failures = list(failures)
        self.calls = []

    async def create(self, image, **kwargs):
        self.calls.append((image, kwargs))
        if self.failures:
            raise self.failures.pop(0)
        return self.sandbox


def fake_sdk(sandbox_class):
    return OpenSandboxSDK(
        Sandbox=sandbox_class,
        ConnectionConfig=record,
        RunCommandOpts=record,
        ExecutionHandlers=record,
        SearchEntry=record,
        WriteEntry=record,
        NetworkPolicy=record,
        NetworkRule=record,
        SandboxImageAuth=record,
        SandboxImageSpec=record,
    )


def factory_for(sandbox_class, **kwargs):
    kwargs.setdefault("health_check_interval", 0)
    kwargs.setdefault("retry_interval", 0.001)
    return OpenSandboxMachineFactory(
        domain=DOMAIN, api_key="test-key", sdk_loader=lambda: fake_sdk(sandbox_class), **kwargs
    )


async def machine_for(sandbox=None, *, spec=None, **kwargs):
    sandbox = sandbox or FakeSandbox()
    machine = await factory_for(FakeSandboxClass(sandbox), **kwargs).create(
        spec or MachineSpec(source=RegistryImage(IMAGE), workdir="")
    )
    sandbox.commands.calls.clear()
    return sandbox, machine


@pytest.mark.asyncio
async def test_create_sends_image_resources_policy_ttl_and_waits_for_health():
    sandbox = FakeSandbox(states=("Pending", "Running"), healthy_after=1)
    sandbox_class = FakeSandboxClass(sandbox)
    spec = MachineSpec(
        source=RegistryImage(IMAGE), workdir="/app", env={"A": "1"}, cpus=2, memory_mb=4096, gpus=1, storage_mb=8192
    )
    machine = await factory_for(
        sandbox_class, use_server_proxy=True, request_timeout_sec=600, ttl_minutes=90, image_auth=("user", "token")
    ).create(spec)

    image, kwargs = sandbox_class.calls[0]
    assert image.image == IMAGE and (image.auth.username, image.auth.password) == ("user", "token")
    assert kwargs["timeout"] == timedelta(minutes=90)
    assert kwargs["env"] == {"A": "1"}
    assert kwargs["resource"] == {"cpu": "2", "memory": "4096Mi", "gpu": "1"}
    assert kwargs["network_policy"].defaultAction == "deny" and kwargs["network_policy"].egress == []
    assert kwargs["skip_health_check"] is True
    connection = kwargs["connection_config"]
    assert connection.protocol == "http" and connection.use_server_proxy is True
    assert connection.headers == {"OPEN-SANDBOX-API-KEY": "test-key"}
    assert connection.request_timeout == timedelta(seconds=600)
    assert sandbox.health_checks == 2
    script, opts = sandbox.commands.calls[0]
    assert script == "mkdir -p /app < /dev/null" and opts.uid == 0 and opts.working_directory == "/"

    await machine.close()
    await machine.close()
    assert sandbox.events == ["kill", "close"]


@pytest.mark.asyncio
async def test_create_retries_transient_sdk_errors_only():
    class SandboxApiException(Exception):
        pass

    sandbox = FakeSandbox()
    sandbox_class = FakeSandboxClass(sandbox, failures=[SandboxApiException("503")])
    machine = await factory_for(sandbox_class).create(MachineSpec(source=RegistryImage(IMAGE), workdir=""))
    assert len(sandbox_class.calls) == 2
    await machine.close()

    class InvalidArgumentException(Exception):
        pass

    sandbox_class = FakeSandboxClass(FakeSandbox(), failures=[InvalidArgumentException("bad image")])
    with pytest.raises(InvalidArgumentException):
        await factory_for(sandbox_class).create(MachineSpec(source=RegistryImage(IMAGE), workdir=""))
    assert len(sandbox_class.calls) == 1


@pytest.mark.asyncio
async def test_failed_sandbox_reports_reason_and_is_torn_down():
    sandbox = FakeSandbox(states=("Failed",), healthy_after=None)
    with pytest.raises(RuntimeError, match="entered Failed before becoming ready: image pull 502"):
        await factory_for(FakeSandboxClass(sandbox)).create(MachineSpec(source=RegistryImage(IMAGE), workdir=""))
    assert sandbox.events == ["kill", "close"]


@pytest.mark.asyncio
async def test_startup_deadline_tears_down_unready_sandbox():
    sandbox = FakeSandbox(healthy_after=None)
    with pytest.raises(TimeoutError):
        await factory_for(FakeSandboxClass(sandbox)).create(
            MachineSpec(source=RegistryImage(IMAGE), workdir="", startup_timeout=0.05)
        )
    assert sandbox.events == ["kill", "close"]


@pytest.mark.asyncio
async def test_workdir_failure_tears_down_sandbox():
    sandbox = FakeSandbox(workdir_error=True)
    with pytest.raises(RuntimeError, match="Failed to create workdir"):
        await factory_for(FakeSandboxClass(sandbox)).create(MachineSpec(source=RegistryImage(IMAGE), workdir="/app"))
    assert sandbox.events == ["kill", "close"]


@pytest.mark.asyncio
async def test_build_contexts_are_rejected_before_any_request(tmp_path):
    sandbox_class = FakeSandboxClass(FakeSandbox())
    source = DockerfileSource(tmp_path, tmp_path / "Dockerfile")
    with pytest.raises(UnsupportedMachineSpec, match="prebuilt registry image"):
        await factory_for(sandbox_class).create(MachineSpec(source=source))
    assert sandbox_class.calls == []


@pytest.mark.asyncio
async def test_run_maps_cwd_env_exit_code_and_output_limit():
    sandbox, machine = await machine_for(spec=MachineSpec(source=RegistryImage(IMAGE), workdir="/app", env={"A": "1"}))
    sandbox.commands.calls.clear()
    sandbox.commands.respond = lambda script, opts: execution(stdout="abcdef", stderr="err", exit_code=3)

    result = await machine.run(Command(("echo", "a b"), env={"B": "2"}, timeout=5, output_limit_bytes=4))

    script, opts = sandbox.commands.calls[0]
    assert script == "echo 'a b' < /dev/null"
    assert opts.working_directory == "/app" and opts.envs == {"A": "1", "B": "2"}
    assert opts.timeout == timedelta(seconds=15) and not hasattr(opts, "uid")
    assert (result.exit_code, result.stdout, result.stdout_truncated) == (3, b"abcd", True)
    assert (result.stderr, result.stderr_truncated, result.reason) == (b"err", False, ExitReason.EXITED)

    sandbox.commands.respond = lambda script, opts: execution(exit_code=None, error=SimpleNamespace(value="killed"))
    assert (await machine.run(Command(("true",)))).exit_code == 1
    await machine.close()


@pytest.mark.asyncio
async def test_run_streams_stdin_through_a_private_file_and_removes_it():
    sandbox, machine = await machine_for()
    result = await machine.run(Command(("cat",), stdin=b"payload"))

    assert result.exit_code == 0
    [(path, data)] = sandbox.files.data.items()
    assert data == b"payload" and sandbox.files.modes[path] == 600
    prefix = path.removesuffix("/in")
    assert (prefix, 700) in sandbox.files.directories
    scripts = [script for script, _ in sandbox.commands.calls]
    assert scripts == [f"cat < {path}", f"rm -rf {prefix}"]
    await machine.close()


@pytest.mark.asyncio
async def test_named_users_resolve_to_guest_uids():
    sandbox, machine = await machine_for()
    sandbox.commands.respond = lambda script, opts: execution(stdout="1001\n" if script.startswith("id -u") else "")

    await machine.run(Command(("whoami",), user="agent"))
    await machine.run(Command(("whoami",), user="agent"))
    await machine.run(Command(("whoami",), user="root"))

    uids = [(script, getattr(opts, "uid", None)) for script, opts in sandbox.commands.calls]
    assert uids == [
        ("id -u agent", 0),
        ("whoami < /dev/null", 1001),
        ("whoami < /dev/null", 1001),
        ("whoami < /dev/null", 0),
    ]
    sandbox.commands.respond = lambda script, opts: execution(exit_code=1)
    with pytest.raises(ValueError, match="nobody-here has no guest account"):
        await machine.run(Command(("whoami",), user="nobody-here"))
    await machine.close()


@pytest.mark.asyncio
async def test_timeout_interrupts_the_command_and_keeps_the_machine():
    sandbox, machine = await machine_for()
    sandbox.commands.respond = lambda script, opts: asyncio.Event().wait()

    result = await machine.run(Command(("sleep", "100"), timeout=0.05))

    assert (result.exit_code, result.reason) == (None, ExitReason.TIMED_OUT)
    assert sandbox.commands.interrupted == ["exec-1"]
    assert sandbox.events == []
    await machine.close()
    assert sandbox.events == ["kill", "close"]


@pytest.mark.asyncio
async def test_failed_interrupt_closes_the_machine():
    sandbox, machine = await machine_for()
    sandbox.commands.respond = lambda script, opts: asyncio.Event().wait()
    sandbox.commands.interrupt_error = ConnectionError("execd gone")

    with pytest.raises(RuntimeError, match="Cannot interrupt"):
        await machine.run(Command(("sleep", "100"), timeout=0.05))
    assert sandbox.events == ["kill", "close"]
    with pytest.raises(RuntimeError, match="closed"):
        await machine.run(Command(("true",)))


@pytest.mark.asyncio
async def test_upload_and_download_files_and_directories(tmp_path):
    sandbox, machine = await machine_for()
    script = tmp_path / "run.sh"
    script.write_text("echo hi\n")
    script.chmod(0o755)
    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    (tree / "sub/data.txt").write_text("data")
    (tree / "sub/data.txt").chmod(0o644)

    await machine.upload(script, "/opt/bin/run.sh")
    await machine.upload(tree, "/srv/tree")

    assert sandbox.files.data["/opt/bin/run.sh"] == b"echo hi\n" and sandbox.files.modes["/opt/bin/run.sh"] == 755
    assert sandbox.files.data["/srv/tree/sub/data.txt"] == b"data"
    assert sandbox.files.modes["/srv/tree/sub/data.txt"] == 644
    assert {"/opt/bin", "/srv/tree", "/srv/tree/sub"} <= {path for path, _ in sandbox.files.directories}

    sandbox.commands.respond = lambda script, opts: execution(exit_code=0 if script == "test -d /srv/tree" else 1)
    sandbox.files.entries = [
        SimpleNamespace(path="/srv/tree/sub", entry_type="directory"),
        SimpleNamespace(path="/srv/tree/sub/data.txt", entry_type="file"),
        SimpleNamespace(path="/srv/tree/dangling", entry_type="symlink"),
    ]
    await machine.download("/srv/tree", tmp_path / "out")
    await machine.download("/opt/bin/run.sh", tmp_path / "single/run.sh")

    assert (tmp_path / "out/sub/data.txt").read_text() == "data"
    assert not (tmp_path / "out/dangling").exists()
    assert (tmp_path / "single/run.sh").read_bytes() == b"echo hi\n"
    await machine.close()


def test_network_policies_map_to_domain_egress_rules():
    sdk = fake_sdk(None)
    allow = OpenSandboxNetworkPolicy(OpenSandboxNetworkMode.DOMAIN_ALLOW_LIST, "pypi.org, *.github.com").build(sdk)
    assert allow.defaultAction == "deny"
    assert [(rule.action, rule.target) for rule in allow.egress] == [("allow", "pypi.org"), ("allow", "*.github.com")]
    assert OpenSandboxNetworkPolicy("block_all").build(sdk).egress == []
    assert OpenSandboxNetworkPolicy("unrestricted").build(sdk) is None
    with pytest.raises(ValueError, match="network_allow_list is not supported"):
        OpenSandboxNetworkPolicy("network_allow_list", "10.0.0.0/8")
    with pytest.raises(ValueError, match="requires an allow list"):
        OpenSandboxNetworkPolicy("domain_allow_list")


@pytest.mark.asyncio
async def test_spec_network_selects_policy_when_none_is_configured():
    sandbox_class = FakeSandboxClass(FakeSandbox())
    factory = factory_for(sandbox_class)
    machine = await factory.create(MachineSpec(source=RegistryImage(IMAGE), workdir="", network=NetworkPolicy.ALLOW))
    assert sandbox_class.calls[0][1]["network_policy"] is None
    await machine.close()


def test_scheme_less_domain_uses_https():
    connection = connection_config(
        fake_sdk(None), domain="sandbox.test", api_key="k", use_server_proxy=False, request_timeout_sec=None
    )
    assert connection.protocol == "https" and not hasattr(connection, "request_timeout")


def test_real_sdk_accepts_the_request_shapes():
    pytest.importorskip("opensandbox")
    from skyrl_train.rollouts.opensandbox_machine import load_sdk

    sdk = load_sdk()
    connection = connection_config(sdk, domain=DOMAIN, api_key="k", use_server_proxy=True, request_timeout_sec=600)
    assert connection.headers["OPEN-SANDBOX-API-KEY"] == "k" and connection.protocol == "http"
    policy = OpenSandboxNetworkPolicy("domain_allow_list", "pypi.org").build(sdk)
    assert policy.default_action == "deny" and policy.egress[0].target == "pypi.org"
    opts = sdk.RunCommandOpts(working_directory="/app", envs={"A": "1"}, timeout=timedelta(seconds=5), uid=0)
    assert opts.uid == 0
    assert sdk.WriteEntry(path="/tmp/x", data=b"x", mode=600).mode == 600
    assert sdk.SandboxImageSpec(image=IMAGE, auth=sdk.SandboxImageAuth(username="u", password="p")).image == IMAGE
    assert sdk.SearchEntry(path="/srv", pattern="**").pattern == "**"


def opensandbox_settings(monkeypatch, **overrides):
    monkeypatch.setenv("TEST_OPENSANDBOX_KEY", "test-key")
    harbor = {
        "environment_type": "opensandbox",
        "ttl_minutes": 45,
        "env_network_policy": {"mode": "domain_allow_list", "value": "pypi.org"},
        "opensandbox": {
            "domain": DOMAIN,
            "api_key_env": "TEST_OPENSANDBOX_KEY",
            "use_server_proxy": True,
            "request_timeout_sec": 600,
            "registry_username_env": None,
            "registry_token_env": None,
        },
    }
    harbor.update(overrides)
    return HarborTaskSettings.from_config(OmegaConf.create({"harbor": harbor}))


def test_harbor_settings_select_opensandbox_factory(monkeypatch):
    settings = opensandbox_settings(monkeypatch)
    # The task worker routes every non-docker type to machine_factory.
    assert settings.environment.type is None
    assert settings.environment.import_path.endswith(":OpenSandboxMachineFactory")
    for phase in ("train", "eval"):
        factory = settings.machine_factory(phase=phase)
        assert isinstance(factory, OpenSandboxMachineFactory)
        assert factory.backend == "opensandbox"
        assert (factory.domain, factory.api_key, factory.use_server_proxy) == (DOMAIN, "test-key", True)
        assert (factory.request_timeout_sec, factory.ttl_minutes, factory.image_auth) == (600, 45, None)
        assert factory.network_policy == OpenSandboxNetworkPolicy("domain_allow_list", "pypi.org")


def test_registry_credentials_come_from_named_variables(monkeypatch):
    monkeypatch.setenv("TEST_REGISTRY_USER", "user")
    monkeypatch.setenv("TEST_REGISTRY_TOKEN", "token")
    settings = opensandbox_settings(monkeypatch)
    block = {**settings.opensandbox.model_dump(), "registry_username_env": "TEST_REGISTRY_USER"}
    block["registry_token_env"] = "TEST_REGISTRY_TOKEN"
    settings = opensandbox_settings(monkeypatch, opensandbox=block)
    assert settings.machine_factory().image_auth == ("user", "token")


def test_missing_opensandbox_key_variable_names_the_variable(monkeypatch):
    settings = opensandbox_settings(monkeypatch)
    monkeypatch.delenv("TEST_OPENSANDBOX_KEY")
    with pytest.raises(ValueError, match="Missing OpenSandbox credential environment variable: TEST_OPENSANDBOX_KEY"):
        settings.machine_factory()


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ({"domain": DOMAIN}, "api_key_env"),
        ({"domain": DOMAIN, "api_key_env": ""}, "api_key_env"),
        ({"api_key_env": "TEST_OPENSANDBOX_KEY"}, "domain"),
        ({"domain": DOMAIN, "api_key_env": "TEST_OPENSANDBOX_KEY", "api_key": "inline"}, "api_key"),
    ],
)
def test_opensandbox_settings_are_validated(monkeypatch, block, message):
    with pytest.raises(ValidationError, match=message):
        opensandbox_settings(monkeypatch, opensandbox=block)


def test_opensandbox_selection_and_settings_must_agree(monkeypatch):
    with pytest.raises(ValueError, match="requires harbor.opensandbox settings"):
        HarborTaskSettings.from_config(OmegaConf.create({"harbor": {"environment_type": "opensandbox"}}))
    with pytest.raises(ValueError, match="require environment_type opensandbox"):
        opensandbox_settings(monkeypatch, environment_type="daytona")


def test_unknown_and_unwired_backends_still_raise():
    with pytest.raises(ValueError, match="not-a-backend"):
        HarborTaskSettings.from_config(OmegaConf.create({"harbor": {"environment_type": "not-a-backend"}}))
    settings = HarborTaskSettings.from_config(OmegaConf.create({"harbor": {"environment_type": "modal"}}))
    with pytest.raises(ValueError, match="No Shellbox factory is configured for Harbor backend"):
        settings.machine_factory()


def test_upload_rejects_closed_machine(tmp_path):
    async def scenario():
        sandbox, machine = await machine_for()
        await machine.close()
        with pytest.raises(RuntimeError, match="closed"):
            await machine.upload(Path(tmp_path), "/x")

    asyncio.run(scenario())
