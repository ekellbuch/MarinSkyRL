import subprocess
import sys


def test_ray_worker_setup_installs_stock_asyncio_without_loading_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio
import os
import sys

import skyrl_train_worker_setup

skyrl_train_worker_setup.force_stock_asyncio_in_worker()
skyrl_train_worker_setup.force_stock_asyncio_in_worker()

assert os.environ["UV_USE_IO_URING"] == "0"
assert isinstance(asyncio.get_event_loop_policy(), asyncio.DefaultEventLoopPolicy)

loop = asyncio.new_event_loop()
assert isinstance(loop, asyncio.SelectorEventLoop)
loop.close()

try:
    import uvloop
except ImportError:
    pass
else:
    loop = uvloop.new_event_loop()
    assert isinstance(loop, asyncio.SelectorEventLoop)
    loop.close()
    uvloop.install()
    assert isinstance(asyncio.get_event_loop_policy(), asyncio.DefaultEventLoopPolicy)

assert "torch" not in sys.modules
print("ok")
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "ok"
