"""SkyRL training package."""

from skyrl_train.models import register_local_models


# AutoConfig is called in both the driver and Ray actors. Package import is the
# earliest common boundary, so checkpoints load without trust_remote_code.
register_local_models()
