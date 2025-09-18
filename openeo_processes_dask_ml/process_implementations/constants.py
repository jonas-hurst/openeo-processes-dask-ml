import os


def _get_boolean_env(env_name, default_value: bool) -> bool:
    env_value = os.environ.get(env_name, None)
    if env_value is None:
        return default_value
    if env_value.lower() == "true":
        return True
    elif env_value.lower() == "false":
        return False
    else:
        raise ValueError(
            f'Env {env_name} only allows values "True" and "False". '
            f'Currently set to "{env_value}".'
        )


CACHE_DIR = os.environ.get("OPD_ML_CACHE_DIR", "./cache")
MODEL_CACHE_DIR = os.environ.get("OPD_ML_MODEL_CACHE_DIR", f"{CACHE_DIR}/model_cache")
DATACUBE_CACHE_DIR = os.environ.get(
    "OPD_ML_DATACUBE_CACHE_DIR", f"{CACHE_DIR}/datacubes"
)

USE_GPU = _get_boolean_env("OPD_ML_USE_GPU", True)

S3_MODEL_REPO_ENDPOINT = os.environ.get("OPD_ML_S3_MODEL_REPO_ENDPOINT", None)
S3_MODEL_REPO_ACCESS_KEY_ID = os.environ.get("OPD_ML_S3_MODEL_REPO_ACCESS_KEY_ID", None)
S3_MODEL_REPO_SECRET_ACCESS_KEY = os.environ.get(
    "OPD_ML_S3_MODEL_REPO_SECRET_ACCESS_KEY", None
)

# make sure that both access_key_id and access_key are set
if (S3_MODEL_REPO_ACCESS_KEY_ID is None) != (S3_MODEL_REPO_SECRET_ACCESS_KEY is None):
    raise ValueError(
        "Either S3_MODEL_REPO_ACCESS_KEY_ID or S3_MODEL_REPO_SECRET_ACCESS_KEY is set "
        "not set. You must set either both, or not set any of the two."
    )
