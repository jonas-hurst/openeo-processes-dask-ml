import os

CACHE_DIR = os.environ.get("OPD_ML_CACHE_DIR", "./cache")
MODEL_CACHE_DIR = os.environ.get("OPD_ML_MODEL_CACHE_DIR", f"{CACHE_DIR}/model_cache")
DATACUBE_CACHE_DIR = os.environ.get(
    "OPD_ML_DATACUBE_CACHE_DIR", f"{CACHE_DIR}/datacubes"
)

_USE_GPU = os.environ.get("OPD_ML_USE_GPU", "true")  # allowed values: "True", "False"
if _USE_GPU.lower() == "true":
    USE_GPU = True
elif _USE_GPU.lower() == "false":
    USE_GPU = False
else:
    raise ValueError('Env OPD_ML_USE_GPU only allows values "True" and "False"')

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
