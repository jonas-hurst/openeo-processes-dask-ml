import json
import os
import re
from typing import Any

import requests
import requests.exceptions
from stac_validator.validate import StacValidate

import pystac

AVAILABLE_ML_FRAMEWORKS: list[str] = []

from .data_model import MLModel

try:
    from .data_model import ONNXModel

    AVAILABLE_ML_FRAMEWORKS.append("ONNX")
except ModuleNotFoundError:
    pass
try:
    from .data_model import TorchModel

    AVAILABLE_ML_FRAMEWORKS.append("PyTorch")
except ModuleNotFoundError:
    pass


def _load_stac_from_remote(uri: str) -> dict[str, Any]:
    # fetch STAC Item
    r = requests.get(uri)
    if r.status_code != 200:
        raise requests.exceptions.HTTPError(
            "Error while fetching STAC Item from URI: "
            "Server did not respond with status code 200"
        )

    try:
        stac = r.json()
    except requests.exceptions.JSONDecodeError:
        raise Exception("The provided URI does not point to a valid JSON file")

    return stac


def _load_stac_from_local(uri: str) -> dict[str, Any]:
    if not os.path.exists(uri):
        raise Exception(f"Could not locate file for the URI provided: {uri}")

    with open(uri) as file:
        try:
            stac = json.load(file)
        except json.decoder.JSONDecodeError:
            raise Exception("The provided URI does not point to a valid JSON file")

        return stac


def load_ml_model(uri: str, model_asset: str = None) -> MLModel:
    if type(uri) is not str:
        raise ValueError("Type of URI parameter must be a string.")

    if uri.startswith("http://") or uri.startswith("https://"):
        # uri is an url that points to a STAC
        stac = _load_stac_from_remote(uri)
    else:
        # assume uri points to a local file
        stac = _load_stac_from_local(uri)

    # check if downloaded JSON is valid STAC
    stac_validator = StacValidate()
    stac_valid = stac_validator.validate_dict(stac)
    if not stac_valid:
        raise Exception("The provided URI does not point to a valid STAC-Item")

    # check if downloaded JSON is valid STAC Item
    stac_type = stac["type"]
    if stac_type != "Feature":
        raise Exception("The provided URI does not point to a STAC-Item.")

    # Check if downloaded STAC Item implements the STAC:MLM extension
    extensions = stac["stac_extensions"]
    regex = (
        r"^https:\/\/stac-extensions\.github\.io\/mlm\/v(\d+\.){0,2}\d*\/schema\.json$"
    )
    pattern = re.compile(regex)
    follows_mlm = any(pattern.match(s) for s in extensions)
    if not follows_mlm:
        raise Exception(
            "The provided STAC Item does not implement the STAC:MLM standard"
        )

    mlm_item = pystac.Item.from_dict(stac)

    # check if the item's extensions are valid (e
    try:
        mlm_item.validate()
    except pystac.errors.STACValidationError:
        raise Exception(
            "The provided STAC Item does not implement STAC:MLM extension correctly."
        )

    # todo: mlm:framework could be in asset

    # Check if model runtime is supported (ONNX!, torch? tf?)
    ml_framework = mlm_item.ext.mlm.framework

    if ml_framework not in AVAILABLE_ML_FRAMEWORKS:
        raise Exception(
            f"The ML framework {ml_framework} as required by the provided STAC:MLM Item"
            f"is not supported by this backend. Supported backends: "
            f"{', '.join(AVAILABLE_ML_FRAMEWORKS)}"
        )

    if ml_framework == "ONNX":
        model_object = ONNXModel(mlm_item)
    elif ml_framework == "PyTorch":
        model_object = TorchModel(mlm_item)
    else:
        raise Exception(f"{ml_framework} runtime is not supported.")

    # download model
    # question: Download here, or when mdoel is actually executed?
    # Answer: Here, so we dont do image preprocessing steps unnecessarily in case the download fails
    # question: where to download the model to?

    # construct model from object

    return model_object
