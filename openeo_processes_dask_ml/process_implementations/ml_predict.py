import xarray as xr

from .data_model import MLModel


def ml_predict(data: xr.DataArray, model: MLModel) -> xr.DataArray:
    out = model.run_model(data)
    return out
