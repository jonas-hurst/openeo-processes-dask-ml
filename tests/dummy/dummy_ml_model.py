import numpy as np
import xarray as xr

from openeo_processes_dask_ml.process_implementations.data_model import MLModel


class DummyMLModel(MLModel):
    """
    This is a dummy model class that is used only for testing purposes
    """

    def init_model_for_prediction(self):
        pass

    def uninit_model_after_prediction(self):
        pass

    def create_model_object(self, filepath: str):
        pass

    def execute_model(self, batch: xr.DataArray) -> xr.DataArray:
        out_shape = self.model_metadata.output[0].result.shape
        out_dims = self.model_metadata.output[0].result.dim_order
        out_dtype = self.model_metadata.output[0].result.data_type

        out_dc_shape = [*out_shape]

        # replace batch dim -1 with actual batch number from input
        if "batch" in out_dims and out_shape[out_dims.index("batch")] == -1:
            in_batch_idx = self.input.input.dim_order.index("batch")
            n_batches = batch.shape[in_batch_idx]

            out_batch_idx = out_dims.index("batch")
            out_dc_shape[out_batch_idx] = n_batches

        print(out_dc_shape)

        r = xr.DataArray(
            np.random.random(out_dc_shape).astype(out_dtype), dims=out_dims
        )
        return r
