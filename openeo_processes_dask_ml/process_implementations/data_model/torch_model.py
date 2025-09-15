import numpy as np
import pystac
import torch

from .data_model import MLModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TorchModel(MLModel):
    def __init__(self, stac_item: pystac.Item, model_asset_name: str = None):
        MLModel.__init__(self, stac_item, model_asset_name)

        self._model_on_device = None

    def create_model_object(self, filepath: str):
        at = self.model_asset_metadata.artifact_type
        if at == "torch.jit.save" or at.lower() == "torchscript":
            self._model_object = torch.jit.load(filepath)
        elif at == "torch.export.save":
            self._model_object = torch.export.load(filepath)
        else:
            raise NotImplemented(
                f"Importing Torch models with artifact type {at} is not supported.\n"
                f"Use a model with artifact type torch.jit.save or torch.export.save "
                f"instead"
            )

    def init_model_for_prediction(self):
        self._model_on_device = self._model_object.to(DEVICE)
        self._model_on_device.eval()

    def uninit_model_after_prediction(self):
        self._model_on_device = self._model_on_device.to("cpu")
        del self._model_on_device
        self._model_on_device = None
        torch.cuda.empty_cache()

    def execute_model(self, batch: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(batch).to(DEVICE)
        with torch.no_grad():
            out = self._model_on_device(tensor)

        out_postproc = self.postprocess_datacube_expression(out)
        if out_postproc.device.type != "cpu":
            out_postproc = out_postproc.cpu()
        out_cube = out_postproc.numpy()

        return out_cube
