import itertools
import logging
import os.path
from abc import ABC, abstractmethod

import dask.delayed
import numpy as np
import pystac
import xarray as xr
import xarray.core.coordinates
from dask import array as da
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionMismatch,
    DimensionMissing,
)
from pystac.extensions.mlm import (
    MLMExtension,
    ModelInput,
    ModelOutput,
    _AssetMLMExtension,
)

from openeo_processes_dask_ml.process_implementations.constants import MODEL_CACHE_DIR
from openeo_processes_dask_ml.process_implementations.exceptions import (
    ExpressionEvaluationException,
    LabelDoesNotExist,
    ReferenceSystemNotFound,
)
from openeo_processes_dask_ml.process_implementations.utils import (
    dim_utils,
    download_utils,
    epsg_utils,
    model_cache_utils,
    proc_expression_utils,
    scaling_utils,
)

logger = logging.getLogger(__name__)


class MLModel(ABC):
    _stac_item: pystac.Item
    _model_asset: pystac.Asset
    _input_index: int
    _output_index: int

    def __init__(
        self,
        stac_item: pystac.Item,
        model_asset_name: str = None,
        input_index: int = 0,
        output_index: int = 0,
    ):
        self._stac_item = stac_item
        self._model_asset = self._get_model_asset(model_asset_name)
        self._input_index = input_index
        self._output_index = output_index
        self._model_object = None

    @property
    def model_metadata(self) -> MLMExtension:
        # todo: account for if metadata is stored with the asset
        return MLMExtension.ext(self._stac_item)

    @property
    def model_asset_metadata(self) -> _AssetMLMExtension:
        """
        Get the asset metadata of the model asset.
        :return: asset metadata
        """
        return self._model_asset.ext.mlm

    @property
    def input(self) -> ModelInput:
        """
        Contains info on how the input to the ML model must look like
        :return:
        """
        return self.model_metadata.input[self._input_index]

    @property
    def output(self) -> ModelOutput:
        """
        Contains info on how the output of ML model will look like
        :return:
        """
        return self.model_metadata.output[self._output_index]

    def _get_model_asset(self, asset_name: str = None) -> pystac.Asset:
        """
        Determine which asset holds the model (has mlm:model in roles)
        Determines whcih asset to use if multiple assets with role mlm:models are found
        :param asset_name: model asset name (could be None)
        :return: model asset name
        """
        assets = self._stac_item.assets
        model_assets = {
            key: assets[key] for key in assets if "mlm:model" in assets[key].roles
        }

        # case 1: no assets with mlm:model role
        if not model_assets:
            raise Exception(
                "The given STAC Item does not have an asset with role mlm:model"
            )

        # case 2: asset_name is given
        if asset_name:
            if asset_name in model_assets:
                return model_assets[asset_name]
            else:
                raise Exception(
                    f"Provided STAC Item does not have an asset named {asset_name} "
                    f"which also lists mlm:model as its asset role"
                )

        # case 3: asset name is not given and there is only one mlm:model asset
        if len(model_assets) == 1:
            return next(iter(model_assets.values()))

        # case 4: multiple mlm:model exist but asset_name is not specified
        raise Exception(
            "Multiple assets with role=mlm:model are found in the provided STAC-Item. "
            "Please sepcify which one to use."
        )

    def _get_model(self) -> str:
        # model_asset = self._get_model_asset(asset_name)
        # url = self.model_asset_metadata.asset_href
        # url = model_asset.href
        url = self._model_asset.href

        # encode URL to directory name and file name
        model_dir_name = model_cache_utils.url_to_dir_string(url)
        model_file_name = model_cache_utils.url_to_dir_string(url.split("/")[-1], True)

        model_cache_dir = os.path.join(MODEL_CACHE_DIR, model_dir_name)
        model_cache_file = os.path.join(model_cache_dir, model_file_name)

        # check if model file has been downloaded to cache already
        if os.path.exists(model_cache_file):
            return model_cache_file

        # check if directory exists already in cache and create if not
        if not os.path.exists(model_cache_dir):
            os.makedirs(model_cache_dir)

        download_utils.download(url, model_cache_file)
        return model_cache_file

    def get_datacube_dimension_mapping(
        self, datacube: xr.DataArray
    ) -> list[None | tuple[str, int]]:
        """
        Maps the model input dimension names to datacube dimension names, as dimension
        names can sometimes differ, e.g. t -> time
        :param datacube: The datacube to map the dimeions agains
        :return: Tuple with dc-equivalent model input dimension names and their index
        """
        model_dims = self.model_metadata.input[0].input.dim_order
        dc_dims = datacube.dims

        dim_mapping = []
        for m_dim_name in model_dims:
            dc_dim_name = dim_utils.get_alternative_datacube_dim_name(
                datacube, m_dim_name
            )
            if dc_dim_name is None:
                dim_mapping.append(None)
            else:
                dim_mapping.append((dc_dim_name, dc_dims.index(dc_dim_name)))

        return dim_mapping

    def get_datacube_output_dimension_mapping(
        self, in_datacube: xr.DataArray
    ) -> list[str]:
        """
        Get model output dimension names, mapped to input datacube dimension names to
        resolve discrepancies (e.g. time <-> times)
        :param in_datacube:
        :return:
        """
        out_dims = self.model_metadata.output[0].result.dim_order
        in_dims = self.model_metadata.input[0].input.dim_order
        in_dims_map = [d[0] for d in self.get_datacube_dimension_mapping(in_datacube)]

        out_dims_map = []
        for out_dim in out_dims:
            if out_dim in in_dims:
                dim_idx = in_dims.index(out_dim)
                out_dims_map.append(in_dims_map[dim_idx])
            else:
                out_dims_map.append(out_dim)

        return out_dims_map

    def _check_dimensions_present_in_datacube(
        self, datacube: xr.DataArray, ignore_batch_dim: bool = False
    ) -> None:
        """
        Checkl whether the datacube contains all dimensions required by the model input
        :param datacube: The datacube to be checked
        :param ignore_batch_dim: Ignore a missing "batch" dimension in the datacube
        :raise DimensionMissing: Raised when a dimension requqired by the model input
        is missing
        :return: None
        """

        input_dims = self.model_metadata.input[0].input.dim_order
        dim_mapping = self.get_datacube_dimension_mapping(datacube)

        unmatched_dims = [input_dims[i] for i, d in enumerate(dim_mapping) if d is None]

        # check if all model input dimensions could be matched to dc dimensions
        # ignore batch dimension, we will take care of this later
        if ignore_batch_dim and "batch" in unmatched_dims:
            unmatched_dims.remove("batch")
        if any(unmatched_dims):
            raise DimensionMissing(
                f"Datacube is missing the following dimensions required by the "
                f"STAC-MLM Input: {', '.join(unmatched_dims)}"
            )

    def _check_datacube_dimension_size(
        self, datacube: xr.DataArray, ignore_batch_dim: bool = False
    ) -> None:
        """
        Check whether each datacube dimension is long enough to satisfy the model
        input requriements
        :param datacube: The datacube to be checked
        :param ignore_batch_dim: Ignore a missing "batch" dimension in the datacube
        :raise DimensionMismatch: raised when a datacube dimension has fewer
        coordinates than requried by the model input
        :return: None
        """

        input_dims = self.model_metadata.input[0].input.dim_order
        dim_mapping = self.get_datacube_dimension_mapping(datacube)

        input_shape = self.model_metadata.input[0].input.shape  # e.g. [-1, 3, 128, 128]
        dc_shape = datacube.shape  # e.g. (12, 1000, 1000, 5)

        # reorder dc_shape to match input_shape
        # xor: (a and not b) or (not a and b)
        dc_shape_reorder = [
            dc_shape[dim_mapping[i][1]]
            for i, inp_dim in enumerate(input_dims)
            if inp_dim != "batch" or (inp_dim == "batch" and not ignore_batch_dim)
        ]

        # ignore "batch" dimension for now, we will take care of that later
        input_shape_reorder = [
            d
            for i, d in zip(input_dims, input_shape)
            if i != "batch" or (i == "batch" and not ignore_batch_dim)
        ]

        # check whether dc shape is big enough to suffice input:
        # input size must be smaller than dc size in every input dimension
        for dc_dim_size, inp_dim_size in zip(dc_shape_reorder, input_shape_reorder):
            if inp_dim_size == -1:
                # -1 as input shape size means all values are allowed
                # e.g. batch=-1 means the models allows for arbitrary batch size
                continue
            if dc_dim_size >= inp_dim_size:
                continue
            raise DimensionMismatch(
                "The model input requires dimension DIM_NAME to have X values. "
                "The datacube only has Y values for dimension DIM_NAME."
            )

    def _check_datacube_bands(self, datacube: xr.DataArray):
        """
        Checks if the required input bands for the model are present in the provided
        `xarray.DataArray` datacube.
        This function verifies that all bands specified in the `model_metadata.input`
        are available within the `datacube`. It handles cases where bands might be
        directly present or need to be computed from other bands.
        :param datacube: The input data as an `xarray.DataArray`, expected to contain
        geospatial and spectral data.
        :raises DimensionMissing: If a 'bands' dimension is required by the model but
        not found in the datacube, or if the dimension is named unconventionally.
        :raises ValueError: If a band definition in `model_metadata` has either
        `format` or `expression` but not both, when a computation is expected.
        :raises LabelDoesNotExist: If any required band (that cannot be computed) is
        missing from the datacube.
        """
        input_bands = self.model_metadata.input[0].bands
        input_bands_str = [i if isinstance(i, str) else i.name for i in input_bands]

        # bands prorety not utilized, list is empty
        if not input_bands:
            return

        # possibilities how the "bands" dimension could be called
        band_dim_name = dim_utils.get_band_dim_name(datacube)

        # dc_bands = datacube.coords[band_dim_name]
        band_coords = datacube.coords[band_dim_name].values.tolist()
        dc_bands = dim_utils.get_dc_band_names(input_bands_str, band_coords)

        band_available_in_datacube: list[bool] = []
        bands_unavailable: list[str] = []
        for band in input_bands:
            if isinstance(band, str):
                if band in dc_bands:
                    band_available_in_datacube.append(True)
                else:
                    band_available_in_datacube.append(False)
                    bands_unavailable.append(band)

            else:
                # this means type(band) must be ModelInput
                band_name = band.name

                if band_name in dc_bands:
                    band_available_in_datacube.append(True)
                    continue

                # two possibilities here:
                # 1) band not in datacube 2) band must be computed via expression
                if band.format is None and band.expression is None:
                    band_available_in_datacube.append(False)
                    bands_unavailable.append(band_name)
                    continue

                if (band.format is None and band.expression is not None) or (
                    band.format is not None and band.expression is None
                ):
                    raise ValueError(
                        f'Properties "format" and "expression" are both required,'
                        f"but only one was given for band with name {band_name}."
                    )

                # if execution gets up to here, it means that bands either available,
                # or may be computed.
                # todo: Check if bands involved in computation are available
                # todo: check if computation is viable

                # if execution of code gets all the way here, this means that the band
                # is unavailable in the datacube, but can computed from other bands
                band_available_in_datacube.append(True)

        if not all(band_available_in_datacube):
            raise LabelDoesNotExist(
                f"The following bands are unavailable in the datacube, but are "
                f"required in the model input: {', '.join(bands_unavailable)}"
            )

    def check_datacube_dimensions(
        self, datacube: xr.DataArray, ignore_batch_dim: bool = False
    ) -> None:
        """
        Check whether the datacube has all dimensions which the model requires.
        :param datacube: The datacube to check
        :param ignore_batch_dim: Ignore a missing "batch" dimension in the datacube
        :raise DimensionMissing: When the stac:mlm item requires an input dimension
        that is not present in the datacube
        :raise DimensionMismatch: When a dimension is smaller in the datacube than
        required by the stac:mlm input shape
        :return:
        """

        self._check_dimensions_present_in_datacube(datacube, ignore_batch_dim)
        self._check_datacube_dimension_size(datacube, ignore_batch_dim)
        self._check_datacube_bands(datacube)

    def get_index_subsets(self, dc: xr.DataArray) -> list[tuple]:
        """
        Get the index per dimension by which the datacube needs to be subset to
        fit the model input
        :param dc: The datacube
        :return: Indexes per dimension, in the order of dim_order
        """
        model_inp_dims = self.model_metadata.input[0].input.dim_order
        model_inp_shape = self.model_metadata.input[0].input.shape
        dim_mapping = self.get_datacube_dimension_mapping(dc)

        # get new dc dim order and shape without "batch" dim
        dc_dims_in_model = [d[0] for d in dim_mapping if d is not None]

        dc_new_input_shape = [
            dim_len
            for dim_name, dim_len in zip(model_inp_dims, model_inp_shape)
            if dim_name != "batch"
        ]

        dc_shape = dc.shape

        dim_ranges = []
        for i in range(len(dc_dims_in_model)):
            step_size = dc_new_input_shape[i]
            n_steps = dc_shape[i] // dc_new_input_shape[i]

            # end at last full step size, remaining values will be cut off
            end = n_steps * step_size
            dim_ranges.append(range(0, end, step_size))
        idx_list = itertools.product(*dim_ranges)
        return idx_list

    def reorder_dc_dims_for_model_input(self, dc: xr.DataArray) -> xr.DataArray:
        """
        Reorders the datacube dimensions according according to model input dims
        :param dc: The datacube
        :return: the reordered datacube
        """
        dim_mapping = self.get_datacube_dimension_mapping(dc)
        dc_dims_in_model = [d[0] for d in dim_mapping if d is not None]
        dc_new_dim_order = [*dc_dims_in_model, ...]
        reordered_dc = dc.transpose(*dc_new_dim_order)
        return reordered_dc

    def reshape_dc_for_input(self, dc: xr.DataArray) -> xr.DataArray:
        """
        Reshapes a datacube into batches to fit the model's input specification.
        Input DC must have only dimensions must be equivalent to what is in the model.
        Dim order of input DC must be the same as in model input.
        :param dc: The datacube to be reshaped
        :return: reshaped DC
        """
        model_inp_dims = self.model_metadata.input[0].input.dim_order
        model_inp_shape = self.model_metadata.input[0].input.shape

        dim_mapping = self.get_datacube_dimension_mapping(dc)

        # get new dc dim order and shape without "batch" dim
        dc_dims_in_model = [d[0] for d in dim_mapping if d is not None]
        dc_new_input_shape = [
            dim_len
            for dim_name, dim_len in zip(model_inp_dims, model_inp_shape)
            if dim_name != "batch"
        ]

        idx_list = self.get_index_subsets(dc)

        # subset dc by indexes to create partial cubes
        part_cubes = []
        for idx in idx_list:
            # dict of idxes by dim by which the DC will be subset
            idxes = {
                dim_name: range(idx[i], idx[i] + dc_new_input_shape[i])
                for i, dim_name in enumerate(dc_dims_in_model)
            }

            dc_part = dc.isel(**idxes)

            # drop DC coordinates (they only cause problems later...
            dc_part = dc_part.drop_vars(
                [
                    dim_name
                    for dim_name in dc_dims_in_model
                    if dim_name in dc_part.coords
                ]
            )

            # add batch dimension
            dc_part = dc_part.expand_dims(
                dim={"batch": 1},
                axis=model_inp_dims.index("batch") if "batch" in model_inp_dims else 0,
            )

            part_cubes.append(dc_part)

        # concat partial cubes by batch dimension
        batched_cube = xr.concat(part_cubes, dim="batch")
        return batched_cube

    def get_batch_size(self) -> int:
        """
        Compute the actual batch size to use when running the model
        :return: batch size
        """
        dim_order = self.model_metadata.input[0].input.dim_order
        shape = self.model_metadata.input[0].input.shape
        batch_size_recommendation = self.model_metadata.batch_size_suggestion
        batch_in_dimensions = "batch" in dim_order

        # todo figure out a good fallback, take RAM, VRAM into consideration
        # todo: dont hard-code fallback value
        fallback_batch_size = 12

        # 1) no batch size anywhere
        # - NO batch size present in in_dims and no recommendation: 1
        if batch_size_recommendation is None and not batch_in_dimensions:
            return 1

        # 2) one batch size available
        # - no batch size present in in_dim, but recommendation: Is that possible???
        if not batch_in_dimensions and batch_size_recommendation is not None:
            return batch_size_recommendation

        # - batch size present in in_dim and not recommendation: size from in_dims
        if batch_in_dimensions and batch_size_recommendation is None:
            batch_size = shape[dim_order.index("batch")]
            if batch_size == -1:
                return fallback_batch_size
            else:
                return batch_size

        # 3) batch size present in in_dim and recommendation:
        if batch_in_dimensions and batch_size_recommendation is not None:
            batch_size = shape[dim_order.index("batch")]
            if batch_size == -1:
                return batch_size_recommendation
            if batch_size == batch_size_recommendation:
                return batch_size_recommendation
            if batch_size != batch_size_recommendation:
                return batch_size

        # this point should never be reached
        raise Exception("Cannot figure out model batch size")

    def get_dims_not_in_model(self, datacube: xr.DataArray) -> list[str]:
        """
        Get datacube dimension names that are not used as model inputs
        :param datacube:
        :return:
        """
        dim_names_in_model = [
            d[0] for d in self.get_datacube_dimension_mapping(datacube) if d is not None
        ]
        dims_not_in_model = [
            str(d) for d in datacube.dims if d not in dim_names_in_model
        ]
        return dims_not_in_model

    def get_input_dims_not_in_output(self, datacube: xr.DataArray) -> list[str]:
        """
        Name of dimensions of input datacube which will not be present in model output
        :param datacube:
        :return:
        """
        in_dims_model = self.model_metadata.input[0].input.dim_order
        in_dims_datacube = self.get_datacube_dimension_mapping(datacube)
        out_dims = self.model_metadata.output[0].result.dim_order

        in_dims_not_in_out_dims = []
        for model_in_dim, dc_in_dim in zip(in_dims_model, in_dims_datacube):
            if model_in_dim not in out_dims:
                in_dims_not_in_out_dims.append(dc_in_dim[0])

        return in_dims_not_in_out_dims

    def get_output_datacube_dimensions(self, in_datacube: xr.DataArray) -> list[str]:
        """
        Order of dimensions of the datacube after prediction
        :param in_datacube: Input datacube before prediction
        :return: List of dimension names
        """
        model_output_dims = self.get_datacube_output_dimension_mapping(in_datacube)
        input_dims_not_in_output = self.get_input_dims_not_in_output(in_datacube)
        dims_not_in_model = self.get_dims_not_in_model(in_datacube)

        # special case "band" dimension
        for band_option in dim_utils.band_dim_options:
            if band_option in input_dims_not_in_output:
                b_idx = input_dims_not_in_output.index(band_option)
                input_dims_not_in_output.pop(b_idx)

        out_datacube_dims = [
            *model_output_dims,
            *input_dims_not_in_output,
            *dims_not_in_model,
        ]

        return out_datacube_dims

    def compare_input_output_dimensions(
        self, in_datacube: xr.DataArray
    ) -> tuple[list[int], list[int]]:
        """
        Compare input and output dimensions and returns indexes of removed and added
        dimensions
        :return: list of indexes of removed dimensions,
        list of indexes of added dimensinos
        """
        in_datacube_dims = [d for d in in_datacube.dims if isinstance(d, str)]
        out_datacube_dims = self.get_output_datacube_dimensions(in_datacube)

        removed_dims, added_dims = dim_utils.compare_input_and_output_datacube_dims(
            in_datacube_dims, out_datacube_dims
        )

        return removed_dims, added_dims

    def get_chunk_output_shape(self, in_datacube: xr.DataArray) -> tuple[int, ...]:
        model_out_dims = self.get_datacube_output_dimension_mapping(in_datacube)
        model_out_shape = self.model_metadata.output[0].result.shape

        input_dims_not_in_output = self.get_input_dims_not_in_output(in_datacube)
        dims_not_in_model = self.get_dims_not_in_model(in_datacube)

        # special case "bands"
        for band_dim in dim_utils.band_dim_options:
            if band_dim in input_dims_not_in_output:
                b_idx = input_dims_not_in_output.index(band_dim)
                input_dims_not_in_output.pop(b_idx)

        # special case "batch"
        batch_dim_len = len(in_datacube.coords["batch"])
        if "batch" in model_out_dims:
            batch_idx = model_out_dims.index("batch")
            model_out_shape[batch_idx] = batch_dim_len
        else:
            model_out_shape.insert(0, batch_dim_len)

        chunk_shape = (
            *model_out_shape,
            *[1 for _ in input_dims_not_in_output],
            *[1 for _ in dims_not_in_model],
        )

        return chunk_shape

    def get_chunk_shape(self, in_datacube: xr.DataArray) -> dict[str, int]:
        """
        Compute the shape of the blocks into which the datacube is chunked before
        applying the ML prediction algorithm
        :return: Diction discribing block shape: key is dimension name, value this
        dimension's block length
        """
        dims_not_in_model = self.get_dims_not_in_model(in_datacube)
        chunk_shape = {}
        for dim_name in in_datacube.dims:
            if dim_name in dims_not_in_model:
                chunk_shape[dim_name] = 1
            else:
                chunk_shape[dim_name] = len(in_datacube.coords[dim_name])
        return chunk_shape

    def feed_datacube_to_model(
        self, datacube: np.ndarray, _, n_batches: int, n_target_dims: int
    ) -> np.ndarray:
        """
        Pass values to the ML model to do prediction.
        :param datacube: the actual values to predict on as numpy.ndarray
        :param _: placeholder for pre-compute hook
        :param n_batches: numbers of sample per batch to use
        :param n_target_dims: Number of dimensions in return array
        :return:
        """
        try:
            # get dimension index of "batch" dimension
            batch_index = self.model_metadata.input[0].input.dim_order.index("batch")
        except ValueError:
            # in case input has no batch dim: use 0
            batch_index = 0

        # at this point we can say the following about the datacube
        # datacube dims: [*dims_in_model (with batch dim), *dims_not_in_model]
        # datacube shape:
        #   - length of dimensions in model are as they need to be to satisfy model
        #   - length of dimensions not in model are 1

        num_input_dims = len(self.model_metadata.input[0].input.shape)
        num_datacube_dims = len(datacube.shape)
        axes_to_squeeze = tuple(range(num_input_dims, num_datacube_dims))
        datacube = datacube.squeeze(
            axis=axes_to_squeeze
        )  # todo: only squeeze not-in-model dims!!!

        b_len = datacube.shape[batch_index]

        returned_dcs = []
        for b_idx in range(0, b_len, n_batches + 1):
            s_dc = datacube[b_idx : b_idx + n_batches + 1]

            # make prediction in framework-specific derived classes
            model_out = self.execute_model(s_dc)

            returned_dcs.append(model_out)

        batch_stack = np.vstack(returned_dcs)
        return_array = np.expand_dims(
            batch_stack, tuple(range(len(batch_stack.shape), n_target_dims))
        )
        return return_array

    def resolve_batch(
        self,
        dc_batched: xr.DataArray,
        batch_indices: tuple[tuple[int, ...], ...],
        input_dc_dim_mapping: list[None | tuple[str, int]],
        output_dc_dim_mapping: list[str],
        dims_not_in_model: list[str],
        input_dc_coords: xarray.core.coordinates.DataArrayCoordinates,
    ) -> xr.DataArray:
        """
        Resolves the datacube batches that come out of the ML model back to a spatio-
        temporal datacube
        :param dc_batched: the batched datacube
        :param batch_indices: The indices used to to create the batches
        :param input_dc_dim_mapping:
        :param output_dc_dim_mapping:
        :dims_not_in_model:
        :param input_dc_coords: Input datacube coordinates
        :return: the un-batched datacube
        """
        # assert DC has a "batch" dimension
        if "batch" not in dc_batched.dims:
            raise Exception("Datacube does not have a batch dimension")

        # assert each batch dimension has appropriate coordinate indices
        if len(dc_batched.coords["batch"]) != len(batch_indices):
            raise Exception(
                f"Different number of batches in datacube that given in batch_indices:\n"
                f"{len(dc_batched.coords['batch'])=}, {len(batch_indices)=}"
            )

        # set name to None to ensure that combine_by_coords will return a DataArray
        dc_batched.name = None

        # get names of datacube input dimensions
        dc_input_dims = [n[0] if n is not None else None for n in input_dc_dim_mapping]
        # filter out "batch" dimension (is None in mapping)
        dc_input_dims_without_batch = [n for n in dc_input_dims if n is not None]

        model_input_shape = self.model_metadata.input[0].input.shape
        model_input_shape_without_batch = [
            d_shape
            for d_name, d_shape in zip(input_dc_dim_mapping, model_input_shape)
            if d_name is not None
        ]

        model_output_dims = output_dc_dim_mapping
        model_output_shape = self.model_metadata.output[0].result.shape

        # re-attach coordinates for dims not used in the model
        for d in dims_not_in_model:
            coords = input_dc_coords[d]
            dc_batched.coords[d] = coords

        reshaped_slices = []

        # resolve "batch" dimension and reassemble datacube
        for batch_idx, batch_coord_idxes in zip(
            dc_batched.coords["batch"], batch_indices
        ):
            # slice datacube by batch index
            dc_slice = dc_batched.sel(batch=batch_idx)

            for inp_dim_name, inp_dim_len, inp_idx in zip(
                dc_input_dims_without_batch,
                model_input_shape_without_batch,
                batch_coord_idxes,
            ):
                self._resolve_dimension(
                    dc_slice,
                    inp_dim_name,
                    model_output_dims,
                    model_output_shape,
                    input_dc_coords,
                    inp_idx,
                    inp_dim_len,
                )

            reshaped_slices.append(dc_slice)

        # re-combine the previously batched datacube parts
        combined_slices = xr.combine_by_coords(reshaped_slices, data_vars="all")

        return combined_slices

    def _resolve_dimension(
        self,
        dc_slice: xr.DataArray,
        inp_dim_name: str,
        model_output_dims: list[str],
        model_output_shape: list[int],
        input_dc_coords: xarray.core.coordinates.DataArrayCoordinates,
        inp_idx: int,
        inp_dim_len: int,
    ) -> None:
        # case: inp_dim_name not in output cube
        if inp_dim_name not in model_output_dims:
            self._resolve_dimension_not_in_output(
                dc_slice, inp_dim_name, input_dc_coords, inp_idx, inp_dim_len
            )

        # inp_dim_name in output cube
        else:
            self._resolve_dimension_in_output(
                dc_slice,
                model_output_shape,
                model_output_dims,
                inp_dim_name,
                input_dc_coords,
                inp_dim_len,
                inp_idx,
            )

    def _resolve_dimension_in_output(
        self,
        dc_slice: xr.DataArray,
        model_output_shape: list[int],
        model_output_dims: list[str],
        inp_dim_name: str,
        input_dc_coords: xarray.core.coordinates.DataArrayCoordinates,
        inp_dim_len: int,
        inp_idx: int,
    ) -> None:
        # get new dim length
        new_dim_len = model_output_shape[model_output_dims.index(inp_dim_name)]

        # get input dc coords for dim
        coords_for_dim = input_dc_coords[inp_dim_name].data

        if inp_dim_name not in input_dc_coords:
            # special case: DC does not have coords for a dimension
            new_coords = np.array(range(new_dim_len))

        elif inp_dim_len == new_dim_len:
            # special case: length is the same in input and output
            # -> simply assign the same coords
            new_coords = coords_for_dim[inp_idx : inp_idx + inp_dim_len]

        elif inp_dim_len == 1:
            # todo
            pass

        # elif new_dim_len == 1:
        #     # todo: is this even necessary? or is it cought below
        #     pass

        elif np.issubdtype(coords_for_dim.dtype, np.number):
            # for numeric coords, space them evenly between min and max
            coord_width = coords_for_dim[1] - coords_for_dim[0]
            image_start = coords_for_dim[inp_idx] - coord_width / 2
            image_end = coords_for_dim[inp_idx + inp_dim_len - 1] + coord_width / 2
            new_coord_width = (image_end - image_start) / new_dim_len
            coord_start = image_start + new_coord_width / 2
            coord_end = image_end - new_coord_width / 2
            new_coords = np.linspace(coord_start, coord_end, new_dim_len)

        elif np.issubdtype(coords_for_dim.dtype, np.datetime64):
            # for datetime coords, space them evenly between start and end
            # This solution is not ideal as time coords are usually not
            # spaced evenly in input DC
            coords_start = coords_for_dim[inp_idx].astype("datetime64[ns]").astype(int)
            try:
                coord_end = (
                    coords_for_dim[inp_idx + inp_dim_len]
                    .astype("datetime64[ns]")
                    .astype(int)
                )
            except IndexError:
                mean_diff = np.mean(coords_for_dim[1:] - coords_for_dim[:-1])
                end_date = coords_for_dim[-1] + mean_diff
                coord_end = end_date.astype("datetime64[ns]").astype(int)

            new_coords = np.linspace(
                coords_start,
                coord_end,
                new_dim_len,
                endpoint=False,
                dtype=int,
            ).astype("datetime64[ns]")

        else:
            # all other cases, e.g. str: join input coords,append a number
            # ex: B1, B2 -> B1.B2-1, B1.B2-2, B1.B2-3
            old_coords = coords_for_dim[inp_idx : inp_idx + inp_dim_len]
            new_coords = np.char.add(
                ".".join(old_coords) + "-",
                np.array(range(new_dim_len)).astype(str),
            )

        dc_slice.coords[inp_dim_name] = new_coords

    def _resolve_dimension_not_in_output(
        self,
        dc_slice: xr.DataArray,
        inp_dim_name: str,
        input_dc_coords: xarray.core.coordinates.DataArrayCoordinates,
        inp_idx: int,
        inp_dim_len: int,
    ):
        # special case: ignore band dimension
        if (
            isinstance(inp_dim_name, str)
            and inp_dim_name.lower() in dim_utils.band_dim_options
        ):
            return

        if inp_dim_name.lower() in dim_utils.spatial_dim_options and np.issubdtype(
            input_dc_coords[inp_dim_name].dtype, np.number
        ):
            # handle spatial coordinates
            coord_start = input_dc_coords[inp_dim_name][inp_idx].data
            coord_end = input_dc_coords[inp_dim_name][inp_idx + inp_dim_len - 1].data
            coord = (coord_start + coord_end) / 2
        else:
            # get coord in appropriate dim (inp_dim_name) at index (inp_idx)
            coord = input_dc_coords[inp_dim_name][inp_idx].data
        dc_slice.coords[inp_dim_name] = [coord]

    def run_model(self, datacube: xr.DataArray) -> xr.DataArray:
        # first check if all dims required by model are in data cube
        self.check_datacube_dimensions(datacube, ignore_batch_dim=True)

        if self._model_object is None:
            self.create_object()

        input_dim_mapping = self.get_datacube_dimension_mapping(datacube)

        pre_datacube = self.preprocess_datacube(datacube)

        # dimensions: [*dims-in-model (without "batch"), *dims-not-in-model]
        reordered_dc = self.reorder_dc_dims_for_model_input(pre_datacube)

        # dimensions: ["batch", *dims-in-model, *dims-not-in-model]
        # with "batch" being wherever it needs to be to satisfy model input
        input_dc = self.reshape_dc_for_input(reordered_dc)

        # batch size to be used during inference
        n_batches = self.get_batch_size()

        # get dimension indices of each batch: tuple[tuple[int, ], ...]
        batch_indices = tuple(self.get_index_subsets(reordered_dc))

        dims_not_in_model = self.get_dims_not_in_model(datacube)
        chunk_shape = self.get_chunk_shape(input_dc)
        new_chunked = input_dc.chunk(chunk_shape)

        output_dim_mapping = self.get_datacube_output_dimension_mapping(input_dc)

        dims_removed, dims_added = self.compare_input_output_dimensions(input_dc)
        chunk_out_shape = self.get_chunk_output_shape(input_dc)

        out_dtype = self.model_metadata.output[0].result.data_type
        try:
            out_dtype_np = np.dtype(out_dtype)
        except TypeError:
            raise Exception(f"Output data type {out_dtype} not supported")

        # ############################################################################
        #  From now on, we operate on the dask array only, not on xarray DataArray!  #
        #  Makes things easier. We will go back to xarray later.                     #
        ##############################################################################

        data: da.Array = new_chunked.data

        # add pre-computehook to dask graph
        init_model = dask.delayed(self.pre_map_block_compute_hook)()

        # Map the function to predict to each datacube block (chunk)
        block_mapped = data.map_blocks(
            self.feed_datacube_to_model,
            init_model,
            dtype=out_dtype_np,
            drop_axis=dims_removed,
            new_axis=dims_added,
            chunks=chunk_out_shape,
            # kwargs passed to callback function
            n_batches=n_batches,  # number of samples to use per batch
            n_target_dims=len(chunk_out_shape),
        )

        # add post-compute hook
        uninit_model = dask.delayed(self.post_map_block_compute_hook)(block_mapped)

        # make dask array from post-compute hook
        model_out = da.from_delayed(
            uninit_model, shape=block_mapped.shape, dtype=block_mapped.dtype
        )

        ##################################
        #  Now back to xarray DataArray  #
        ##################################

        model_out_datacube = xr.DataArray(
            model_out, dims=self.get_output_datacube_dimensions(input_dc)
        )

        # resolve "batch" dimension in datacube
        resolved_datacube = self.resolve_batch(
            model_out_datacube,
            batch_indices,
            input_dim_mapping,
            output_dim_mapping,
            dims_not_in_model,
            pre_datacube.coords,
        )

        post_cube = self.postprocess_datacube(datacube, resolved_datacube)
        return post_cube

    def pre_map_block_compute_hook(self):
        self.init_model_for_prediction()

    def post_map_block_compute_hook(self, result):
        self.uninit_model_after_prediction()
        return result

    def reorder_out_dc_dims(
        self, in_cube: xr.DataArray, out_cube: xr.DataArray
    ) -> xr.DataArray:
        spatial_dims = dim_utils.get_spatial_dim_names(in_cube)

        old_non_sptial_dims = [
            d for d in in_cube.dims if d in out_cube.dims and d not in spatial_dims
        ]

        new_dims = [d for d in out_cube.dims if d not in in_cube.dims]

        old_spatial_dims = [
            d for d in in_cube.dims if d in out_cube.dims and d in spatial_dims
        ]

        dim_order = [*old_non_sptial_dims, *new_dims, *old_spatial_dims]

        reorederd_cube = out_cube.transpose(*dim_order)
        return reorederd_cube

    def select_bands(self, datacube: xr.DataArray) -> xr.DataArray:
        model_inp_bands = self.model_metadata.input[0].bands
        if not model_inp_bands:
            return datacube

        band_dim_name = dim_utils.get_band_dim_name(datacube)
        band_coords = datacube.coords[band_dim_name].values.tolist()

        model_band_names = []
        for b in model_inp_bands:
            if isinstance(b, str):
                model_band_names.append(b)
            else:
                model_band_names.append(b.name)

        bands_to_select = dim_utils.get_dc_band_names(band_coords, model_band_names)
        return datacube.sel(**{band_dim_name: bands_to_select})

    def scale_values(self, datacube: xr.DataArray) -> xr.DataArray:
        scaling = self.model_metadata.input[0].value_scaling

        if scaling is None:
            return datacube

        band_dim_name = dim_utils.get_band_dim_name(datacube)

        if len(scaling) == 1:
            # scale all bands the same
            scale_obj = scaling[0]
            scaling_utils.scale_datacube(datacube, scale_obj)
            return datacube

        # if code execution reaches this point, each band is scaled individually

        # assert number of scaling items equals number of bands
        if len(scaling) != len(datacube.coords[band_dim_name]):
            raise ValueError(
                f"Number of ValueScaling entries does not match number of bands in "
                f"Data Cube. Number of entries: {len(scaling)}; "
                f"Number of bands: {len(datacube.coords[band_dim_name])}"
            )

        scaled_bands = []
        for band_name, scale_obj in zip(datacube.coords[band_dim_name], scaling):
            scaled_band = scaling_utils.scale_datacube(
                datacube.sel(**{band_dim_name: band_name}), scale_obj
            )
            scaled_bands.append(scaled_band)

        return xr.concat(scaled_bands, dim=band_dim_name)

    def preprocess_datacube_expression(self, input_obj) -> xr.DataArray:
        pre_proc_expression = self.model_metadata.input[0].pre_processing_function
        if pre_proc_expression is None:
            return input_obj

        try:
            pre_processing_result = proc_expression_utils.run_process_expression(
                input_obj, pre_proc_expression
            )
        except ExpressionEvaluationException as e:
            raise Exception(
                f"Error applying pre-processing function to datacube: {str(e)}"
            )
        return pre_processing_result

    def postprocess_datacube_expression(self, output_obj):
        post_proc_expression = self.model_metadata.output[0].post_processing_function
        if post_proc_expression is None:
            return output_obj

        try:
            post_processed_output = proc_expression_utils.run_process_expression(
                output_obj, post_proc_expression
            )
        except ExpressionEvaluationException as e:
            raise Exception(f"Error applying post-processing function: {str(e)}")
        return post_processed_output

    def preprocess_datacube(self, datacube: xr.DataArray) -> xr.DataArray:
        # processing expression formats
        # gdal-calc, openeo, rio-calc, python, docker, uri

        # todo: datacube compute new bands?
        subset_datacube = self.select_bands(datacube)

        scaled_dc = self.scale_values(subset_datacube)
        preproc_dc_casted = scaled_dc.astype(
            self.model_metadata.input[0].input.data_type
        )
        # todo: datacube padding?
        return preproc_dc_casted

    def postprocess_datacube(
        self, in_datacube: xr.DataArray, resolved_datacube: xr.DataArray
    ) -> xr.DataArray:
        post_cube_reorderd = self.reorder_out_dc_dims(in_datacube, resolved_datacube)

        try:
            epsg = epsg_utils.get_epsg_from_datacube(in_datacube)
            post_cube_reorderd.rio.write_crs(epsg, inplace=True)
        except ReferenceSystemNotFound:
            logger.warning(
                "Could not detect the CRS datacube which is used in prediction, "
                "therefore no reference system is assigned to the prediction output"
            )

        return post_cube_reorderd

    def create_object(self):
        if self._model_object is not None:
            # model object has already been created
            return

        model_filepath = self._get_model()
        self.create_model_object(model_filepath)

    @abstractmethod
    def create_model_object(self, filepath: str):
        """
        Create a model object (self._model_object) in the respective framework.
        :param filepath: Path to the model object
        :return: None
        """
        pass

    @abstractmethod
    def init_model_for_prediction(self):
        """
        Initiate the model object, e.g. move it to the cuda device. This function will
        be executed before prediction from datacube values.
        :return: None
        """
        pass

    @abstractmethod
    def uninit_model_after_prediction(self):
        """
        Uninitialize the model after model prediction, e.g. take it off the cuda device.
        This function will be executed after all prediction on the datacube are
        completed
        :return: None
        """
        pass

    @abstractmethod
    def execute_model(self, batch: np.ndarray) -> xr.DataArray:
        """
        Make a prediction with the model object
        :param batch: The object which will be passed to the model to make a
        prediction on. Must have been transformed to a shape to be accepted by the model
        :return: None
        """
        pass
