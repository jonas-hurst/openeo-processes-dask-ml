import os.path
import itertools

import pystac
from pystac.extensions.mlm import MLMExtension
from abc import ABC, abstractmethod

import xarray as xr

from openeo_processes_dask_ml.process_implementations.constants import MODEL_CACHE_DIR

from openeo_processes_dask_ml.process_implementations.utils import (
    model_cache_utils, download_utils, scaling_utils, proc_expression_utils, dim_utils
)

from openeo_processes_dask.process_implementations.exceptions import (
    DimensionMissing, DimensionMismatch
)
from openeo_processes_dask_ml.process_implementations.exceptions import (
    LabelDoesNotExist, ExpressionEvaluationException
)


class MLModel(ABC):
    stac_item: pystac.Item

    def __init__(self, stac_item: pystac.Item, model_asset_name: str = None):
        self.stac_item = stac_item
        self._model_asset_name = model_asset_name
        self._model_object = None

    @property
    def model_metadata(self) -> MLMExtension:
        # todo: account for if metadata is stored with the asset
        return MLMExtension.ext(self.stac_item)

    def _get_model_asset(self, asset_name: str = None) -> pystac.Asset:
        """
        Determine which asset holds the model (has mlm:model in roles)
        Determines whcih asset to use if multiple assets with role mlm:models are found
        :param asset_name:
        :return:
        """
        assets = self.stac_item.assets
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
                    f"Provided STAC Item does not have an asset named {asset_name} which "
                    f"also lists mlm:model as its asset role"
                )

        # case 3: asset name is not given and there is only one mlm:model asset
        if len(model_assets) == 1:
            return next(iter(model_assets.values()))

        # case 4: multiple mlm:model exist but asset_name is not specified
        raise Exception(
            "Multiple assets with role=mlm:model are found in the provided STAC-Item. "
            "Please sepcify which one to use."
        )

    def _get_model(self, asset_name=None) -> str:
        model_asset = self._get_model_asset(asset_name)
        url = model_asset.href

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

    def get_datacube_dimension_mapping(self, datacube: xr.DataArray) -> list[None|tuple[str,int]]:
        """
        Maps the model input dimension names to datacube dimension names, as dimension
        names can sometimes differ, e.g. t -> time
        :param datacube: The datacube to map the dimeions agains
        :return: Tuple with dc-equivalent model input dimension names and their index
        """
        model_dims = self.model_metadata.input[0].input.dim_order
        dc_dims = datacube.dims

        def get_dc_dim_name(model_dim_name: str) -> str|None:
            if model_dim_name in dc_dims:
                return model_dim_name

            t_dim_names = ["time", "times", "t", "date", "dates", "DATE"]
            if model_dim_name in t_dim_names:
                return next((t_dim for t_dim in t_dim_names if t_dim in dc_dims), None)

            b_dim_names = ["band", "bands", "b", "channel", "channels"]
            if model_dim_name in b_dim_names:
                return next((b_dim for b_dim in b_dim_names if b_dim in dc_dims), None)

            x_dim_names = ["x", "lon", "lng", "longitude"]
            if model_dim_name in x_dim_names:
                return next((x_dim for x_dim in x_dim_names if x_dim in dc_dims), None)

            y_dim_names = ["y", "lat", "latitude"]
            if model_dim_name in y_dim_names:
                return next((y_dim for y_dim in y_dim_names if y_dim in dc_dims), None)

            batch_dim_names = ["batch", "batches"]
            if model_dim_name in batch_dim_names:
                return next((batch_dim for batch_dim in batch_dim_names if batch_dim in dc_dims), None)

            return None

        dim_mapping = []
        for m_dim_name in model_dims:

            dc_dim_name = get_dc_dim_name(m_dim_name)
            if dc_dim_name is None:
                dim_mapping.append(None)
            else:
                dim_mapping.append((dc_dim_name, dc_dims.index(dc_dim_name)))

        return dim_mapping

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
            dc_shape[dim_mapping[i][1]] for i, inp_dim in enumerate(input_dims)
            if inp_dim != "batch" or (inp_dim == "batch" and not ignore_batch_dim)
        ]

        # ignore "batch" dimension for now, we will take care of that later
        input_shape_reorder = [
            d for i, d in zip(input_dims, input_shape)
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

        # bands prorety not utilized, list is empty
        if not input_bands:
            return

        # possibilities how the "bands" dimension could be called
        band_dim_name = dim_utils.get_band_dim_name(datacube)

        dc_bands = datacube.coords[band_dim_name]
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

                if (
                        (band.format is None and band.expression is not None) or
                        (band.format is not None and band.expression is None)
                ):
                    raise ValueError(
                        f"Properties \"format\" and \"expression\" are both required,"
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
                dim_name: range(idx[i], idx[i]+dc_new_input_shape[i])
                for i, dim_name in enumerate(dc_dims_in_model)
            }

            dc_part = dc.isel(**idxes)

            # drop DC coordinates (they only cause problems later...
            dc_part = dc_part.drop_vars(
                [dim_name for dim_name in dc_dims_in_model if dim_name in dc_part.coords]
            )

            # add batch dimension
            dc_part = dc_part.expand_dims(
                dim={"batch": 1},
                axis=model_inp_dims.index("batch") if "batch" in model_inp_dims else 0)

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

    def feed_datacube_to_model(
            self, datacube: xr.DataArray, n_batches: int
    ) -> xr.DataArray:
        b_len = len(datacube.coords["batch"])

        returned_dcs = []
        for b_idx in range(0, b_len, n_batches):
            batch_subsets = range(
                b_idx,
                # account for "end" of DC where there are fewer batches left
                b_idx + n_batches if b_idx + n_batches < b_len else b_len
            )

            s_dc = datacube.isel(batch=batch_subsets)
            model_out = self.execute_model(s_dc)
            returned_dcs.append(model_out)
        return xr.concat(returned_dcs, dim="batch")

    def get_datacube_subset_indices(
            self, datacube: xr.DataArray
    ) -> list[dict]:
        # get datacube dimensions which are not in the model
        dim_names_in_model = [
            d[0] for d in self.get_datacube_dimension_mapping(datacube) if d is not None
        ]
        dims_not_in_model = [d for d in datacube.dims if d not in dim_names_in_model]

        # if a "batch" dimension is not in the model, we will take care of that later
        if "batch" in dims_not_in_model:
            dims_not_in_model.remove("batch")

        # create subsets of cubes:
        # iterate over each dimension that is not used for model input

        coords = [datacube.coords[d].values.tolist() for d in dims_not_in_model]
        idx_sets = itertools.product(*coords)

        # todo: handle cases where all dims are model inputs (= no subcube_idx_sets)

        subcube_idx_sets = []
        for idx_set in idx_sets:
            subset = {
                dim_name: idx for idx, dim_name in zip(idx_set, dims_not_in_model)
            }
            subcube_idx_sets.append(subset)
            # subcube = datacube.sel(**subset)

        return subcube_idx_sets

    def run_model(self, datacube: xr.DataArray) -> xr.DataArray:
        # first check if all dims required by model are in data cube
        self.check_datacube_dimensions(datacube, ignore_batch_dim=True)

        if self._model_object is None:
            self.create_object()

        pre_datacube = self.preprocess_datacube(datacube)

        # todo: datacube rechunk?
        reordered_dc = self.reorder_dc_dims_for_model_input(pre_datacube)
        input_dc = self.reshape_dc_for_input(reordered_dc)
        input_dc = input_dc.compute()

        # get list of datacube subset coordinates
        # these are dimensions with coordinates that are not used in model input
        subcube_idx_sets = self.get_datacube_subset_indices(input_dc)

        n_batches = self.get_batch_size()
        inferred_batches = []

        # iterate over coordinates of unused dimensions
        # perform inference for each individually
        # todo: handle cases where all dims are model inputs (= no subcube_idx_sets)
        for subcube_idx_set in subcube_idx_sets:
            # slice datacube by unused dimension coordinates
            subcube = input_dc.sel(**subcube_idx_set)

            # run inference on datacube subsets
            model_out = self.feed_datacube_to_model(subcube, n_batches)
            inferred_batches.append(model_out)

        post_cube = xr.concat(inferred_batches, dim="batch")
        # todo reassemble data cube from batches
        # -resolve batches per subcube
        # -resolve subcubes

        # could or could not have x/y lan/lon dimensions
        # attach missing dimensions
        # attach appropriate coordinates to dimensions
        return post_cube

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

    def preprocess_datacube_expression(self, datacube: xr.DataArray):
        # todo: is xarray datacube really input/output?
        pre_proc_expression = self.model_metadata.input[0].pre_processing_function
        if pre_proc_expression is None:
            return datacube

        try:
            proc_expression_utils.run_process_expression(datacube, pre_proc_expression)
        except ExpressionEvaluationException as e:
            raise Exception(
                f"Error applying pre-processing function to datacube: {str(e)}"
            )
        # todo: is this return correct???
        return proc_expression_utils

    def postprocess_datacube_expression(self, output_obj):
        post_proc_expression = self.model_metadata.output[0].post_processing_function
        if post_proc_expression is None:
            return output_obj

        try:
            post_processed_output = proc_expression_utils.run_process_expression(
                output_obj, post_proc_expression
            )
        except ExpressionEvaluationException as e:
            raise Exception(
                f"Error applying post-processing function: {str(e)}"
            )
        return post_processed_output

    def preprocess_datacube(self, datacube: xr.DataArray) -> xr.DataArray:

        # processing expression formats
        # gdal-calc, openeo, rio-calc, python, docker, uri

        # todo: datacube compute new bands?
        subset_datacube = self.select_bands(datacube)

        scaled_dc = self.scale_values(subset_datacube)
        preproc_dc = self.preprocess_datacube_expression(scaled_dc)
        preproc_dc_casted = preproc_dc.astype(
            self.model_metadata.input[0].input.data_type
        )
        # todo: datacube padding?
        return preproc_dc_casted

    def postprocess_datacube(self, result_cube) -> xr.DataArray:
        # todo: output gemäß mlm-spec post-processing transformieren
        # todo: output zu neuem datacube zusammenführen
        pass

    def create_object(self):
        if self._model_object is not None:
            # model object has already been created
            return

        model_filepath = self._get_model(self._model_asset_name)
        self.create_model_object(model_filepath)

    @abstractmethod
    def create_model_object(self, filepath: str):
        pass

    @abstractmethod
    def execute_model(self, batch: xr.DataArray) -> xr.DataArray:
        pass
