import copy
import math
import pickle
import random
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import xarray as xr
from dask import array as da
from dask import dataframe as ddf
from dask import delayed
from pystac.extensions.classification import Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from openeo_processes_dask_ml.model_execution import run_sklearn_model
from openeo_processes_dask_ml.process_implementations.constants import MODEL_CACHE_DIR

from .data_model import MLModel


class SkLearnModel(MLModel):
    def make_predictions(
        self,
        model_filepath: str,
        files: Iterable[Path],
        tmp_dir_output: Path,
        preproc_expression,
        postproc_expression,
    ):
        run_sklearn_model.predict(
            model_filepath,
            tmp_dir_output,
            files,
            preproc_expression,
            postproc_expression,
        )

    def get_run_command(self, tmp_dir_input, tmp_dir_output) -> list[str]:
        run_command = [
            sys.executable,
            run_sklearn_model.__file__,
            self._model_filepath,
            tmp_dir_input,
            tmp_dir_output,
        ]
        return run_command

    def predict_func(self, arr: np.ndarray, model_path: str, input_dims: list[str]):
        with open(model_path, "rb") as file:
            model = pickle.load(file)

        n_dims_arr = len(arr.shape)

        # arr shape: (..., bands)
        orig_shape = arr.shape[: n_dims_arr - len(input_dims)]
        feature_shape = arr.shape[n_dims_arr - len(input_dims) :]
        n_features = math.prod(feature_shape)

        arr2d = arr.reshape(-1, n_features)

        pred = model.predict(arr2d)

        return pred.reshape(orig_shape)

    def run_model(self, datacube: xr.DataArray) -> xr.DataArray:
        # !!!!
        # At the moment only works for models that take in a single dim (e.g. bands)
        # if len(self.input.input.dim_order) > 1:
        #     raise NotImplementedError(
        #         "this model is not supported as it takes more than one dim as input"
        #     )

        if len(self.output.result.dim_order) > 1:
            raise NotImplementedError(
                "this model is not supported as it outputs more than 1 dimension"
            )

        if self.output.result.shape[0] != 1:
            raise NotImplementedError(
                f"this model is not supported a its output length is "
                f"{self.output.result.shape[0]} "
                f" but only length 1 is supported."
            )

        # first check if all dims required by model are in data cube
        self.check_datacube_dimensions(datacube, ignore_batch_dim=True)

        if self._model_filepath is None:
            self._model_filepath = self._get_model()

        # resolve delayed model_filepath: Turn delayed string into 0-d dask array
        model_path_da = da.from_delayed(
            self._model_filepath, shape=(), dtype=object
        ).persist()

        # using .persist() is not that elegant, as it starts computation already on a worker
        # This is (so far) the only solution that I found to make the object resolve only once
        # before passing it to apply_ufunc below, and not once for each chunk.
        # we obviously only want to fit the model once, then re-use it for each chunk.

        input_dim_mapping = self.get_datacube_dimension_mapping(datacube)
        input_dims = [d[0] for d in input_dim_mapping]

        pre_datacube = self.preprocess_datacube(datacube)

        # dimensions: [*dims-in-model, *dims-not-in-model]
        # if input_dim_mapping[0][1] != 0:
        #     pre_datacube = self.reorder_dc_dims_for_model_input(pre_datacube)

        pre_datacube = pre_datacube.chunk({d: -1 for d in input_dims})

        result = xr.apply_ufunc(
            self.predict_func,
            pre_datacube,
            model_path_da,
            input_core_dims=[input_dims, []],
            output_core_dims=[[]],
            kwargs={"input_dims": input_dims},
            # vectorize=False,
            dask="parallelized",
            output_dtypes=[self.output.result.data_type],
        )

        result = result.expand_dims(self.output.result.dim_order[0])

        return result


class RfClassModel(SkLearnModel):
    @staticmethod
    def init_model(
        max_features: int | str | float | None,
        n_trees: int,
        model_id: str,
        seed: int = None,
    ) -> str:
        # todo: input dimensions: bands, time-bands, embedding

        r = RandomForestClassifier(
            n_trees, max_features=max_features, random_state=seed
        )

        # save model to disk
        modelpath = MODEL_CACHE_DIR + "/" + model_id + ".pkl"
        with open(modelpath, "wb") as file:
            pickle.dump(r, file)

        return modelpath

    @delayed
    def fit(self, training_set_df: ddf.DataFrame, pred_col_names: list[str]) -> str:
        random.seed(self.seed)
        np.random.seed(self.seed)

        model_path = self._model_filepath

        with open(model_path, "rb") as file:
            model: RandomForestClassifier = pickle.load(file)

        out_col_name = self.output.result.dim_order[0]

        X = training_set_df[pred_col_names]
        y = training_set_df[out_col_name]

        encoder = LabelEncoder()

        # +1 because we want labels to start at 1 instead of 0
        # for compatability with R backend
        y_enc = encoder.fit_transform(y) + 1

        X_train, X_val, y_train, y_val = train_test_split(
            X, y_enc, test_size=0.15, random_state=self.seed
        )

        # Here we finally fit the model!!!
        model.fit(X_train.values, y_train)

        # todo add this to the resulting (new) ml-object
        # classes = [
        #     Classification.create(i + 1, name=name)
        #     for i, name in enumerate(encoder.classes_)
        # ]

        with open(model_path, "wb") as file:
            pickle.dump(model, file)

        y_pred = model.predict(X_val)
        report = classification_report(
            y_val,
            y_pred,
            labels=range(
                1, len(encoder.classes_) + 1
            ),  # start range at 1 and end at 1 more due to encodeed labels starting at 1
            target_names=encoder.classes_,
        )
        print("Classification Report: \n")
        print(report)
        print()
        print(f"Overall Accuracy: {accuracy_score(y_val, y_pred)}")
        print()
        print(f"Cohens Kappa: {cohen_kappa_score(y_val, y_pred)}")

        return model_path

    def fit_model(self, training_set: xr.DataArray) -> Self:
        training_set = training_set.chunk(-1).persist()
        out_dims = self.output.result.dim_order
        if len(out_dims) > 1:
            raise ValueError("Only one output dimension is allowed in RF classifier")

        dim_mapping = self.get_datacube_dimension_mapping(training_set)
        dims = [d[0] for d in dim_mapping]

        stacked = training_set.stack(feature=dims)

        # I pitty anyone who at one point will have to debug this line...
        new_cols = [
            "_".join(str(v) for v in vals)
            for vals in zip(*[stacked.coords[d].values for d in dims])
        ]
        stacked = stacked.drop_vars(["feature", *dims]).assign_coords(feature=new_cols)

        training_set_ds = stacked.to_dataset(dim="feature")
        training_set_df = training_set_ds.to_dask_dataframe().reset_index(drop=True)

        fitted_model_path = self.fit(training_set_df, new_cols)

        rf_model_copy = copy.deepcopy(self)
        rf_model_copy._model_filepath = fitted_model_path

        return rf_model_copy
