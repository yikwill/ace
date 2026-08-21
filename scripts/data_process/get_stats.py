# The dependencies of this script are installed in the "fv3net" conda environment
# which can be installed using fv3net's Makefile. See
# https://github.com/ai2cm/fv3net/blob/8ed295cf0b8ca49e24ae5d6dd00f57e8b30169ac/Makefile#L310

import dataclasses
import logging
import os
import shutil
import tempfile
import time
from glob import glob
from typing import Literal, Optional

import click
import dacite
import fsspec
import numpy as np
import xarray as xr
import yaml
from xarray.coding.times import CFDatetimeCoder

# these are auxiliary variables that exist in dataset for convenience, e.g. to do
# masking or to more easily compute vertical integrals. But they are not inputs
# or outputs to the ML model, so we don't need normalization constants for them.
DROP_VARIABLES = (
    [
        "land_sea_mask",
        "pressure_thickness_of_atmospheric_layer_0",
        "pressure_thickness_of_atmospheric_layer_1",
        "pressure_thickness_of_atmospheric_layer_2",
        "pressure_thickness_of_atmospheric_layer_3",
        "pressure_thickness_of_atmospheric_layer_4",
        "pressure_thickness_of_atmospheric_layer_5",
        "pressure_thickness_of_atmospheric_layer_6",
        "pressure_thickness_of_atmospheric_layer_7",
        "mask_HI",
        "mask_sea_ice_volume",
        "mask_sea_ice_fraction",
        "mask_ocean_sea_ice_fraction",
    ]
    + [f"ak_{i}" for i in range(9)]
    + [f"bk_{i}" for i in range(9)]
    + [f"idepth_{i}" for i in range(19)]
    + [f"mask_{i}" for i in range(19)]
)

DIMS = {
    "FV3GFS": ["time", "grid_xt", "grid_yt"],
    "E3SMV2": ["time", "lat", "lon"],
    "E3SMV3": ["time", "lat", "lon"],
    "ERA5": ["time", "latitude", "longitude"],
    "CM4": ["time", "lat", "lon"],
}

ClimateDataType = Literal["FV3GFS", "E3SMV2", "E3SMV3", "ERA5", "CM4"]


def add_history_attrs(ds, input_source, start_date, end_date, n_samples):
    ds.attrs["history"] = (
        "Created by full-model/scripts/data_process/get_stats.py. INPUT:"
        f" {input_source}, START_DATE: {start_date}, END_DATE: {end_date}."
    )
    ds.attrs["input_samples"] = n_samples


def copy(source: str, destination: str):
    """Copy between any two 'filesystems'. Do not use for large files.

    Args:
        source: Path to source file/object.
        destination: Path to destination.
    """
    with fsspec.open(source) as f_source:
        with fsspec.open(destination, "wb") as f_destination:
            shutil.copyfileobj(f_source, f_destination)


@dataclasses.dataclass
class StatsConfig:
    output_directory: str
    data_type: ClimateDataType
    exclude_runs: list[str] = dataclasses.field(default_factory=list)
    start_date: str | None = None
    end_date: str | None = None
    beaker_dataset: str | None = None
    data_path: str | None = None
    file_pattern: str | None = None


@dataclasses.dataclass
class TimeCoarsenConfig:
    """
    Configuration for time coarsening of a dataset.

    Attributes:
        data_output_directory: Directory to save the coarsened datasets as zarr stores.
        stats_output_directory: Directory to save the stats of the coarsened datasets.
    """

    data_output_directory: str
    stats_output_directory: str


@dataclasses.dataclass
class Config:
    runs: dict[str, str]
    stats: StatsConfig
    data_output_directory: str | None = None
    time_coarsen: TimeCoarsenConfig | None = None


def _glob_paths(data_path: str, file_pattern: str) -> list[str]:
    return sorted(glob(os.path.join(data_path, file_pattern)))


def _filter_empty_netcdf_paths(paths: list[str]) -> list[str]:
    kept = []
    for path in paths:
        with xr.open_dataset(path, decode_times=False, decode_timedelta=False) as ds:
            if ds.sizes.get("time", 0) > 0:
                kept.append(path)
            else:
                logging.info(f"Skipping NetCDF file with no timesteps: {path}")
    return kept


def _select_numeric_data_vars(ds: xr.Dataset) -> xr.Dataset:
    numeric_vars = [
        name for name in ds.data_vars if np.issubdtype(ds[name].dtype, np.number)
    ]
    dropped = set(ds.data_vars) - set(numeric_vars)
    if dropped:
        logging.info(f"Dropping non-numeric variables: {sorted(dropped)}")
    return ds[numeric_vars]


def _input_source(config: StatsConfig, input_zarr: str | None) -> str:
    if config.data_path is not None:
        return os.path.join(config.data_path, config.file_pattern or "")
    if input_zarr is None:
        raise ValueError("input_zarr is required when stats.data_path is not set.")
    return input_zarr


def _open_stats_input(
    config: StatsConfig,
    input_zarr: str | None,
    dask,
) -> xr.Dataset:
    if config.data_path is not None:
        if config.file_pattern is None:
            raise ValueError(
                "stats.file_pattern is required when stats.data_path is set."
            )
        paths = _glob_paths(config.data_path, config.file_pattern)
        paths = _filter_empty_netcdf_paths(paths)
        if not paths:
            raise ValueError(
                f"No files found matching {config.data_path}/{config.file_pattern}."
            )
        logging.info(
            f"Opening {len(paths)} NetCDF files from "
            f"{config.data_path}/{config.file_pattern}"
        )
        open_kwargs = {
            "decode_times": CFDatetimeCoder(use_cftime=True),
            "decode_timedelta": False,
            "combine": "by_coords",
            "compat": "override",
            "data_vars": "minimal",
            "coords": "minimal",
        }
        if dask is not None:
            open_kwargs["chunks"] = {"time": "auto"}
        open_start = time.time()
        ds = xr.open_mfdataset(paths, **open_kwargs)
        logging.info(
            f"Opened {len(paths)} NetCDF files in {time.time() - open_start:0.2f}"
            f" seconds."
        )
        return ds

    if input_zarr is None:
        raise ValueError("input_zarr is required when stats.data_path is not set.")
    logging.info(f"Reading data from {input_zarr}")
    # Open data with roughly 128 MiB chunks via dask's automatic chunking. This
    # is useful when opening sharded zarr stores with an inner chunk size of 1,
    # which is otherwise inefficient for the type of computation done here.
    if dask is not None:
        with dask.config.set({"array.chunk-size": "128MiB"}):
            return xr.open_zarr(input_zarr, chunks={"time": "auto"})
    return xr.open_zarr(input_zarr)


def _out_dir_exists(out_dir: str) -> bool:
    """Check if the stats output directory already has results."""
    if out_dir.startswith("gs:"):
        fs = fsspec.filesystem("gs")
        return fs.exists(out_dir + "/centering.nc")
    else:
        return os.path.exists(os.path.join(out_dir, "centering.nc"))


def get_stats(
    config: StatsConfig,
    out_dir: str,
    debug: bool,
    input_zarr: str | None = None,
):
    if not debug and _out_dir_exists(out_dir):
        logging.info(f"Stats already exist at {out_dir}. Skipping.")
        return

    # Import dask-related things here to enable testing in environments without dask.
    try:
        import dask
        import distributed

        client = distributed.Client(n_workers=16)
    except ImportError as e:
        # warn and continue
        logging.warning(f"Could not import dask ({e}), chunking is disabled.")
        client = None
        dask = None

    initial_time = time.time()
    input_source = _input_source(config, input_zarr)

    xr.set_options(keep_attrs=True, display_max_rows=100)
    ds = _open_stats_input(config, input_zarr, dask)

    ds = ds.drop_vars(DROP_VARIABLES, errors="ignore")
    ds = _select_numeric_data_vars(ds)
    ds = ds.sel(time=slice(config.start_date, config.end_date))

    dims = DIMS[config.data_type]

    # Explicitly compute the statistics here, since xarray does not support
    # writing netCDFs with the scipy engine with the distributed scheduler.
    # There is no harm to computing here versus later, since the end result is
    # not something memory intensive.
    centering = ds.mean(dim=dims).compute()
    logging.info("Computed centering")
    scaling_full_field = ds.std(dim=dims).compute()
    logging.info("Computed scaling_full_field")
    scaling_residual = ds.diff("time").std(dim=dims).compute()
    logging.info("Computed scaling_residual")
    time_means = ds.mean(dim="time").compute()
    logging.info("Computed time_means")

    for dataset in [
        centering,
        scaling_full_field,
        scaling_residual,
        time_means,
    ]:
        n_samples = len(ds.time)
        add_history_attrs(
            dataset,
            input_source,
            config.start_date,
            config.end_date,
            n_samples,
        )

    if debug:
        normed_data = (ds - centering) / scaling_full_field
        logging.info(f"Average of normed data: {normed_data.mean(dim=dims).compute()}")
        logging.info(
            f"Standard deviation of normed data: {normed_data.std(dim=dims).compute()}"
        )
        try:
            all_var_stddev = (
                normed_data.to_array().std(dim=["variable"] + dims).compute()
            )
            logging.info(
                f"Standard deviation computed over all variables: "
                f"{all_var_stddev.values}"
            )
        except ValueError as e:
            logging.info(
                f"Skipping all-variable stddev check ({e}). "
                f"Dataset has {len(normed_data.data_vars)} variables."
            )
    else:
        if out_dir.startswith("gs:"):
            temp_dir = tempfile.TemporaryDirectory()
            local_dir = temp_dir.name
            remote_dir: Optional[str] = out_dir
        else:
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            local_dir = out_dir
            remote_dir = None

        centering.to_netcdf(os.path.join(local_dir, "centering.nc"))
        if remote_dir is not None:
            copy(
                os.path.join(local_dir, "centering.nc"),
                remote_dir + "/centering.nc",
            )
        scaling_full_field.to_netcdf(os.path.join(local_dir, "scaling-full-field.nc"))
        if remote_dir is not None:
            copy(
                os.path.join(local_dir, "scaling-full-field.nc"),
                remote_dir + "/scaling-full-field.nc",
            )
        scaling_residual.to_netcdf(os.path.join(local_dir, "scaling-residual.nc"))
        if remote_dir is not None:
            copy(
                os.path.join(local_dir, "scaling-residual.nc"),
                remote_dir + "/scaling-residual.nc",
            )
        time_means.to_netcdf(os.path.join(local_dir, "time-mean.nc"))
        if remote_dir is not None:
            copy(
                os.path.join(local_dir, "time-mean.nc"),
                remote_dir + "/time-mean.nc",
            )

    total_time = time.time() - initial_time
    logging.info(f"Total time for computing stats: {total_time:0.2f} seconds.")

    if client is not None:
        client.close()
    client = None


@click.command()
@click.argument("config_yaml", type=str)
@click.argument("run", type=int)
@click.option(
    "--debug",
    is_flag=True,
    help="If set, print some statistics instead of writing normalization coefficients.",
)
def main(config_yaml: str, run: int, debug: bool):
    """
    Compute statistics for the data processing pipeline.

    Arguments:
    config_yaml -- Path to the configuration file for the data processing pipeline.
    run -- Run index for the data processing pipeline.
    """

    logging.basicConfig(level=logging.INFO)

    with open(config_yaml, "r") as f:
        config_data = yaml.load(f, Loader=yaml.CLoader)
    config = dacite.from_dict(data_class=Config, data=config_data)
    run_name = list(config.runs.keys())[run]
    if run_name in config.stats.exclude_runs:
        logging.info(f"Skipping run {run_name}")
        return
    out_dir = config.stats.output_directory + "/" + run_name
    if config.stats.data_path is not None:
        get_stats(
            config=config.stats,
            out_dir=out_dir,
            debug=debug,
        )
    else:
        if config.data_output_directory is None:
            raise ValueError(
                "data_output_directory is required when stats.data_path is not set."
            )
        if config.data_output_directory.endswith("/"):
            config.data_output_directory = config.data_output_directory[:-1]
        input_zarr = config.data_output_directory + "/" + run_name + ".zarr"
        get_stats(
            config=config.stats,
            input_zarr=input_zarr,
            out_dir=out_dir,
            debug=debug,
        )
    if config.time_coarsen is not None:
        time_coarsened_zarr = (
            config.time_coarsen.data_output_directory + "/" + run_name + ".zarr"
        )
        time_coarsened_out_dir = (
            config.time_coarsen.stats_output_directory + "/" + run_name
        )
        get_stats(
            config=config.stats,
            input_zarr=time_coarsened_zarr,
            out_dir=time_coarsened_out_dir,
            debug=debug,
        )


if __name__ == "__main__":
    main()
