# The dependencies of this script are installed in the "fv3net" conda environment
# which can be installed using fv3net's Makefile. See
# https://github.com/ai2cm/fv3net/blob/8ed295cf0b8ca49e24ae5d6dd00f57e8b30169ac/Makefile#L310

import dataclasses
import logging
import os
import shutil
import tempfile
import time
from typing import Literal, Optional

import click
import dacite
import fsspec
import numpy as np
import xarray as xr
import yaml

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
    "ERA5": ["time", "latitude", "longitude"],
    "CM4": ["time", "lat", "lon"],
}

ClimateDataType = Literal["FV3GFS", "E3SMV2", "ERA5", "CM4"]


def add_history_attrs(ds, input_zarr, start_date, end_date, n_samples):
    ds.attrs["history"] = (
        "Created by full-model/scripts/data_process/get_stats.py. INPUT_ZARR:"
        f" {input_zarr}, START_DATE: {start_date}, END_DATE: {end_date}."
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
    max_time_samples: int | None = None
    include_variables: list[str] | None = None
    # Process this many timesteps at a time to bound memory (no dask required).
    time_chunk_size: int | None = None


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
    data_output_directory: str
    stats: StatsConfig
    time_coarsen: TimeCoarsenConfig | None = None


def _out_dir_exists(out_dir: str) -> bool:
    """Check if the stats output directory already has results."""
    if out_dir.startswith("gs:"):
        fs = fsspec.filesystem("gs")
        return fs.exists(out_dir + "/centering.nc")
    else:
        return os.path.exists(os.path.join(out_dir, "centering.nc"))


def _compute_stats_time_chunked(
    ds: xr.Dataset,
    dims: list[str],
    chunk_size: int,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
    """Compute stats by loading ``chunk_size`` timesteps at a time."""
    n_time = len(ds.time)
    grid_count = int(np.prod([ds.sizes[d] for d in dims if d != "time"]))
    total_count = 0
    total_sum: xr.Dataset | None = None
    total_sumsq: xr.Dataset | None = None
    time_sum: xr.Dataset | None = None
    diff_sum: xr.Dataset | None = None
    diff_sumsq: xr.Dataset | None = None
    diff_count = 0
    prev_timestep: xr.Dataset | None = None

    for start in range(0, n_time, chunk_size):
        end = min(start + chunk_size, n_time)
        chunk = ds.isel(time=slice(start, end)).load()
        chunk_count = (end - start) * grid_count
        chunk_sum = chunk.sum(dim=dims)
        chunk_sumsq = (chunk**2).sum(dim=dims)
        total_sum = chunk_sum if total_sum is None else total_sum + chunk_sum
        total_sumsq = chunk_sumsq if total_sumsq is None else total_sumsq + chunk_sumsq
        total_count += chunk_count

        chunk_time_sum = chunk.sum(dim="time")
        time_sum = chunk_time_sum if time_sum is None else time_sum + chunk_time_sum

        if prev_timestep is not None:
            boundary_diff = chunk.isel(time=[0]) - prev_timestep
            inner_diff = chunk.diff("time")
            diffs = xr.concat([boundary_diff, inner_diff], dim="time")
        else:
            diffs = chunk.diff("time")
        prev_timestep = chunk.isel(time=[-1])

        diff_chunk_count = len(diffs.time) * grid_count
        diff_chunk_sum = diffs.sum(dim=dims)
        diff_chunk_sumsq = (diffs**2).sum(dim=dims)
        diff_sum = diff_chunk_sum if diff_sum is None else diff_sum + diff_chunk_sum
        diff_sumsq = (
            diff_chunk_sumsq if diff_sumsq is None else diff_sumsq + diff_chunk_sumsq
        )
        diff_count += diff_chunk_count
        logging.info(
            f"Processed timesteps {start}:{end} of {n_time} "
            f"({100.0 * end / n_time:.1f}%)"
        )

    assert total_sum is not None and total_sumsq is not None
    assert time_sum is not None
    assert diff_sum is not None and diff_sumsq is not None

    centering = total_sum / total_count
    variance = total_sumsq / total_count - centering**2
    scaling_full_field = np.sqrt(variance.clip(min=0.0))
    diff_variance = diff_sumsq / diff_count - (diff_sum / diff_count) ** 2
    scaling_residual = np.sqrt(diff_variance.clip(min=0.0))
    time_means = time_sum / n_time
    return centering, scaling_full_field, scaling_residual, time_means


def get_stats(
    config: StatsConfig,
    input_zarr: str,
    out_dir: str,
    debug: bool,
    force: bool = False,
):
    if not debug and not force and _out_dir_exists(out_dir):
        logging.info(f"Stats already exist at {out_dir}. Skipping.")
        return

    # Import dask-related things here to enable testing in environments without dask.
    client = None
    dask_module = None
    use_time_chunks = config.time_chunk_size is not None
    if not use_time_chunks:
        try:
            import dask as _dask
            import distributed

            dask_module = _dask
            n_workers = int(os.environ.get("DASK_NUM_WORKERS", "16"))
            client = distributed.Client(n_workers=n_workers)
            logging.info(f"Started dask client with {n_workers} workers")
        except ImportError as e:
            logging.warning(f"Could not import dask ({e}), chunking is disabled.")

    initial_time = time.time()

    xr.set_options(keep_attrs=True, display_max_rows=100)
    logging.info(f"Reading data from {input_zarr}")

    # Open data with roughly 128 MiB chunks via dask's automatic chunking. This
    # is useful when opening sharded zarr stores with an inner chunk size of 1,
    # which is otherwise inefficient for the type of computation done here.
    if dask_module is not None:
        with dask_module.config.set({"array.chunk-size": "128MiB"}):
            ds = xr.open_zarr(input_zarr, chunks={"time": "auto"})
    else:
        ds = xr.open_zarr(input_zarr)

    ds = ds.drop_vars(DROP_VARIABLES, errors="ignore")
    ds = ds.sel(time=slice(config.start_date, config.end_date))

    if config.max_time_samples is not None and len(ds.time) > config.max_time_samples:
        idx = np.random.default_rng(0).choice(
            len(ds.time), config.max_time_samples, replace=False
        )
        ds = ds.isel(time=sorted(idx))
        logging.info(
            f"Subsampled to {config.max_time_samples} timesteps for stats computation"
        )

    if config.include_variables is not None:
        missing = set(config.include_variables) - set(ds.data_vars)
        if missing:
            raise ValueError(f"include_variables not in dataset: {sorted(missing)}")
        ds = ds[config.include_variables]
        logging.info(f"Restricted stats to {len(config.include_variables)} variables")

    dims = DIMS[config.data_type]

    if use_time_chunks:
        assert config.time_chunk_size is not None
        logging.info(
            f"Computing stats in time chunks of {config.time_chunk_size} timesteps"
        )
        centering, scaling_full_field, scaling_residual, time_means = (
            _compute_stats_time_chunked(ds, dims, config.time_chunk_size)
        )
        logging.info(
            "Computed centering, scaling_full_field, scaling_residual, time_means"
        )
    else:
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
            input_zarr,
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
        all_var_stddev = normed_data.to_array().std(dim=["variable"] + dims)
        logging.info(
            f"Standard deviation computed over all variables: {all_var_stddev.values}"
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
@click.option(
    "--force",
    is_flag=True,
    help="Recompute stats even if output files already exist.",
)
def main(config_yaml: str, run: int, debug: bool, force: bool):
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
    if config.data_output_directory.endswith("/"):
        config.data_output_directory = config.data_output_directory[:-1]
    input_zarr = config.data_output_directory + "/" + run_name + ".zarr"
    out_dir = config.stats.output_directory + "/" + run_name
    get_stats(
        config=config.stats,
        input_zarr=input_zarr,
        out_dir=out_dir,
        debug=debug,
        force=force,
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
            force=force,
        )


if __name__ == "__main__":
    main()
