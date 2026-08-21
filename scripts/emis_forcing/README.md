# E3SM aerosol emission / external-forcing datasets for ACE

Build ACE-compatible 6-hourly zarr stores of per-species **surface emissions** and
**column-integrated elevated (external) forcings** for use when `aerindexall` and
`colccn.3` are prognostic.

## Dataset map (CFS)

Root: `/global/cfs/projectdirs/e3sm/yikwill/datasets/`

| Role | Paths |
|------|-------|
| Native 6h simulation | `e3sm-aerosol-{PI,PD}-1945-1980.zarr` |
| Aerosol clim → 6h (`aerindexall`, `colccn.3` only; no `ccn.3bl`) | `e3sm-aerosol-{PI,PD}-aerosol-clim-6hourly-1945-1980.zarr` |
| Emis clim → 6h | `e3sm-aerosol-{PI,PD}-emis-forcing-6hourly-1945-1980.zarr` |

Monthly NetCDF source dirs (including any `ccn.3bl` fields) are left untouched.

## Stats trees

Built by `build_aerosol_stats_trees.py` (PRECT ×1000). Each of `PI/`, `PD/`,
`combined/` has `centering.nc`, `scaling-full-field.nc`, `scaling-residual.nc`,
`time-mean.nc`.

| Tree | Atmos | `aerindexall` / `colccn.3` | `emis_*` |
|------|-------|----------------------------|----------|
| `e3sm-aerosol-stats-aerosol-clim-forcing/` | sim zarr | aerosol clim zarr | — |
| `e3sm-aerosol-stats-aerosol-prognostic/` | sim zarr | sim zarr | emis zarr |

```bash
# On an interactive node (prefer tmux + salloc):
python build_aerosol_stats_trees.py --workers 48 --force
python verify_e3sm_aerosol_train_assets.py
```

## Emis zarr build

| Era | Path |
|-----|------|
| PI | `/global/cfs/projectdirs/e3sm/yikwill/datasets/e3sm-aerosol-PI-emis-forcing-6hourly-1945-1980.zarr` |
| PD | `/global/cfs/projectdirs/e3sm/yikwill/datasets/e3sm-aerosol-PD-emis-forcing-6hourly-1945-1980.zarr` |

Layout: `(time, lat, lon)` with `time=52560` (1945–1980, 6-hourly, noleap),
`lat=180`, `lon=360` matching the existing PI/PD state zarr Gaussian grid.

### Source data

Species → file map follows [`.misc/emission_files.md`](../../../.misc/emission_files.md)
under `/global/cfs/projectdirs/e3sm/inputdata/atm/cam/chem/trop_mozart_aero/emis/`.

| Era | MAM4 / SOAG | Chem gases / DMS / NO2 aircraft |
|-----|-------------|----------------------------------|
| **PI** | Year **1850** from `*_1850-2014_*.nc`, repeated as seasonal clim | `*2010-as-1850*` 12-month clim (fluxes = 2010) |
| **PD** | `*_2010_clim_*.nc` | `*_2010_clim_*` |

`ISOP_VBS` is skipped (same file as `ISOP`).

### Processing steps

1. **Sum sectors** within each file.
2. **Column-integrate** elevated / aircraft fields (`molecules/cm3/s` → `molecules/cm2/s`).
3. **Regrid** to the ACE Gaussian 180×360 grid with **xesmf conservative**.
4. **Interpolate** monthly clim → 6-hourly via periodic cubic spline, tiled over 1945–1980.
5. Write float32 zarr v3 (time chunk 124).

Variable names: `emis_<species>_sfc` and/or `emis_<species>_elev`.

```bash
conda activate fme
python build_emis_forcing_zarr.py --era both \
  --emis-root /global/cfs/projectdirs/e3sm/inputdata/atm/cam/chem/trop_mozart_aero/emis \
  --out-dir /global/cfs/projectdirs/e3sm/yikwill/datasets
```

Repair a known corrupt PI chunk (if needed):

```bash
python repair_pi_emis_c10h16_chunk.py
```
