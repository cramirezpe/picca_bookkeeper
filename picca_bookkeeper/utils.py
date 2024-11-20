"""Utils."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import fitsio
import healpy as hp
import numpy as np

from picca_bookkeeper.constants import forest_regions

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Tuple

    from picca_bookkeeper.hints import wave_grid, wave_grid_int, wave_grid_rf


def find_bins(original_array: wave_grid, grid_array: wave_grid) -> wave_grid_int:
    idx = np.searchsorted(grid_array, original_array)
    np.clip(idx, 0, len(grid_array) - 1, out=idx)

    prev_index_closer = (grid_array[idx - 1] - original_array) ** 2 <= (
        grid_array[idx] - original_array
    ) ** 2
    return idx - prev_index_closer


def find_qso_pixel(los_id: int, catalog: str | Path) -> Tuple[int, str]:
    """Find healpix pixel where given quasar is located

    Arguments
    ---------
    los_id: Line of sight id of quasar.
    catalog: Path to quasar catalog.
    """
    with fitsio.FITS(catalog) as hdul:
        indx = np.where(hdul["ZCATALOG"]["TARGETID"][:] == los_id)[0][0]

        ra = hdul["ZCATALOG"]["TARGET_RA"][indx]
        dec = hdul["ZCATALOG"]["TARGET_DEC"][indx]

        return (
            hp.pixelfunc.ang2pix(64, ra, dec, lonlat=True, nest=True),
            hdul["ZCATALOG"]["SURVEY"][indx],
        )


def compute_zeff(
    export_files: List[Path],
    rmins: List[float] | float = 0,
    rmaxs: List[float] | float = 1000,
) -> float:
    """Compute zeff from a set of export files"""
    if not isinstance(rmins, list) or (isinstance(rmins, list) and len(rmins) == 1):
        rmins = [rmins for file in export_files]  # type: ignore

    if not isinstance(rmaxs, list) or (isinstance(rmaxs, list) and len(rmaxs) == 1):
        rmaxs = [rmaxs for file in export_files]  # type: ignore

    zeff_list = []
    weights = []
    for export_file, rmin, rmax in zip(export_files, rmins, rmaxs):
        with fitsio.FITS(export_file) as hdul:
            r_arr = np.sqrt(hdul[1].read()["RP"] ** 2 + hdul[1].read()["RT"] ** 2)
            cells = (r_arr > rmin) * (r_arr < rmax)

            inverse_variance = 1 / np.diag(hdul[1].read()["CO"])
            zeff = np.average(
                hdul[1].read()["Z"][cells], weights=inverse_variance[cells]
            )
            weight = np.sum(inverse_variance[cells])

        zeff_list.append(zeff)
        weights.append(weight)

    return np.average(zeff_list, weights=weights)


def get_spectra_from_los_id(
    los_id: int,
    catalog: str | Path,
    input_healpix: Optional[str | Path] = None,
    lambda_grid: wave_grid = np.arange(3600, 9824, 0.8),
) -> Tuple[wave_grid, wave_grid, wave_grid]:
    """Get quasar spectra given los_id

    Arguments
    ---------
    los_id: Line of sight id of quasar.
    catalog: Path to quasar catalog.
    lambda_grid: Grid to show spectra for (np.array).
    """
    healpix, survey = find_qso_pixel(los_id, catalog)
    healpix_str = str(healpix)

    if input_healpix is None:
        input_healpix = Path("/global/cfs/cdirs/desi/science/lya/fugu_healpix/healpix")
    else:
        input_healpix = Path(input_healpix)

    coadd_file = (
        input_healpix
        / survey
        / "dark"
        / healpix_str[:-2]
        / healpix_str
        / f"coadd-{survey}-dark-{healpix_str}.fits"
    )

    assert coadd_file.is_file()

    with fitsio.FITS(coadd_file) as hdul:
        assert los_id in hdul["FIBERMAP"]["TARGETID"]

        idx = np.where(hdul["FIBERMAP"]["TARGETID"][:] == los_id)[0][0]

        flux_data = dict(
            B_WAVE=hdul["B_WAVELENGTH"].read(),
            R_WAVE=hdul["R_WAVELENGTH"].read(),
            Z_WAVE=hdul["Z_WAVELENGTH"].read(),
            B_FLUX=hdul["B_FLUX"].read()[idx],
            R_FLUX=hdul["R_FLUX"].read()[idx],
            Z_FLUX=hdul["Z_FLUX"].read()[idx],
            B_MASK=hdul["B_MASK"].read()[idx],
            R_MASK=hdul["R_MASK"].read()[idx],
            Z_MASK=hdul["Z_MASK"].read()[idx],
            B_IVAR=hdul["B_IVAR"].read()[idx] * (hdul["B_MASK"].read()[idx] == 0),
            R_IVAR=hdul["R_IVAR"].read()[idx] * (hdul["R_MASK"].read()[idx] == 0),
            Z_IVAR=hdul["Z_IVAR"].read()[idx] * (hdul["Z_MASK"].read()[idx] == 0),
        )

    return coadd_flux_data(flux_data, lambda_grid)


def coadd_flux_data(
    flux_data: Dict, lambda_grid: wave_grid
) -> Tuple[wave_grid, wave_grid, wave_grid]:
    """Coadd flux data for different arms

    Arguments
    ---------
    flux_data: Dictionary containing flux data in keys 'B', 'R', 'Z'.
    lambda_grid: Grid to show spectra for (np.array).
    """
    flux = np.zeros_like(lambda_grid)
    ivar = np.zeros_like(lambda_grid)

    for color in "B", "R", "Z":
        bins = find_bins(flux_data[f"{color}_WAVE"], lambda_grid)

        flux += np.bincount(
            bins,
            weights=flux_data[f"{color}_IVAR"] * flux_data[f"{color}_FLUX"],
            minlength=flux.size,
        )
        ivar += np.bincount(
            bins, weights=flux_data[f"{color}_IVAR"], minlength=ivar.size
        )

    w = ivar > 0
    flux[w] /= ivar[w]

    return lambda_grid, flux, ivar


def Ulambda(lambda_rest: wave_grid, region: str = "lya") -> wave_grid_rf:
    """Helper function to buid quasar continuum from aq bq parameters"""
    lambda_min = forest_regions[region]["lambda-rest-min"]
    lambda_max = forest_regions[region]["lambda-rest-max"]

    return (np.log10(lambda_rest) - np.log10(lambda_min)) / (
        np.log10(lambda_max) - np.log10(lambda_min)
    )


def compute_cont(
    los_id: int, attrs_file: str | Path, region: str = "lya"
) -> Tuple[wave_grid_rf, wave_grid_rf]:
    """Compute quasar continuum using attributes file (will read aq, bq and mean cont)

    Arguments:
    ----------
    los_id: Line of sight id of quasar.
    attrs_file: Path to delta attributes file.
    region: Region where to compute the continuum.
    """
    with fitsio.FITS(attrs_file) as hdul:
        lambda_rest = 10 ** hdul["CONT"]["loglam_rest"][:]
        mean_cont = hdul["CONT"]["mean_cont"][:]

        idx = np.where(hdul["FIT_METADATA"]["LOS_ID"][:] == los_id)[0][0]
        aq = hdul["FIT_METADATA"]["ZERO_POINT"][idx]
        bq = hdul["FIT_METADATA"]["SLOPE"][idx]

        cont = mean_cont * (aq + bq * (Ulambda(lambda_rest, region=region)))

    return lambda_rest, cont
