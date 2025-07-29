"""
utils.py
--------

Utility functions for the picca_bookkeeper package.

This module provides supporting functions for quasar and Lyman-alpha forest
analysis, including array bin finding, Healpix pixel identification, zeff
computation, spectra extraction, flux coaddition, and quasar continuum construction.

Utilities facilitate reading and processing FITS files, handling wavelength
grids, and support continuum modeling using survey metadata.

Key functionality:
------------------
    - find_bins: Map data arrays to wavelength grids using nearest bin search.
    - find_qso_pixel: Locate the Healpix pixel and survey for a given quasar
      by los_id.
    - compute_zeff: Calculate the effective redshift (zeff) from survey export
      files.
    - get_spectra_from_los_id: Extract spectra for a given quasar, including
      flux and inverse variance.
    - coadd_flux_data: Combine flux data from multiple spectral arms into a
      single grid.
    - Ulambda: Normalize rest-frame wavelengths for continuum construction.
    - compute_cont: Build the quasar continuum using fitted parameters from
      attributes files.
"""
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
    """
    Find the nearest bin indices for values in `original_array` within `grid_array`.

    Arguments
    ----------
    original_array : wave_grid
        Array of values to map onto the grid.
    grid_array : wave_grid
        The reference grid to map to.

    Returns
    -------
    wave_grid_int
        Array of indices in `grid_array` corresponding to nearest neighbors
        of `original_array`.
    """
    idx = np.searchsorted(grid_array, original_array)
    np.clip(idx, 0, len(grid_array) - 1, out=idx)

    prev_index_closer = (grid_array[idx - 1] - original_array) ** 2 <= (
        grid_array[idx] - original_array
    ) ** 2
    return idx - prev_index_closer


def find_qso_pixel(los_id: int, catalog: str | Path) -> Tuple[int, str]:
    """
    Find the HEALPix pixel and survey for a given quasar.

    Arguments
    ----------
    los_id : int
        Line-of-sight ID of the quasar.
    catalog : str or Path
        Path to the quasar catalog FITS file.

    Returns
    -------
    Tuple[int, str]
        Tuple containing the HEALPix pixel index (nested, Nside=64) and
        survey name.
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
    """
    Compute the effective redshift (zeff) from a list of export files.

    Arguments
    ----------
    export_files : List[Path]
        Paths to the FITS export files.
    rmins : List[float] or float, optional
        Minimum R values for filtering (can be a scalar or list), by default 0.
    rmaxs : List[float] or float, optional
        Maximum R values for filtering (can be a scalar or list), by default 1000.

    Returns
    -------
    float
        Weighted average effective redshift.
    """
    if not isinstance(rmins, list) or (isinstance(rmins, list) and len(rmins) == 1):
        rmins = [rmins for file in export_files]  # type: ignore

    if not isinstance(rmaxs, list) or (isinstance(rmaxs, list) and len(rmaxs) == 1):
        rmaxs = [rmaxs for file in export_files]  # type: ignore

    zeff_list = []
    weights = []
    for export_file, rmin, rmax in zip(export_files, rmins, rmaxs):
        with fitsio.FITS(export_file) as hdul:
            r_arr = np.sqrt(hdul[1].read()["RP"] ** 2 +
                            hdul[1].read()["RT"] ** 2)
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
    """
    Retrieve flux, wavelength, and inverse variance arrays for a given quasar.

    Arguments
    ----------
    los_id : int
        Line-of-sight ID of the quasar.
    catalog : str or Path
        Path to the quasar catalog FITS file.
    input_healpix : str or Path, optional
        Path to the root healpix directory, by default uses DESI standard location.
    lambda_grid : wave_grid, optional, np.array
        Wavelength grid for interpolated output,
        by default np.arange(3600, 9824, 0.8)

    Returns
    -------
    Tuple[wave_grid, wave_grid, wave_grid]
        Tuple of (lambda_grid, flux, inverse variance)
    """
    healpix, survey = find_qso_pixel(los_id, catalog)
    healpix_str = str(healpix)

    if input_healpix is None:
        input_healpix = Path(
            "/global/cfs/cdirs/desi/science/lya/fugu_healpix/healpix")
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
            B_IVAR=hdul["B_IVAR"].read()[idx] *
            (hdul["B_MASK"].read()[idx] == 0),
            R_IVAR=hdul["R_IVAR"].read()[idx] *
            (hdul["R_MASK"].read()[idx] == 0),
            Z_IVAR=hdul["Z_IVAR"].read()[idx] *
            (hdul["Z_MASK"].read()[idx] == 0),
        )

    return coadd_flux_data(flux_data, lambda_grid)


def coadd_flux_data(
    flux_data: Dict, lambda_grid: wave_grid
) -> Tuple[wave_grid, wave_grid, wave_grid]:
    """
    Combine flux data from B, R, and Z arms into a coadded spectrum on a common grid.

    Arguments
    ----------
    flux_data : dict
        Dictionary containing flux, inverse variance, and wavelength arrays for
        each arm (keys 'B', 'R', 'Z').
    lambda_grid : wave_grid, np.array
        Wavelength grid to coadd onto.

    Returns
    -------
    Tuple[wave_grid, wave_grid, wave_grid]
        Tuple of (lambda_grid, coadded_flux, coadded_ivar)
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
    """
    Normalize rest-frame wavelengths for continuum modeling.

    Helper function to buid quasar continuum from aq bq parameters.

    Arguments
    ----------
    lambda_rest : wave_grid, np.array
        Rest-frame wavelength array.
    region : str, optional
        Spectral region to use for normalization, by default "lya".

    Returns
    -------
    wave_grid_rf
        Normalized log-scale rest-frame wavelength array.
    """

    lambda_min = forest_regions[region]["lambda-rest-min"]
    lambda_max = forest_regions[region]["lambda-rest-max"]

    return (np.log10(lambda_rest) - np.log10(lambda_min)) / (
        np.log10(lambda_max) - np.log10(lambda_min)
    )


def compute_cont(
    los_id: int, attrs_file: str | Path, region: str = "lya"
) -> Tuple[wave_grid_rf, wave_grid_rf]:
    """
    Construct quasar continuum from metadata attributes file
    (will read aq, bq and mean cont).

    Arguments
    ----------
    los_id : int
        Line-of-sight ID of the quasar.
    attrs_file : str or Path
        Path to the delta attributes FITS file containing fitted Arguments.
    region : str, optional
        Region to compute continuum over (e.g., "lya"), by default "lya".

    Returns
    -------
    Tuple[wave_grid_rf, wave_grid_rf]
        Tuple of (rest-frame wavelength, continuum model)
    """
    with fitsio.FITS(attrs_file) as hdul:
        lambda_rest = 10 ** hdul["CONT"]["loglam_rest"][:]
        mean_cont = hdul["CONT"]["mean_cont"][:]

        idx = np.where(hdul["FIT_METADATA"]["LOS_ID"][:] == los_id)[0][0]
        aq = hdul["FIT_METADATA"]["ZERO_POINT"][idx]
        bq = hdul["FIT_METADATA"]["SLOPE"][idx]

        cont = mean_cont * (aq + bq * (Ulambda(lambda_rest, region=region)))

    return lambda_rest, cont
