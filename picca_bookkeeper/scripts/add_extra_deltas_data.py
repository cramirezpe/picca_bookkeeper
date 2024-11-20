""" Read all masks applied during a picca run, for studying its effects"""

from __future__ import annotations

import argparse
import itertools
import logging
import multiprocessing
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import fitsio
import numpy as np
from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.masks.bal_mask import BalMask
from picca.delta_extraction.masks.dla_mask import DlaMask, dla_profile
from picca.delta_extraction.survey import Survey

from picca_bookkeeper.utils import find_bins

if TYPE_CHECKING:
    from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ReadExtraDeltasData:
    """Class to read extra data from picca deltas."""

    def __init__(
        self,
        picca_config: Path | str,
    ):
        """Initialize class instance

        Args:
            picca_config: Picca configuration file (output is better since it has all the info.)
        """
        self.picca_config = picca_config

        self.survey = Survey()
        self.survey.load_config(str(self.picca_config))

    def _get_survey_mask(self, classinfo: DlaMask | BalMask) -> DlaMask | BalMask:
        for mask in self.survey.masks:
            if isinstance(mask, classinfo):
                return mask
        else:
            raise ValueError(
                "Could not find compatible mask in survey masks", self.survey.masks
            )

    def read_data(self) -> None:
        """Read data from DESI using picca survey object."""
        self.survey.read_corrections()
        self.survey.read_masks()
        self.survey.read_data()
        self.survey.apply_corrections()

    def read_DLA_mask(self, num_processors: int) -> None:
        """Read DLA mask using picca code."""
        logger.info("Reading DLA masks.")
        mask = self._get_survey_mask(DlaMask)

        context = multiprocessing.get_context("fork")
        chunksize = int(len(self.survey.data.forests) / num_processors / 3)
        chunksize = max(1, chunksize)

        with context.Pool(processes=num_processors) as pool:
            self.survey.data.forests = pool.starmap(
                read_DLA_mask_single,
                zip(
                    self.survey.data.forests,
                    itertools.repeat(mask),
                ),
                chunksize=chunksize,
            )

    def read_BAL_mask(self, num_processors: int) -> None:
        """Read BAL mask using picca code."""
        logger.info("Reading BAL masks.")
        mask = self._get_survey_mask(BalMask)

        context = multiprocessing.get_context("fork")
        chunksize = int(len(self.survey.data.forests) / num_processors / 3)
        chunksize = max(1, chunksize)

        # for forest in self.survey.data.forests:
        #     read_BAL_mask_single(forest, mask)

        with context.Pool(processes=num_processors) as pool:
            self.survey.data.forests = pool.starmap(
                read_BAL_mask_single,
                zip(
                    self.survey.data.forests,
                    itertools.repeat(mask),
                ),
                chunksize=chunksize,
            )

    def save_data_mask(
        self,
        out_dir: Path | str,
        sorted_ids: Dict,
        num_processors: Optional[int] = None,
    ) -> None:
        logger.info("Converting statistics into rest frame.")
        Path(out_dir).mkdir(exist_ok=True)

        context = multiprocessing.get_context("fork")
        args = [(forest,) for forest in self.survey.data.forests]
        with context.Pool(processes=num_processors) as pool:
            forests = pool.starmap(
                compute_RF_statistics,
                args,
            )

        logger.info("Saving mask data.")
        healpixs = np.array([forest.healpix for forest in forests])
        unique_healpixs = np.unique(healpixs)

        arguments = []
        for healpix in unique_healpixs:
            this_idx = np.nonzero(healpix == healpixs)[0]
            grouped_forests = [forests[i] for i in this_idx]
            arguments.append((out_dir, healpix, grouped_forests, sorted_ids[healpix]))

        context = multiprocessing.get_context("fork")
        with context.Pool(processes=num_processors) as pool:
            pool.starmap(
                save_healpix_data_mask,
                arguments,
            )

    def save_data_flux(
        self,
        out_dir: Path | str,
        sorted_ids: Dict,
        num_processors: Optional[int] = None,
    ) -> None:
        Path(out_dir).mkdir(exist_ok=True)

        logger.info("Saving flux data.")
        forests = self.survey.data.forests
        healpixs = np.array([forest.healpix for forest in forests])
        unique_healpixs = np.unique(healpixs)

        arguments = []
        for healpix in unique_healpixs:
            this_idx = np.nonzero(healpix == healpixs)[0]
            grouped_forests = [forests[i] for i in this_idx]
            arguments.append((out_dir, healpix, grouped_forests, sorted_ids[healpix]))

        context = multiprocessing.get_context("fork")
        with context.Pool(processes=num_processors) as pool:
            pool.starmap(
                save_healpix_data_flux,
                arguments,
            )

    def read_original_deltas_ids(self, in_dir: Path | str) -> Dict:
        logger.info("Reading original LOS_ID sorting.")
        target_ids = dict()
        for delta_file in Path(in_dir).glob("delta-*.fits.gz"):
            healpix = int(delta_file.name.split("-")[1].split(".")[0])
            with fitsio.FITS(delta_file) as hdul:
                target_ids[healpix] = hdul["METADATA"].read()["LOS_ID"]

        return target_ids


def compute_RF_statistics(
    forest: Forest,
) -> Forest:
    bins = find_bins(
        10**forest.log_lambda / (1 + forest.z),
        10**Forest.log_lambda_rest_frame_grid,
    )

    count = np.bincount(bins, minlength=len(Forest.log_lambda_rest_frame_grid))
    w = count > 0

    rebin = np.bincount(
        bins, weights=forest.dla_mask, minlength=len(Forest.log_lambda_rest_frame_grid)
    )
    forest.dla_mask_rf = np.array(rebin, dtype=float)
    forest.dla_mask_rf[w] /= count[w]

    rebin = np.bincount(
        bins, weights=forest.bal_mask, minlength=len(Forest.log_lambda_rest_frame_grid)
    )
    forest.bal_mask_rf = np.array(rebin, dtype=float)
    forest.bal_mask_rf[w] /= count[w]

    rebin = np.bincount(
        bins,
        weights=forest.dla_transmission,
        minlength=len(Forest.log_lambda_rest_frame_grid),
    )
    forest.dla_transmission_rf = np.array(rebin, dtype=float)
    forest.dla_transmission_rf[w] /= count[w]

    forest.log_lambda_rest_frame_index = np.unique(bins)

    return forest


def save_healpix_data_mask(
    out_dir: Path | str,
    healpix: int,
    forests: List[Forest],
    sorted_ids: List[int],
) -> None:
    out_dir = Path(out_dir)

    # Sorting forests by LOS_ID will make identification faster
    forests = sorted(forests, key=lambda x: x.los_id)
    ids = [forest.los_id for forest in forests]

    # Check all original LOS_IDs are present
    if not np.all(np.in1d(sorted_ids, ids)):
        logger.warning("Input deltas have LOS_IDS that do not appear on extended data")

    # We search each sorted_ids into ids (which is sorted numerically to speed up the process)
    save_indexs = np.searchsorted(ids, sorted_ids)
    save_forests = np.asarray(forests)[save_indexs]

    with fitsio.FITS(
        out_dir / f"mask-info-{healpix}.fits.gz", "rw", clobber=True
    ) as hdul:
        hdul.write(
            np.array(
                [tuple(forest.get_metadata()) for forest in save_forests],
                dtype=save_forests[0].get_metadata_dtype(),
            ),
            extname="METADATA",
        )

        hdul.write(
            10**Forest.log_lambda_grid,
            extname="LAMBDA",
        )

        hdul.write(
            10**Forest.log_lambda_rest_frame_grid,
            extname="LAMBDA_RF",
        )

        in_mask = np.full((len(save_forests), len(Forest.log_lambda_grid)), 0)
        for i, forest in enumerate(save_forests):
            in_mask[i][forest.log_lambda_index] = np.ones_like(
                forest.log_lambda_index, dtype=int
            )
        hdul.write(
            in_mask,
            extname="INPUT_MASK",
        )

        dla_mask = np.full((len(save_forests), len(Forest.log_lambda_grid)), False)
        for i, forest in enumerate(save_forests):
            dla_mask[i][forest.log_lambda_index] = forest.dla_mask
        hdul.write(
            np.array(dla_mask, dtype=int),
            extname="DLA_MASK",
        )

        dla_transmission = np.full(
            (len(save_forests), len(Forest.log_lambda_grid)),
            0,
            dtype=float,
        )
        for i, forest in enumerate(save_forests):
            dla_transmission[i][forest.log_lambda_index] = forest.dla_transmission
        hdul.write(dla_transmission, extname="DLA_TRANSMISSION")

        bal_mask = np.full((len(save_forests), len(Forest.log_lambda_grid)), False)
        for i, forest in enumerate(save_forests):
            bal_mask[i][forest.log_lambda_index] = forest.bal_mask
        hdul.write(np.array(bal_mask, dtype=int), extname="BAL_MASK")

        # SAVE RF properties
        hdul.write(
            10**Forest.log_lambda_rest_frame_grid,
            extname="LAMBDA_RF",
        )

        in_mask = np.full(
            (len(save_forests), len(Forest.log_lambda_rest_frame_grid)), 0
        )
        for i, forest in enumerate(save_forests):
            in_mask[i][forest.log_lambda_rest_frame_index] = np.ones_like(
                forest.log_lambda_rest_frame_index, dtype=int
            )
        hdul.write(
            in_mask,
            extname="INPUT_MASK_RF",
        )

        dla_mask_rf = np.asarray([forest.dla_mask_rf for forest in save_forests])
        hdul.write(
            np.array(dla_mask_rf, dtype=int),
            extname="DLA_MASK_RF",
        )

        dla_transmission_rf = np.asarray(
            [forest.dla_transmission_rf for forest in save_forests]
        )
        hdul.write(dla_transmission_rf, extname="DLA_TRANSMISSION_RF")

        bal_mask_rf = np.asarray([forest.bal_mask_rf for forest in save_forests])
        hdul.write(np.array(bal_mask_rf, dtype=int), extname="BAL_MASK_RF")


def save_healpix_data_flux(
    out_dir: Path | str,
    healpix: int,
    forests: List[Forest],
    sorted_ids: List[int],
) -> None:
    out_dir = Path(out_dir)

    # Sorting forests by LOS_ID will make identification faster
    forests = sorted(forests, key=lambda x: x.los_id)
    ids = [forest.los_id for forest in forests]

    # Check all original LOS_IDs are present
    if not np.all(np.in1d(sorted_ids, ids)):
        logger.warning("Input deltas have LOS_IDS that do not appear on extended data")

    # We search each sorted_ids into ids (which is sorted numerically to speed up the process)
    save_indexs = np.searchsorted(ids, sorted_ids)
    save_forests = np.asarray(forests)[save_indexs]

    with fitsio.FITS(
        out_dir / f"flux-info-{healpix}.fits.gz", "rw", clobber=True
    ) as hdul:
        hdul.write(
            np.array(
                [tuple(forest.get_metadata()) for forest in save_forests],
                dtype=save_forests[0].get_metadata_dtype(),
            ),
            extname="METADATA",
        )

        flux = np.full((len(save_forests), len(Forest.log_lambda_grid)), np.nan)
        for i, forest in enumerate(save_forests):
            flux[i][forest.log_lambda_index] = forest.flux

        hdul.write(
            flux,
            extname="FLUX",
        )

        ivar = np.full((len(save_forests), len(Forest.log_lambda_grid)), np.nan)
        for i, forest in enumerate(save_forests):
            ivar[i][forest.log_lambda_index] = forest.ivar

        hdul.write(
            ivar,
            extname="FLUX_IVAR",
        )


def read_DLA_mask_single(forest: Forest, mask: DlaMask) -> Forest:
    lambda_ = 10**forest.log_lambda

    # load DLAs
    if mask.los_ids.get(forest.los_id) is not None:
        dla_transmission = np.ones(len(lambda_))
        for z_abs, nhi in mask.los_ids.get(forest.los_id):
            dla_transmission *= dla_profile(lambda_, z_abs, nhi)

        # find out which pixels to mask
        w = dla_transmission > mask.dla_mask_limit
        if len(mask.mask) > 0:
            for mask_range in mask.mask:
                for z_abs, nhi in mask.los_ids.get(forest.los_id):
                    w &= (lambda_ / (1.0 + z_abs) < mask_range["wave_min"]) | (
                        lambda_ / (1.0 + z_abs) > mask_range["wave_max"]
                    )

        # do the actual masking
        forest.transmission_correction *= dla_transmission
        forest.dla_transmission = dla_transmission
        forest.dla_mask = w
    else:
        forest.dla_mask = np.ones_like(lambda_)
        forest.dla_transmission = forest.transmission_correction

    return forest


def read_BAL_mask_single(forest: Forest, mask: BalMask) -> Forest:
    if not hasattr(forest, "los_id"):
        pass
    if not hasattr(mask, "los_ids"):
        pass

    mask_table = mask.los_ids.get(forest.los_id)
    if (mask_table is None) or len(mask_table) == 0:
        forest.bal_mask = np.ones_like(forest.log_lambda)
        return forest

    # find out which pixels to mask
    w = np.ones(forest.log_lambda.size, dtype=bool)
    rest_frame_log_lambda = forest.log_lambda - np.log10(1.0 + forest.z)
    mask_idx_ranges = np.searchsorted(
        rest_frame_log_lambda,
        [mask_table["log_lambda_min"], mask_table["log_lambda_max"]],
    ).T
    # Make sure first index comes before the second
    mask_idx_ranges.sort(axis=1)

    for idx1, idx2 in mask_idx_ranges:
        w[idx1:idx2] = 0

    forest.bal_mask = w

    return forest


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = getArgs()

    level = logging.getLevelName(args.log_level)
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(levelname)s:%(message)s",
    )

    read = ReadExtraDeltasData(args.picca_config)
    read.read_data()

    target_ids = read.read_original_deltas_ids(in_dir=args.deltas_input)
    if args.output_dir_mask is not None:
        read.read_DLA_mask(args.num_processors)
        read.read_BAL_mask(args.num_processors)
        read.save_data_mask(
            out_dir=args.output_dir_mask,
            sorted_ids=target_ids,
            num_processors=args.num_processors,
        )

    if args.output_dir_flux is not None:
        read.survey.apply_masks()
        read.survey.filter_forests()

        read.save_data_flux(
            out_dir=args.output_dir_flux,
            sorted_ids=target_ids,
            num_processors=args.num_processors,
        )
    logger.info("Done.")


def getArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deltas-input",
        type=Path,
        required=True,
        help="Deltas to add information to (used for sorting)",
    )
    parser.add_argument(
        "--picca-config",
        type=Path,
        required=True,
        help="picca configuration file (output is better since it has all the info.)",
    )

    parser.add_argument(
        "--num-processors",
        type=int,
        help="num of processors to use with the multiprocessing tool.",
    )

    parser.add_argument(
        "--output-dir-mask",
        type=Path,
        required=False,
        default=None,
        help="Output directory to save mask info into.",
    )

    parser.add_argument(
        "--output-dir-flux",
        type=Path,
        required=False,
        default=None,
        help="Output directory to save flux info into.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
