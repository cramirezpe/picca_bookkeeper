"""
    Script to combine multiple DLA catalogs.
"""
import argparse, textwrap
import fitsio
import astropy
from astropy.table import Table, vstack, unique
from pathlib import Path
import logging
import sys
import numpy as np

logger = logging.getLogger(__name__)


def select_table(table: astropy.table.table.Table, selection: int = 2):
    """Select on confidence flags/snr, also select type==DLA only"""
    # print('first select unique DLA ID')
    # tab = select_table_1(tab)

    logger.info("Start selecting on conf./SNR")

    if selection == 1:
        conf = table["CNN_DLA_CONFIDENCE"]

        msk = table["ABSORBER_TYPE"] == "DLA"
        msk &= ((conf > 0.3) % (table["S2N"] > 0)) | (
            (conf > 0.2) & (table["S2N"] >= 3)
        )

    elif selection == 2:
        conf_min = np.minimum(
            table["CNN_DLA_CONFIDENCE"],
            table["GP_DLA_CONFIDENCE"],
        )

        msk = table["ABSORBER_TYPE"] == "DLA"
        msk &= conf_min > 0.5
    elif selection == 3:
        conf_cnn = table["CNN_DLA_CONFIDENCE"]
        msk_cnn = ((conf_cnn > 0.3) & (table["S2N"] > 0)) | (
            (conf_cnn > 0.2) & (table["S2N"] >= 3)
        )

        msk_gp = table["GP_DLA_CONFIDENCE"] > 0.9

        msk = table["ABSORBER_TYPE"] == "DLA"
        msk &= msk_gp | msk_cnn

    else:
        raise ValueError("Selection not valid.")

    return table[msk]


def main(args=None):
    if args is None:
        args = getArgs()

    level = logging.getLevelName(args.log_level)
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(levelname)s:%(name)s:%(funcName)s:%(message)s",
    )

    logger.info("Reading DLA catalogues.")
    catalogues = [
        Table(fitsio.FITS(catalogue)[1].read()) for catalogue in args.dla_catalogues
    ]

    logger.info("Selecting objects from catalogues.")
    if args.selection != 0:
        catalogues = [
            select_table(catalogue, selection=args.selection)
            for catalogue in catalogues
        ]

    logger.info("Combining catalogues.")
    final_catalogue = vstack(catalogues)
    final_catalogue = unique(final_catalogue)
    columns = list(final_catalogue.columns)

    logger.info("Writing final catalogue.")
    with fitsio.FITS(args.output_catalogue, "rw", clobber=True) as results:
        header = {}
        results.write(
            [final_catalogue[column] for column in columns],
            names=columns,
            header=header,
            extname="DLACAT",
        )
    logger.info("Done.")


def getArgs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "dla_catalogues",
        type=Path,
        nargs="+",
        help="Input DLA catalogs",
    )

    parser.add_argument(
        "--output-catalogue",
        type=Path,
        required=True,
        help="Output path to the final DLA catalogue.",
    )

    parser.add_argument(
        "--selection",
        type=int,
        default=0,
        choices=[
            0,
            1,
            2,
            3,
        ],
        help=textwrap.dedent(
            """
0: No cuts.
1: First recommendation by DLA Team.
    Select DLA absorber type.
    Select absorbers with 'DLA_CONFIDENCE'>0.2 as valid detections for 'S2N'>3, 'DLA_CONFIDENCE'>0.3 for 'S2N'<3.
    Only use CNN.
2: Second selection (IFAE suggestion). Built to generate approximately 1 DLA for each 3 QSOs.
    Select DLA absorber type.
    Select absorbers detected by both CNN and GP with confidence>0.5.
            """
        ),
    )

    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
