"""
mix_DLA_catalogues.py
---------------------

Script to combine and filter multiple Damped Lyman-Alpha (DLA) catalogues into
a single, catalogue. Enables flexible selection criteria for DLA candidates and
standardized catalogue merging, supporting large-scale Lyman-alpha forest analyses.

Functionality:
---------------
    - Reads multiple DLA catalogue FITS files and selects DLA candidates based
      on configurable confidence, S2N (signal-to-noise), and NHI (neutral hydrogen
      column density) thresholds.
    - Offers a variety of selection modes tailored to common requirements.
    - Combines and deduplicates input catalogues into a single output FITS file
      with a standardized format.

Arguments:
----------
    - Positional:
        dla_catalogues [Path]: One or more input FITS DLA catalogue files.
    - Required:
        --output-catalogue [Path]: Output path for the combined FITS catalogue.
    - Optional:
        --NHI [float]: Minimum DLA NHI (only used in selection 5 or 6,
                                        default=20.3).
        --S2N [float]: Minimum QSO S2N (only used in selection 5 or 6,
                                        default=0).
        --selection [int]: Filtering mode (see below;
                                        default=0 for no filtering).
        --log-level [str]: Logging verbosity (default=INFO).

Selection Modes:
----------------
     0: No cuts (all objects included).
     2: IFAE suggestion – DLA type, detected by both CNN and GP with confidence > 0.5.
     3: DLA paper recommendation – combined CNN/GP criteria based on S2N and confidence.
     4: DLA team original – CNN-based confidence/S2N thresholds.
     5: Both CNN and GP confidence > 0.5, NHI > --NHI, S2N > --S2N.
     6: Use GP NHI if available, else CNN NHI + 0.17; NHI > --NHI, S2N > --S2N.

Usage Example:
--------------
Combine three DLA catalogues into one, using selection mode 2 and custom output:
    python mix_DLA_catalogues.py cat1.fits cat2.fits cat3.fits \
        --output-catalogue merged.fits \
        --selection 2

Combine with stricter NHI/S2N thresholds:
    python mix_DLA_catalogues.py cat1.fits cat2.fits \
        --output-catalogue merged_strict.fits \
        --selection 5 --NHI 21 --S2N 3

"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import astropy
import fitsio
import numpy as np
from astropy.table import Table, unique, vstack

if TYPE_CHECKING:
    from typing import Dict, Optional

logger = logging.getLogger(__name__)


def select_table(
    table: astropy.table.table.Table,
    selection: int = 2,
    NHI: float = 20.3,
    S2N: float = 0,
) -> astropy.table.table.Table:
    """
    Filters a DLA catalogue `astropy.table.Table` based on a specified selection
    mode, neutral hydrogen column density (NHI), and signal-to-noise ratio (S2N)
    thresholds. (Select on confidence flags/snr, also select type==DLA only.)

    Arguments:
    ----------
        - table (astropy.table.Table): Input DLA catalogue with required columns
            depending on selection mode, e.g., 'DLA_CONFIDENCE', 'S2N', 'NHI',
            'CNN_DLA_CONFIDENCE', 'GP_DLA_CONFIDENCE', 'ABSORBER_TYPE', etc.
        - selection (int, optional): Filtering strategy to apply. See script-level
            docstring for available options. Default is 2.
        - NHI (float, optional): Minimum NHI value. Only used in selections 5 and 6.
            Default is 20.3.
        - S2N (float, optional): Minimum signal-to-noise ratio. Only used in
            selections 5 and 6. Default is 0.

    Returns:
    --------
        astropy.table.Table: Filtered table of DLA candidates meeting the
                             selection criteria.

    Raises:
    -------
        ValueError: If an invalid `selection` number is provided.

    Note:
    -----
        - For selection mode 6, if GP_NHI is unavailable (value == 0), CNN NHI
            is corrected by adding +0.17 as a proxy.
        - All selection modes require 'ABSORBER_TYPE' == "DLA" as a baseline filter.
    """
    # print('first select unique DLA ID')
    # tab = select_table_1(tab)

    logger.info("Start selecting on conf./SNR")

    if selection == 2:
        if "CNN_DLA_CONFIDENCE" in table.keys() and "GP_DLA_CONFIDENCE" in table.keys():
            conf_min = np.minimum(
                table["CNN_DLA_CONFIDENCE"],
                table["GP_DLA_CONFIDENCE"],
            )
        else:
            conf_min = table["DLA_CONFIDENCE"]

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
    elif selection == 4:
        conf = table["CNN_DLA_CONFIDENCE"]

        msk = table["ABSORBER_TYPE"] == "DLA"
        msk &= ((conf > 0.3) & (table["S2N"] > 0)) | (
            (conf > 0.2) & (table["S2N"] >= 3)
        )
    elif selection == 5:
        conf_min = np.minimum(
            table["CNN_DLA_CONFIDENCE"], table["GP_DLA_CONFIDENCE"])
        msk = table["NHI"] > NHI
        msk &= conf_min > 0.5
        msk &= table["S2N"] > S2N
    elif selection == 6:
        # gp_nhi being 0 means nhi is taken from cnn
        # we try correcting the cnn_nhi by adding 0.17
        idx = np.where(table["GP_NHI"] == 0)
        table["NHI"][idx] += 0.17

        msk = table["NHI"] > NHI
        msk &= table["CNN_DLA_CONFIDENCE"] > 0.5
        msk &= table["S2N"] > S2N

    else:
        raise ValueError("Selection not valid.")

    return table[msk]


def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    Main function for merging and filtering DLA catalogues based on CLI args.

    Functionality:
    --------------
        - Parses command-line arguments.
        - Reads multiple input DLA FITS catalogues into Astropy Tables.
        - Applies user-specified selection/filtering criteria (optional).
        - Stacks and deduplicates all tables into a final catalogue.
        - Writes the result to an output FITS file in standardized format.

    Arguments:
    ----------
        args (Optional[argparse.Namespace]): Parsed arguments from argparse.
            If None, arguments are parsed from sys.argv using getArgs().

    Raises:
    -------
        SystemExit: If argparse fails or required arguments are missing.
    """
    if args is None:
        args = getArgs()

    level = logging.getLevelName(args.log_level)
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(levelname)s:%(message)s",
    )

    logger.info("Reading DLA catalogues.")
    catalogues = [
        Table(fitsio.FITS(catalogue)[1].read()) for catalogue in args.dla_catalogues
    ]

    logger.info("Selecting objects from catalogues.")
    if args.selection != 0:
        catalogues = [
            select_table(
                catalogue, selection=args.selection, NHI=args.NHI, S2N=args.S2N
            )
            for catalogue in catalogues
        ]

    logger.info("Combining catalogues.")
    final_catalogue = vstack(catalogues)
    final_catalogue = unique(final_catalogue)
    columns = list(final_catalogue.columns)

    logger.info("Writing final catalogue.")
    with fitsio.FITS(args.output_catalogue, "rw", clobber=True) as results:
        header: Dict = {}
        results.write(
            [final_catalogue[column] for column in columns],
            names=columns,
            header=header,
            extname="DLACAT",
        )
    logger.info("Done.")


def getArgs() -> argparse.Namespace:
    """
    Parses command-line arguments for the DLA catalogue merging script.

    Returns:
    --------
        argparse.Namespace: Namespace containing parsed arguments:
            - dla_catalogues (List[Path]): One or more input FITS catalogue paths.
            - output_catalogue (Path): Path to the output FITS catalogue.
            - NHI (float): Minimum NHI threshold for DLA selection (default=20.3).
            - S2N (float): Minimum S2N threshold for QSO selection (default=0).
            - selection (int): Filtering strategy to use (see script docstring).
            - log_level (str): Logging verbosity (default='INFO').

    Raises:
    -------
        argparse.ArgumentError: If required arguments are missing or invalid.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
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
        "--NHI",
        type=float,
        default=20.3,
        required=False,
        help="minimum DLA NHI, used only in selection 5 or 6",
    )

    parser.add_argument(
        "--S2N",
        type=float,
        default=0,
        required=False,
        help="minimum QSO S2N, used only in selection 5 or 6",
    )

    parser.add_argument(
        "--selection",
        type=int,
        default=0,
        choices=[
            0,
            2,
            3,
            4,
            5,
            6,
        ],
        help=textwrap.dedent(
            """
0: No cuts.
1: Invalid (to reserve v1 for full catalog.)
2: (IFAE suggestion). Built to generate approximately 1 DLA for each 3 QSOs.
    - Select DLA absorber type.
    - Select absorbers detected by both CNN and GP with confidence>0.5.
3: DLA paper recommendation.
    - Select DLA absorber type
    - Select absorbers fulfilling any:
        - For CNN: select absorbers with 'DLA_CONFIDENCE'>0.2 as valid detections
          for 'S2N'>3, 'DLA_CONFIDENCE'>0.3 for 'S2N'<3.
        - For GP: select absorbers with ¡DLA_CONFIDENCE'>0.9.
4: Original recommendation by DLA Team.
    - Select DLA absorber type.
    - Select absorbers with 'DLA_CONFIDENCE'>0.2 as valid detections for 'S2N'>3,
      'DLA_CONFIDENCE'>0.3 for 'S2N'<3.
    - Only use CNN.
5: CONF_CNN > 0.5 and CONF_GP > 0.5 and NHI > args.NHI and S2N > args.S2N
6: CONF_CNN > 0.5, if NHI_GP available use else use NHI_CNN + 0.17,
   NHI > args.NHI and S2N > args.S2N
            """
        ),
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
