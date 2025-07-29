"""
fake_bookkeeper.py

FakeBookkeeper class to use for plotting non-bookkeeper runs.

This module provides the FakeBookkeeper and FakePaths classes, which are
lightweight, mock implementations of the Bookkeeper and PathBuilder interfaces.

 - FakeBookkeeper can be used for plotting or testing scenarios where a real
   bookkeeper run is not available or not needed. It allows users to pass in
   custom file paths for attributes, exports, and fits, without requiring the
   full bookkeeping infrastructure.

 - FakePaths mimics the behavior of PathBuilder, returning file paths from
   user-provided dictionaries and inputs, enabling flexible, controlled access
   to file locations for downstream code or plotting utilities.

Usage: Development, testing, or quick visualization workflows where
       reproducibility and speed are prioritized over strict bookkeeping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from picca_bookkeeper.bookkeeper import Bookkeeper, PathBuilder

if TYPE_CHECKING:
    from typing import Dict, Optional

logger = logging.getLogger(__name__)


class FakeBookkeeper(Bookkeeper):
    """
    Mock Bookkeeper used for plotting or testing without requiring
    the full bookkeeping pipeline.

    Args:
        attributes_files (Dict): Mapping from region names to attribute file paths.
        export_files (Dict): Mapping from file labels to export file paths.
        fit_file (Path or str, optional): Path to the fit output file.

    Example:
        fbk = FakeBookkeeper(
            attributes_files={"lya": "path/to/lya_attr.fits"},
            export_files={"lyalya_lyalya": "path/to/corr.fits"},
            fit_file="path/to/fit.json")
    """

    def __init__(
        self,
        attributes_files: Dict = dict(),
        export_files: Dict = dict(),
        fit_file: Optional[Path | str] = None,
    ):
        self.paths = FakePaths(
            attributes_files=attributes_files,
            export_files=export_files,
            fit_file=fit_file,
        )


class FakePaths(PathBuilder):
    """
    Mock PathBuilder used to return paths to attributes, correlation
    functions, and fit outputs using user-defined dictionaries.

    Args:
        attributes_files (Dict): Dictionary mapping region names to attribute
                                file paths.
        export_files (Dict): Dictionary mapping file identifiers to export
                                file paths.
        fit_file (Path or str, optional): Path to the fit output file.
    """

    def __init__(
        self,
        attributes_files: Dict = dict(),
        export_files: Dict = dict(),
        fit_file: Optional[Path | str] = None,
    ):
        self.attributes_files = attributes_files
        self.export_files = export_files
        if fit_file is not None:
            self.fit_file = Path(fit_file)

    def delta_attributes_file(
        self, region: Optional[str] = None, calib_step: Optional[int] = None
    ) -> Path:
        """
        Returns the path to the delta attributes file for a given region.

        Args:
            region (str, optional): Region key used in the attributes_files
                                    dictionary.
            calib_step (int, optional): Calibration step (ignored in FakePaths).

        Returns:
            Path: File path to the delta attributes file.
        """
        return Path(self.attributes_files[region])

    def exp_cf_fname(
        self,
        absorber: str,
        region: str,
        absorber2: Optional[str] = None,
        region2: Optional[str] = None,
    ) -> Path:
        """
        Returns the path to the exported correlation function file.

        Args:
            absorber (str): First absorber type (e.g., "lya").
            region (str): First region name.
            absorber2 (str, optional): Second absorber type. Defaults to absorber 1.
            region2 (str, optional): Second region name. Defaults to region 1.

        Returns:
            Path: File path to the exported correlation function file.
        """
        if absorber is None:
            absorber = "lya"
        if region is None:
            region = "lya"
        if absorber2 is None:
            absorber2 = "lya"
        if region2 is None:
            region2 = "lya"

        return Path(self.export_files[f"{absorber}{region}_{absorber2}{region2}"])

    def exp_xcf_fname(self, absorber: str, region: str, tracer: str = "qso") -> Path:
        """
        Returns the path to the exported cross-correlation function file.

        Args:
            absorber (str): Absorber type (e.g., "lya").
            region (str): Region name.
            tracer (str, optional): Tracer type (default: "qso").

        Returns:
            Path: File path to the exported cross-correlation file.
        """
        if absorber is None:
            absorber = "lya"
        if region is None:
            region = "lya"
        return Path(self.export_files[f"{tracer}_{absorber}{region}"])

    def fit_out_fname(self) -> Path:
        """
        Returns the path to the fit output file.

        Returns:
            Path: File path to the fit result file.
        """
        return self.fit_file
