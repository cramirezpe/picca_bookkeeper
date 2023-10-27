""" FakeBookkeeper class to use for plotting non-bookkeeper runs"""
from __future__ import annotations

import configparser
import copy
import filecmp
import logging
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from importlib_resources import files
from yaml import SafeDumper

from picca_bookkeeper.bookkeeper import Bookkeeper, PathBuilder

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class FakeBookkeeper(Bookkeeper):
    def __init__(self, attributes_files: Dict=dict(), export_files:Dict=dict(), fit_file: Optional[Path | str] = None):
        self.paths = FakePaths(
            attributes_files=attributes_files,
            export_files=export_files,
            fit_file=fit_file,
        )

class FakePaths(PathBuilder):
    def __init__(self, attributes_files: Dict =dict(), export_files: Dict=dict(), fit_file: Optional[Path | str]=None):
        self.attributes_files = attributes_files
        self.export_files = export_files
        if fit_file is not None:
            self.fit_file  = Path(fit_file)

    def delta_attributes_file(self, region:str, calib_step: Optional[int]=None) -> Path:
        return Path(self.attributes_files[region])

    def exp_cf_fname(
        self, absorber: str,region:str, absorber2: Optional[str] = None, region2: Optional[str]=None
    ) -> Path:
        if absorber is None:
            absorber = "lya"
        if region is None:
            region = "lya"
        if absorber2 is None:
            absorber2 = "lya"
        if region2 is None:
            region2 = "lya"

        return Path(self.export_files[f"{absorber}{region}_{absorber2}{region2}"])

    def exp_xcf_fname(self, absorber: str,region:str)-> Path:
        if absorber is None:
            absorber = "lya"
        if region is None:
            region = "lya"
        return Path(self.export_files[f"{absorber}{region}"])
    
    def fit_out_fname(self):
        return self.fit_file