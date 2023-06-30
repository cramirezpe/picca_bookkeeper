""" Script to run vega fit given a bookkeeper config file"""
from pathlib import Path
import argparse
from picca_bookkeeper.bookkeeper import Bookkeeper
from picca_bookkeeper.dict_utils import DictUtils
import copy


def main(args=None):
    if args is None:
        args = get_args()

    bookkeeper = Bookkeeper(
        args.bookkeeper_config,
        read_mode=True,
    )
    if bookkeeper.correlations is None:
        raise ValueError("Bookkeeper should contain correlations information.")

    if bookkeeper.fits == None:
        bookkeeper.fits = dict()

    if args.run_name is None:
        raise ValueError("Should specify run name for the fits")
    bookkeeper.fits["run name"] = args.run_name

    bookkeeper.fits["delta extraction"] = bookkeeper.paths.continuum_tag
    bookkeeper.fits["correlation run name"] = bookkeeper.paths.correlations_path.name

    # Generate all the structure if it is not already there in the bookkeeper config
    # The defined fields in the config file will remain the same.
    if "extra args" not in bookkeeper.fits:
        bookkeeper.fits["extra args"] = dict()

    bookkeeper.fits["extra args"] = DictUtils.merge_dicts(
        {
            "vega_auto": {
                "data": {},
                "cuts": {},
                "model": {},
                "metals": {},
            },
            "vega_cross": {
                "data": {},
                "cuts": {},
                "model": {},
                "metals": {},
            },
            "vega_main": {
                "data sets": {},
                "cosmo-fit type": {},
                "fiducial": {},
                "control": {},
                "output": {},
                "Polychord": {},
                "sample": {},
                "parameters": {},
            },
        },
        bookkeeper.fits["extra args"],
    )

    auto_correlations = [
        "lya.lya-lya.lya",
    ]
    cross_correlations = [
        "lya.lya",
    ]

    if args.lyb:
        auto_correlations.append("lya.lya-lya.lyb")
        cross_correlations.append("lya.lyb")

    if args.no_bao:
        bookkeeper.fits["extra args"] = DictUtils.merge_dicts(
            bookkeeper.fits["extra args"],
            {
                "remove_vega_main": {  # This removes fields
                    "sample": {
                        "ap": "",
                        "at": "",
                    }
                },
                "vega_main": {  # This adds/modifies fields
                    "parameters": {
                        "bao_amp": 0,
                    }
                },
            },
        )

    if args.no_hcd:
        bookkeeper.fits["extra args"] = DictUtils.merge_dicts(
            bookkeeper.fits["extra args"],
            {
                "remove_vega_main": {
                    "sample": {
                        "bias_hcd": "",
                        "beta_hcd": "",
                    }
                },
                "vega_auto": {
                    "model": {
                        "model-hcd": "None",
                    }
                },
                "vega_cross": {
                    "model": {
                        "model-hcd": "None",
                    }
                },
            },
        )

    if args.no_metals:
        bookkeeper.fits["extra args"] = DictUtils.merge_dicts(
            bookkeeper.fits["extra args"],
            {
                "remove_vega_auto": {
                    "metals": "",
                },
                "remove_vega_cross": {
                    "metals": "",
                },
                "remove_vega_main": {
                    "sample": {
                        "bias_eta_SiII(1190)": "",
                        "bias_eta_SiII(1193)": "",
                        "bias_eta_SiII(1260)": "",
                        "bias_eta_SiIII(1207)": "",
                    }
                },
            },
        )

    if args.no_sky:
        bookkeeper.fits["extra args"] = DictUtils.merge_dicts(
            bookkeeper.fits["extra args"],
            {
                "remove_vega_main": {
                    "sample": {
                        "desi_inst_sys_amp": "",
                    }
                },
                "vega_main": {
                    "parameters": {
                        "desi_inst_sys_amp": "0",
                    }
                },
            },
        )

    if args.no_qso_rad:
        bookkeeper.fits["extra args"] = DictUtils.merge_dicts(
            bookkeeper.fits["extra args"],
            {
                "remove_vega_main": {
                    "sample": {
                        "qso_rad_strength": "",
                    }
                },
                "vega main": {
                    "parameters": {
                        "qso_rad_strength": "0",
                    }
                },
            },
        )

    if args.rmin_cf is not None:
        bookkeeper.fits["extra args"] = DictUtils.merge_dicts(
            bookkeeper.fits["extra args"],
            {
                "vega_auto": {
                    "cuts": {
                        "r-min": str(args.rmin_cf),
                    }
                }
            },
        )
    if args.rmax_cf is not None:
        bookkeeper.fits["extra args"] = DictUtils.merge_dicts(
            bookkeeper.fits["extra args"],
            {
                "vega_auto": {
                    "cuts": {
                        "r-max": str(args.rmax_cf),
                    }
                }
            },
        )
    if args.rmin_xcf is not None:
        bookkeeper.fits["extra args"] = DictUtils.merge_dicts(
            bookkeeper.fits["extra args"],
            {
                "vega_cross": {
                    "cuts": {
                        "r-min": str(args.rmin_xcf),
                    }
                }
            },
        )
    if args.rmax_xcf is not None:
        bookkeeper.fits["extra args"] = DictUtils.merge_dicts(
            bookkeeper.fits["extra args"],
            {
                "vega_cross": {
                    "cuts": {
                        "r-max": str(args.rmax_xcf),
                    }
                }
            },
        )

    out_config = copy.deepcopy(bookkeeper.config)
    bookkeeper.write_bookkeeper(out_config, args.out_config)

    # The previous one was a full bookkeeper file, but we want
    # to generate one with only the fits section:
    out_config.pop("delta extraction")
    out_config.pop("correlations")

    # Now we load the run itself.
    if args.send_job:
        bookkeeper_run = Bookkeeper(args.out_config)

        fit = bookkeeper_run.get_fit_tasker(
            auto_correlations=auto_correlations,
            cross_correlations=cross_correlations,
            wait_for=args.wait_for,
        )
        fit.write_job()

        fit.send_job()
        print(fit.jobid)
        return fit.jobid
    else:
        return


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bookkeeper_config", type=Path, help="Path to bookkeeper file to use."
    )

    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Name of the fit run (if not specified in bookkeeper file)",
    )

    parser.add_argument(
        "--out-config",
        type=Path,
        default=".fit_bookkeeper.yaml",
        help="Store configuration file before loading bookkeeper.",
    )

    parser.add_argument("--no-bao", action="store_true", default=False, required=False)
    parser.add_argument("--no-hcd", action="store_true", default=False, required=False)
    parser.add_argument(
        "--no-metals", action="store_true", default=False, required=False
    )
    parser.add_argument("--no-sky", action="store_true", default=False, required=False)
    parser.add_argument(
        "--no-qso-rad", action="store_true", default=False, required=False
    )
    parser.add_argument("--lyb", action="store_true", default=False, required=False)

    parser.add_argument("--rmin-cf", type=int, default=None, required=False)
    parser.add_argument("--rmax-cf", type=int, default=None, required=False)
    parser.add_argument("--rmin-xcf", type=int, default=None, required=False)
    parser.add_argument("--rmax-xcf", type=int, default=None, required=False)

    parser.add_argument(
        "--send-job", action="store_true", default=False, required=False
    )

    parser.add_argument("--wait-for", nargs="+", type=int, default=None, required=False)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
