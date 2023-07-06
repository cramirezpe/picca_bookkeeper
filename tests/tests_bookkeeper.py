from pathlib import Path
import unittest
import shutil
from picca_bookkeeper.bookkeeper import Bookkeeper
import random
import codecs
from unittest.mock import patch
import itertools
import filecmp
import os

THIS_DIR = Path(__file__).parent


def mock_get_3d_catalog(
    release, survey, catalog, dla=False, bal=False, dla_version=None
):
    if dla:
        return THIS_DIR / "test_files" / "dummy_dla_catalog.fits"
    else:
        if bal:
            return THIS_DIR / "test_files" / "dummy_catalog-bal.fits"
        else:
            return THIS_DIR / "test_files" / "dummy_catalog.fits"


def mock_run(command, shell, capture_output):
    random.seed(command)

    class Out:
        stdout = codecs.encode(str(random.randint(10000000, 99999999)))
        returncode = 0

    return Out()


def write_full_analysis(bookkeeper, calib=False, region="lya", region2=None):
    if calib:
        calib = bookkeeper.get_calibration_extraction_tasker()
        calib.write_job()
        calib.send_job()

    deltas = bookkeeper.get_delta_extraction_tasker(
        region=region,
    )
    deltas.write_job()
    deltas.send_job()

    if region2 is not None:
        deltas2 = bookkeeper.get_delta_extraction_tasker(region=region2)
        deltas2.write_job()
        deltas2.send_job()
        wait_for = [deltas, deltas2]
    else:
        wait_for = deltas

    cf = bookkeeper.get_cf_tasker(
        region=region,
        region2=region2,
        wait_for=wait_for,
    )

    xcf = bookkeeper.get_xcf_tasker(
        region=region,
        wait_for=wait_for,
    )

    dmat = bookkeeper.get_dmat_tasker(
        region=region,
        region2=region2,
        wait_for=wait_for,
    )

    xdmat = bookkeeper.get_xdmat_tasker(
        region=region,
        wait_for=wait_for,
    )

    metal = bookkeeper.get_metal_tasker(
        region=region,
        region2=region2,
        wait_for=wait_for,
    )

    xmetal = bookkeeper.get_xmetal_tasker(
        region=region,
        wait_for=wait_for,
    )

    for task in (cf, dmat, metal, xcf, xdmat, xmetal):
        task.write_job()
        task.send_job()

    cf_exp = bookkeeper.get_cf_exp_tasker(
        region=region,
        region2=region2,
        wait_for=[cf, dmat, metal],
    )

    xcf_exp = bookkeeper.get_xcf_exp_tasker(
        region=region,
    )

    for task in (cf_exp, xcf_exp):
        task.write_job()
        task.send_job()

    fit = bookkeeper.get_fit_tasker(
        auto_correlations=["lya.lya-lya.lya"],
        cross_correlations=["lya.lya"],
    )

    fit.write_job()
    fit.send_job()


def copy_config_substitute(config, out_name="output"):
    out_path = THIS_DIR / "test_files" / out_name
    fitter_path = THIS_DIR.parent / "Examples" / "fit_config"
    out_file = out_path / "tmp.yaml"
    print(out_path)

    with open(config, "r") as file:
        filedata = file.read()
    for pattern, repl in zip(
        (
            "test_files",
            "fitter_directory",
            "results_directory",
        ),
        (out_path.parent, fitter_path, out_path),
    ):
        filedata = filedata.replace(pattern, str(repl))

    with open(out_file, "w") as file:
        file.write(filedata)


def rename_path(filename):
    with open(filename, "r") as file:
        filedata = file.read()

    filedata = filedata.replace(str(THIS_DIR.parent), "/picca_bookkeeper")

    with open(filename, "w") as file:
        file.write(filedata)


class TestBookkeeper(unittest.TestCase):
    files_path = THIS_DIR / "test_files"

    def setUp(self):
        (self.files_path / "output").mkdir(exist_ok=True)
        (self.files_path / "output2").mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.files_path / "output")
        shutil.rmtree(self.files_path / "output2")

    def compare_two_files(self, filename1, filename2):
        with open(filename1, "r") as file1, open(filename2, "r") as file2:
            try:
                self.assertListEqual(list(file1), list(file2))
            except AssertionError as e:
                raise AssertionError(
                    f"Files are different:\n\t{filename1}\n\t{filename2}"
                ).with_traceback(e.__traceback__)

    def compare_bookkeeper_output(self, test_folder, bookkeeper_folder):
        for root, dirs, files in os.walk(bookkeeper_folder):
            path = root[len(str(bookkeeper_folder)) + 1 :]
            for file in files:
                if file.split(".")[-1] not in ("fits", "gz"):
                    self.compare_two_files(Path(root) / file, test_folder / path / file)

    def replace_paths_bookkeeper_output(self, paths):
        for folder in (paths.run_path, paths.correlations_path, paths.fits_path):
            for file in itertools.chain(
                (folder / "scripts").iterdir(),
                (folder / "configs").iterdir(),
            ):
                if str(file.name).split(".")[-1] not in ("fits", "gz"):
                    rename_path(file)

    def update_test_output(self, test_folder, bookkeeper_folder):
        if test_folder.is_dir():
            shutil.rmtree(test_folder)

        shutil.copytree(bookkeeper_folder, test_folder)

    def test_config_read(self):
        with self.assertRaises(FileNotFoundError):
            bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "fake_cfg.yaml")

        copy_config_substitute(self.files_path / "example_config_guadalupe.yaml")
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        with self.assertRaises(ValueError):
            bookkeeper.validate_region("lyc")

    # @patch("builtins.input", return_value="yes")
    # def test_config_overwrite_on_yes(self, mock_print):
    #     copy_config_substitute(self.files_path / "example_config_guadalupe.yaml")
    #     bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

    #     copy_config_substitute(
    #         self.files_path / "example_config_guadalupe_conflict.yaml"
    #     )
    #     bookkeeper = Bookkeeper(
    #         THIS_DIR / "test_files" / "output" / "tmp.yaml", overwrite_config=False
    #     )

    #     self.assertTrue(
    #         filecmp.cmp(
    #             bookkeeper.paths.delta_config_file,
    #             THIS_DIR / "test_files" / "output" / "tmp.yaml",
    #         )
    #     )

    # @patch("builtins.input", return_value="no")
    # def test_config_overwrite_exit_on_no(self, mock_print):
    #     copy_config_substitute(self.files_path / "example_config_guadalupe.yaml")
    #     bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

    #     copy_config_substitute(
    #         self.files_path / "example_config_guadalupe_conflict.yaml"
    #     )

    #     with self.assertRaises(SystemExit):
    #         bookkeeper = Bookkeeper(
    #             THIS_DIR / "test_files" / "output" / "tmp.yaml", overwrite_config=False
    #         )

    # def test_catalog_changed(self):
    #     copy_config_substitute(self.files_path / "example_config_guadalupe.yaml")

    #     def side_effect(*args, **kwargs):
    #         return THIS_DIR / "test_files" / "dummy_catalog.fits"

    #     with patch("picca_bookkeeper.bookkeeper.get_quasar_catalog", side_effect=side_effect):
    #         bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

    #     def side_effect(*args, **kwargs):
    #         return THIS_DIR / "test_files" / "alt_cat" / "dummy_catalog.fits"

    #     with patch(
    #         "picca_bookkeeper.bookkeeper.get_quasar_catalog", side_effect=side_effect
    #     ), self.assertRaises(FileExistsError):
    #         bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_example_guadalupe(self, mock_func_1, mock_func_2):
        copy_config_substitute(self.files_path / "example_config_guadalupe.yaml")
        test_files = THIS_DIR / "test_files" / "guadalupe"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        write_full_analysis(bookkeeper, calib=True)

        self.replace_paths_bookkeeper_output(bookkeeper.paths)
        if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
            self.update_test_output(test_files, bookkeeper.paths.run_path)
        self.compare_bookkeeper_output(test_files, bookkeeper.paths.run_path)

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_example_guadalupe_perlmutter(self, mock_func_1, mock_func_2):
        copy_config_substitute(
            self.files_path / "example_config_guadalupe_perlmutter.yaml"
        )
        test_files = THIS_DIR / "test_files" / "guadalupe_perlmutter"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        write_full_analysis(bookkeeper, calib=True)

        self.replace_paths_bookkeeper_output(bookkeeper.paths)
        if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
            self.update_test_output(test_files, bookkeeper.paths.run_path)
        self.compare_bookkeeper_output(test_files, bookkeeper.paths.run_path)

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_invalid_calib(self, mock_func_1, mock_func_2):
        copy_config_substitute(self.files_path / "example_config_guadalupe.yaml")
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        bookkeeper.config["delta extraction"]["calib"] = 11
        with self.assertRaises(ValueError) as cm:
            deltas = bookkeeper.get_calibration_extraction_tasker()
        self.assertEqual(
            "Invalid calib value in config file. (Valid values are 0 1 2 3 10)",
            str(cm.exception),
        )

    # @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    # @patch("picca_bookkeeper.bookkeeper.get_quasar_catalog", side_effect=mock_get_3d_catalog)
    # def test_custom_calib(self, mock_func_1, mock_func_2):
    #     copy_config_substitute(self.files_path / "example_config_custom_calib_with_custom.yaml")
    #     test_files = THIS_DIR / "test_files" / "calib_custom_custom"
    #     bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

    #     write_full_analysis(bookkeeper, calib=True)

    #     self.replace_paths_bookkeeper_output(bookkeeper.paths)
    #     if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
    #         self.update_test_output(test_files, bookkeeper.paths.run_path)
    #     self.compare_bookkeeper_output(test_files, bookkeeper.paths.run_path)

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_example_guadalupe_custom_tracer_cat(self, mock_func_1, mock_func_2):
        copy_config_substitute(self.files_path / "example_config_guadalupe_tracer.yaml")
        test_files = THIS_DIR / "test_files" / "guadalupe_tracer"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        write_full_analysis(bookkeeper, calib=True)

        self.replace_paths_bookkeeper_output(bookkeeper.paths)
        if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
            self.update_test_output(test_files, bookkeeper.paths.run_path)
        self.compare_bookkeeper_output(test_files, bookkeeper.paths.run_path)

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_example_calib(self, mock_func_1, mock_func_2):
        copy_config_substitute(self.files_path / "example_config_guadalupe.yaml")
        test_files = THIS_DIR / "test_files" / "calib"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        write_full_analysis(bookkeeper, calib=True, region="lyb", region2="lya")

        self.replace_paths_bookkeeper_output(bookkeeper.paths)
        if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
            self.update_test_output(test_files, bookkeeper.paths.run_path)
        self.compare_bookkeeper_output(test_files, bookkeeper.paths.run_path)

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_example_calib_diff_path(self, mock_func_1, mock_func_2):
        copy_config_substitute(self.files_path / "example_config_guadalupe.yaml")
        test_files = THIS_DIR / "test_files" / "calib_diff_path"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        write_full_analysis(bookkeeper, calib=True, region="lyb", region2="lya")

        # self.replace_paths_bookkeeper_output(bookkeeper.paths)

        # Now main run:
        # copy_config_substitute(self.files_path / "example_config_guadalupe_calib_diff_path.yaml", out_name="output2")
        copy_config_substitute(
            self.files_path / "example_config_guadalupe_calib_diff_path.yaml"
        )

        with open(THIS_DIR / "test_files" / "output" / "tmp.yaml", "r") as file:
            filedata = file.read()

        filedata = filedata.replace(
            "step 1:", f"step 1: {str(bookkeeper.paths.delta_attributes_file(None, 1))}"
        )
        filedata = filedata.replace(
            "step 2:", f"step 2: {str(bookkeeper.paths.delta_attributes_file(None, 2))}"
        )

        with open(THIS_DIR / "test_files" / "output" / "tmp.yaml", "w") as file:
            file.write(filedata)

        bookkeeper2 = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        write_full_analysis(bookkeeper2, calib=False, region="lyb", region2="lya")

        self.replace_paths_bookkeeper_output(bookkeeper2.paths)
        if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
            self.update_test_output(test_files, bookkeeper2.paths.run_path)
        self.compare_bookkeeper_output(test_files, bookkeeper2.paths.run_path)

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_run_only_correlation_on_another_bookkeeper(self, mock_func_1, mock_func_2):
        copy_config_substitute(self.files_path / "example_config_guadalupe.yaml")
        test_files = THIS_DIR / "test_files" / "second_correlation"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        write_full_analysis(bookkeeper, calib=True, region="lyb", region2="lya")

        # Now failing run
        copy_config_substitute(
            self.files_path / "example_config_guadalupe_only_corr_fail.yaml"
        )

        with open(THIS_DIR / "test_files" / "output" / "tmp.yaml", "r") as file:
            filedata = file.read()

        with self.assertRaises(ValueError) as cm:
            bookkeeper2 = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")
        self.assertEqual(
            "delta extraction section of config file should match delta extraction "
            "section from file already in the bookkeeper.",
            str(cm.exception)[0:114],
        )

        # Now failing run (2)
        copy_config_substitute(
            self.files_path / "example_config_guadalupe_only_corr_fail_2.yaml"
        )

        with open(THIS_DIR / "test_files" / "output" / "tmp.yaml", "r") as file:
            filedata = file.read()

        with self.assertRaises(ValueError) as cm:
            bookkeeper2 = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")
        self.assertEqual(
            "correlations section of config file should match correlation section "
            "from file already in the bookkeeper.",
            str(cm.exception)[0:105],
        )

        # Now main run:
        copy_config_substitute(
            self.files_path / "example_config_guadalupe_only_corr.yaml"
        )

        with open(THIS_DIR / "test_files" / "output" / "tmp.yaml", "r") as file:
            filedata = file.read()

        bookkeeper2 = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        cf = bookkeeper2.get_cf_tasker(
            region="lyb",
            region2="lya",
            wait_for=None,
        )

        cf.write_job()
        cf.send_job()

        self.replace_paths_bookkeeper_output(bookkeeper2.paths)
        if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
            self.update_test_output(test_files, bookkeeper2.paths.run_path)
        self.compare_bookkeeper_output(test_files, bookkeeper2.paths.run_path)

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_run_only_fit_on_another_bookkeeper(self, mock_func_1, mock_func_2):
        copy_config_substitute(self.files_path / "example_config_guadalupe.yaml")
        test_files = THIS_DIR / "test_files" / "second_fit"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        write_full_analysis(bookkeeper, calib=True, region="lyb", region2="lya")

        # Now failing run
        copy_config_substitute(
            self.files_path / "example_config_guadalupe_only_fit_fail.yaml"
        )

        with open(THIS_DIR / "test_files" / "output" / "tmp.yaml", "r") as file:
            filedata = file.read()

        with self.assertRaises(ValueError) as cm:
            bookkeeper2 = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")
        self.assertEqual(
            "correlations section of config file should match correlation section from "
            "file already in the bookkeeper.",
            str(cm.exception)[0:105],
        )

        # Now failing run (2)
        copy_config_substitute(
            self.files_path / "example_config_guadalupe_only_fit_fail_2.yaml"
        )

        with open(THIS_DIR / "test_files" / "output" / "tmp.yaml", "r") as file:
            filedata = file.read()

        with self.assertRaises(ValueError) as cm:
            bookkeeper2 = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")
        self.assertEqual(
            "fits section of config file should match fits section from file already in the bookkeeper.",
            str(cm.exception)[0:90],
        )

        # Now main run:
        copy_config_substitute(
            self.files_path / "example_config_guadalupe_only_fit.yaml"
        )

        with open(THIS_DIR / "test_files" / "output" / "tmp.yaml", "r") as file:
            filedata = file.read()

        bookkeeper2 = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        fit = bookkeeper2.get_fit_tasker(
            auto_correlations=["lya.lya-lya.lya"],
            cross_correlations=["lya.lya"],
        )

        fit.write_job()
        fit.send_job()

        self.replace_paths_bookkeeper_output(bookkeeper2.paths)
        if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
            self.update_test_output(test_files, bookkeeper2.paths.run_path)
        self.compare_bookkeeper_output(test_files, bookkeeper2.paths.run_path)

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_example_custom(self, mock_func_1, mock_func_2):
        copy_config_substitute(self.files_path / "example_config_custom.yaml")
        test_files = THIS_DIR / "test_files" / "customconfig"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        write_full_analysis(bookkeeper, calib=False, region="lyb", region2="lya")

        self.replace_paths_bookkeeper_output(bookkeeper.paths)
        if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
            self.update_test_output(test_files, bookkeeper.paths.run_path)
        self.compare_bookkeeper_output(test_files, bookkeeper.paths.run_path)

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_example_custom_raise_if_calib(self, mock_func_1, mock_func_2):
        copy_config_substitute(self.files_path / "example_config_custom.yaml")
        test_files = THIS_DIR / "test_files" / "customconfig"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        with self.assertRaises(ValueError) as cm:
            write_full_analysis(bookkeeper, calib=True, region="lyb", region2="lya")
        self.assertEqual(
            "Trying to run calibration with calib=0 in config file.", str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            deltas = bookkeeper.get_delta_extraction_tasker(
                region="lya",
                calib_step=1,
            )
            deltas.write_job()
            deltas.send_job()
        self.assertEqual(
            "Trying to run calibration with calib = 0 in config file.",
            str(cm.exception),
        )

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_example_custom_calib(self, mock_func_1, mock_func_2):
        copy_config_substitute(self.files_path / "example_config_custom_calib.yaml")
        test_files = THIS_DIR / "test_files" / "customconfigcalib"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        with self.assertRaises(FileNotFoundError) as cm:
            deltas = bookkeeper.get_delta_extraction_tasker(
                region="lya",
            )
            deltas.write_job()
            deltas.send_job()
        self.assertEqual(
            "Calibration folder does not exist. run get_calibration_tasker before running deltas.",
            str(cm.exception),
        )

        write_full_analysis(bookkeeper, calib=True, region="lyb", region2="lya")

        self.replace_paths_bookkeeper_output(bookkeeper.paths)
        if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
            self.update_test_output(test_files, bookkeeper.paths.run_path)
        self.compare_bookkeeper_output(test_files, bookkeeper.paths.run_path)

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_example_custom_calib_raise_if_corrections_added(
        self, mock_func_1, mock_func_2
    ):
        copy_config_substitute(
            self.files_path / "example_config_custom_calib_with_calib_corrections.yaml"
        )
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        with self.assertRaises(ValueError) as cm:
            write_full_analysis(bookkeeper, calib=True, region="lyb", region2="lya")
        self.assertEqual(
            "Calibration corrections added by user with calib option != 10",
            str(cm.exception),
        )

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_example_guadalupe_dla_bal(self, mock_func_1, mock_func_2):
        copy_config_substitute(
            self.files_path / "example_config_guadalupe_dla_bal.yaml"
        )
        test_files = THIS_DIR / "test_files" / "guadalupe_dlabal"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        write_full_analysis(bookkeeper, calib=True, region="lyb", region2="lya")

        self.replace_paths_bookkeeper_output(bookkeeper.paths)
        if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
            self.update_test_output(test_files, bookkeeper.paths.run_path)
        self.compare_bookkeeper_output(test_files, bookkeeper.paths.run_path)

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_example_guadalupe_dla_fail(self, mock_func_1, mock_func_2):
        copy_config_substitute(
            self.files_path / "example_config_guadalupe_dla_bal_dla_fail.yaml"
        )
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        with self.assertRaises(ValueError) as cm:
            write_full_analysis(bookkeeper, calib=True, region="lyb", region2="lya")
        self.assertEqual("DlaMask set by user with dla option != 0", str(cm.exception))

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_example_guadalupe_bal_fail(self, mock_func_1, mock_func_2):
        copy_config_substitute(
            self.files_path / "example_config_guadalupe_dla_bal_bal_fail.yaml"
        )
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        with self.assertRaises(ValueError) as cm:
            write_full_analysis(bookkeeper, calib=True, region="lyb", region2="lya")
        self.assertEqual("BalMask set by user with bal option !=0", str(cm.exception))

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_raw(self, mock_func_1, mock_func_2):
        copy_config_substitute(self.files_path / "example_config_raw.yaml")
        test_files = THIS_DIR / "test_files" / "raw"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        with self.assertRaises(ValueError) as cm:
            deltas = bookkeeper.get_delta_extraction_tasker(region="lya")
            deltas.write_job()
            deltas.send_job()
        self.assertEqual(
            "raw continuum fitting provided in config file, use get_raw_deltas_tasker instead",
            str(cm.exception),
        )

        deltas = bookkeeper.get_raw_deltas_tasker(
            region="lya",
        )
        deltas.write_job()
        deltas.send_job()

        self.replace_paths_bookkeeper_output(bookkeeper.paths)
        if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
            self.update_test_output(test_files, bookkeeper.paths.run_path)
        self.compare_bookkeeper_output(test_files, bookkeeper.paths.run_path)

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_true_failing(self, mock_func_1, mock_func_2):
        copy_config_substitute(self.files_path / "example_config_true.yaml")
        test_files = THIS_DIR / "test_files" / "true"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        bookkeeper.config["delta extraction"]["extra args"]["picca_delta_extraction"][
            "expected flux"
        ] = dict()

        with self.assertRaises(ValueError) as cm:
            write_full_analysis(bookkeeper, calib=False, region="lyb", region2="lya")
        self.assertEqual(
            "Should define expected flux and raw statistics file in extra args section in order to run TrueContinuum",
            str(cm.exception),
        )

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_true(self, mock_func_1, mock_func_2):
        copy_config_substitute(self.files_path / "example_config_true.yaml")
        test_files = THIS_DIR / "test_files" / "true"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        write_full_analysis(bookkeeper, calib=False, region="lyb", region2="lya")

        self.replace_paths_bookkeeper_output(bookkeeper.paths)
        if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
            self.update_test_output(test_files, bookkeeper.paths.run_path)
        self.compare_bookkeeper_output(test_files, bookkeeper.paths.run_path)

    def test_bookkeeper_paths(self):
        copy_config_substitute(self.files_path / "example_config_guadalupe.yaml")
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        out = bookkeeper.paths

        out.config["data"]["release"] = "everest"
        self.assertEqual(
            str(out.healpix_data),
            "/global/cfs/cdirs/desi/spectro/redux/everest/healpix",
        )
        out.config["data"]["release"] = "fuji"
        self.assertEqual(
            str(out.healpix_data), "/global/cfs/cdirs/desi/spectro/redux/fuji/healpix"
        )
        out.config["data"]["release"] = "everrest"
        with self.assertRaises(ValueError) as cm:
            _ = out.healpix_data
        self.assertEqual("everrest", cm.exception.args[1])

        out.config["data"]["healpix data"] = "/global/healpix_data_fake/"
        with self.assertRaises(FileNotFoundError) as cm:
            _ = out.healpix_data
        self.assertEqual("/global/healpix_data_fake", str(cm.exception.args[1]))

        out.config["data"]["healpix data"] = THIS_DIR
        self.assertEqual(str(out.healpix_data), str(THIS_DIR))

        out.config["data"]["catalog"] = "/globaL/cat.fits"
        with self.assertRaises(ValueError) as cm:
            out.get_catalog_from_field("catalog")
        self.assertEqual("/globaL/cat.fits", cm.exception.args[1])

        self.assertEqual(out.get_fits_file_name("test.fits.gz"), "test")

        with self.assertRaises(ValueError) as cm:
            out.get_fits_file_name("test.fita")
        self.assertEqual("test.fita", cm.exception.args[1])
