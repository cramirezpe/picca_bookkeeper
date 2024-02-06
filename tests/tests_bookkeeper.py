from pathlib import Path
import argparse
import unittest
import shutil
from picca_bookkeeper.bookkeeper import Bookkeeper
from picca_bookkeeper.scripts.run_full_analysis import main as run_full
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
    jobid_seed = Path(command.split(' ')[-1]).name

    class Out:
        stdout = codecs.encode(
            str(int.from_bytes(jobid_seed.encode(), "little"))[:8]
        )
        
        returncode = 0

    return Out()


def write_full_analysis(config_path, args=dict()):
    args = argparse.Namespace(**{
        **dict(
            bookkeeper_config = config_path,
            overwrite = False,
            overwrite_config = False,
            skip_sent = False,
            auto_correlations = [],
            cross_correlations = [],
            system = None,
            no_deltas=False,
            no_correlations = False,
            no_fits = False,
            sampler = False,
            debug= False,
            only_write= False,
            wait_for = None,
            log_level = "CRITICAL",
        ),
        **args,
    })
    
    run_full(args)


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

        self.patcher = patch('picca_bookkeeper.bookkeeper.PathBuilder.catalog_exists')
        self.patcher.return_value = True
        self.mock = self.patcher.start()
        self.addCleanup(self.patcher.stop)

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
            bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "fake_cfg.yaml", read_mode=False)

        copy_config_substitute(self.files_path / "example_config_guadalupe.yaml")
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        with self.assertRaises(ValueError):
            bookkeeper.validate_region("lyc")

    # @patch("builtins.input", return_value="yes")
    # def test_config_overwrite_on_yes(self, mock_print):
    #     copy_config_substitute(self.files_path / "example_config_guadalupe.yaml")
    #     bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

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
    #     bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

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
    #         bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

    #     def side_effect(*args, **kwargs):
    #         return THIS_DIR / "test_files" / "alt_cat" / "dummy_catalog.fits"

    #     with patch(
    #         "picca_bookkeeper.bookkeeper.get_quasar_catalog", side_effect=side_effect
    #     ), self.assertRaises(FileExistsError):
    #         bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_example_guadalupe(self, mock_func_1, mock_func_2):
        copy_config_substitute(self.files_path / "example_config_guadalupe.yaml")
        test_files = THIS_DIR / "test_files" / "guadalupe"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml")

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
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        self.replace_paths_bookkeeper_output(bookkeeper.paths)
        if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
            self.update_test_output(test_files, bookkeeper.paths.run_path)
        self.compare_bookkeeper_output(test_files, bookkeeper.paths.run_path)

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_example_kp6(self, mock_func_1, mock_func_2):
        copy_config_substitute(
            self.files_path / "example_config_kp6.yaml"
        )
        test_files = THIS_DIR / "test_files" / "kp6"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        self.replace_paths_bookkeeper_output(bookkeeper.paths)
        if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
            self.update_test_output(test_files, bookkeeper.paths.run_path)
        self.compare_bookkeeper_output(test_files, bookkeeper.paths.run_path)

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_example_guadalupe_perlmutter_compute_zeff(self, mock_func_1, mock_func_2):
        copy_config_substitute(
            self.files_path / "example_config_guadalupe_perlmutter_compute_zeff.yaml"
        )
        test_files = THIS_DIR / "test_files" / "guadalupe_perlmutter_compute_zeff"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        self.replace_paths_bookkeeper_output(bookkeeper.paths)
        if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
            self.update_test_output(test_files, bookkeeper.paths.run_path)
        self.compare_bookkeeper_output(test_files, bookkeeper.paths.run_path)

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_example_guadalupe_sampler(self, mock_func_1, mock_func_2):
        copy_config_substitute(
            self.files_path / "example_config_guadalupe_sampler.yaml"
        )
        test_files = THIS_DIR / "test_files" / "guadalupe_sampler"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml", args=dict(sampler=True))

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
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

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
    #     bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

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
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml")

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
        copy_config_substitute(self.files_path / "example_config_guadalupe_lyb.yaml")
        test_files = THIS_DIR / "test_files" / "calib"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml")

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
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml")
        bookkeeper.paths.deltas_log_path(None, calib_step=1).mkdir(
            exist_ok=True, parents=True
        )
        bookkeeper.paths.deltas_log_path(None, calib_step=2).mkdir(
            exist_ok=True, parents=True
        )
        bookkeeper.paths.delta_attributes_file(None, calib_step=1).touch()
        bookkeeper.paths.delta_attributes_file(None, calib_step=2).touch()
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

        bookkeeper2 = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        self.replace_paths_bookkeeper_output(bookkeeper2.paths)
        if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
            self.update_test_output(test_files, bookkeeper2.paths.run_path)
        self.compare_bookkeeper_output(test_files, bookkeeper2.paths.run_path)

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_example_copy_files(self, mock_func_1, mock_func_2):
        copy_config_substitute(self.files_path / "example_config_guadalupe_lyb.yaml")
        test_files = THIS_DIR / "test_files" / "copy_correlation_files"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml")
        bookkeeper.paths.deltas_log_path(None, calib_step=1).mkdir(
            exist_ok=True, parents=True
        )
        
        bookkeeper.paths.deltas_path("lyb").mkdir(exist_ok=True, parents=True)
        bookkeeper.paths.delta_attributes_file("lyb").parent.mkdir(exist_ok=True, parents=True)
        (bookkeeper.paths.deltas_path("lyb") / "Delta-1.fits.gz").write_text("1231")
        bookkeeper.paths.delta_attributes_file("lyb").write_text("3452")

        absorber = "lya"
        region = "lya"
        absorber2 = "lya"
        region2 = "lyb"
        bookkeeper.paths.cf_fname(absorber, region, absorber2, region2).parent.mkdir(
            exist_ok=True, parents=True
        )
        bookkeeper.paths.cf_fname(absorber, region, absorber2, region2).write_text(
            "10"
        )
        bookkeeper.paths.dmat_fname(absorber, region, absorber2, region2).write_text(
            "20"
        )
        bookkeeper.paths.metal_fname(absorber, region, absorber2, region2).write_text(
            "30"
        )

        bookkeeper.paths.xcf_fname(absorber, region).parent.mkdir(
            exist_ok=True, parents=True
        )
        bookkeeper.paths.xcf_fname(absorber, region2).write_text("40")
        bookkeeper.paths.xdmat_fname(absorber, region2).write_text("50")
        bookkeeper.paths.xmetal_fname(absorber, region2).write_text("60")

        copy_config_substitute(
            self.files_path / "example_config_guadalupe_calib_copy_corrs.yaml"
        )

        with open(THIS_DIR / "test_files" / "output" / "tmp.yaml", "r") as file:
            filedata = file.read()

        filedata = filedata.replace(
            "deltaslyb:",
            f"lyb: {str(bookkeeper.paths.deltas_path('lyb'))} {str(bookkeeper.paths.delta_attributes_file('lyb'))}"
        )

        filedata = filedata.replace(
            "cflyalya_lyalyb:",
            f"lyalya_lyalyb: {str(bookkeeper.paths.cf_fname(absorber, region, absorber2, region2))}",
        )
        filedata = filedata.replace(
            "xcflyalyb:", f"lyalyb: {str(bookkeeper.paths.xcf_fname(absorber, region2))}"
        )
        filedata = filedata.replace(
            "dmatlyalya_lyalyb:",
            f"lyalya_lyalyb: {str(bookkeeper.paths.dmat_fname(absorber, region, absorber2, region2))}",
        )
        filedata = filedata.replace(
            "xdmatlyalyb:",
            f"lyalyb: {str(bookkeeper.paths.xdmat_fname(absorber, region2))}",
        )
        filedata = filedata.replace(
            "metallyalya_lyalyb:",
            f"lyalya_lyalyb: {str(bookkeeper.paths.metal_fname(absorber, region, absorber2, region2))}",
        )
        filedata = filedata.replace(
            "xmetallyalyb:",
            f"lyalyb: {str(bookkeeper.paths.xmetal_fname(absorber, region2))}",
        )

        with open(THIS_DIR / "test_files" / "output" / "tmp.yaml", "w") as file:
            file.write(filedata)

        bookkeeper2 = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        assert(
            (bookkeeper2.paths.deltas_path('lyb') / "Delta-1.fits.gz").read_text() == "1231"
        )

        assert(
            bookkeeper2.paths.delta_attributes_file("lyb").read_text() == "3452"
        )

        assert (
            bookkeeper2.paths.cf_fname(absorber, region, absorber2, region2).read_text()
            == "10"
        )
        assert (
            bookkeeper2.paths.dmat_fname(
                absorber, region, absorber2, region2
            ).read_text()
            == "20"
        )
        assert (
            bookkeeper2.paths.metal_fname(
                absorber, region, absorber2, region2
            ).read_text()
            == "30"
        )

        assert bookkeeper2.paths.xcf_fname(absorber, region2).read_text() == "40"
        assert bookkeeper2.paths.xdmat_fname(absorber, region2).read_text() == "50"
        assert bookkeeper2.paths.xmetal_fname(absorber, region2).read_text() == "60"

        self.replace_paths_bookkeeper_output(bookkeeper2.paths)
        if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
            self.update_test_output(test_files, bookkeeper2.paths.run_path)
        self.compare_bookkeeper_output(test_files, bookkeeper2.paths.run_path)

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_example_copy_full_files(self, mock_func_1, mock_func_2):
        copy_config_substitute(self.files_path / "example_config_guadalupe_lyb.yaml")
        test_files = THIS_DIR / "test_files" / "copy_correlation_files_full"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml")
        bookkeeper.paths.deltas_log_path(None, calib_step=1).mkdir(
            exist_ok=True, parents=True
        )
        
        bookkeeper.paths.deltas_path("lyb").mkdir(exist_ok=True, parents=True)
        bookkeeper.paths.delta_attributes_file("lyb").parent.mkdir(exist_ok=True, parents=True)
        (bookkeeper.paths.deltas_path("lyb") / "Delta-1.fits.gz").write_text("1231")
        bookkeeper.paths.delta_attributes_file("lyb").write_text("3452")

        absorber = "lya"
        region = "lya"
        absorber2 = "lya"
        region2 = "lyb"
        bookkeeper.paths.cf_fname(absorber, region, absorber2, region2).parent.mkdir(
            exist_ok=True, parents=True
        )
        bookkeeper.paths.cf_fname(absorber, region, absorber2, region2).write_text(
            "10"
        )
        bookkeeper.paths.dmat_fname(absorber, region, absorber2, region2).write_text(
            "20"
        )
        bookkeeper.paths.metal_fname(absorber, region, absorber2, region2).write_text(
            "30"
        )

        bookkeeper.paths.xcf_fname(absorber, region).parent.mkdir(
            exist_ok=True, parents=True
        )
        bookkeeper.paths.xcf_fname(absorber, region2).write_text("40")
        bookkeeper.paths.xdmat_fname(absorber, region2).write_text("50")
        bookkeeper.paths.xmetal_fname(absorber, region2).write_text("60")

        copy_config_substitute(
            self.files_path / "example_config_guadalupe_calib_copy_corrs_full.yaml"
        )

        with open(THIS_DIR / "test_files" / "output" / "tmp.yaml", "r") as file:
            filedata = file.read()

        filedata = filedata.replace(
            "link deltas:",
            f"link deltas: {str(bookkeeper.paths.run_path / 'results')}"
        )

        filedata = filedata.replace(
            "link correlations:",
            f"link correlations: {str(bookkeeper.paths.correlations_path / 'results')}"
        )

        filedata = filedata.replace(
            "link distortion matrices:",
            f"link distortion matrices: {str(bookkeeper.paths.correlations_path / 'results')}"
        )
        filedata = filedata.replace(
            "link metals:",
            f"link metals: {str(bookkeeper.paths.correlations_path / 'results')}"
        )


        with open(THIS_DIR / "test_files" / "output" / "tmp.yaml", "w") as file:
            file.write(filedata)

        bookkeeper2 = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        assert(
            (bookkeeper2.paths.deltas_path('lyb') / "Delta-1.fits.gz").read_text() == "1231"
        )

        assert(
            bookkeeper2.paths.delta_attributes_file("lyb").read_text() == "3452"
        )

        assert (
            bookkeeper2.paths.cf_fname(absorber, region, absorber2, region2).read_text()
            == "10"
        )
        assert (
            bookkeeper2.paths.dmat_fname(
                absorber, region, absorber2, region2
            ).read_text()
            == "20"
        )
        assert (
            bookkeeper2.paths.metal_fname(
                absorber, region, absorber2, region2
            ).read_text()
            == "30"
        )

        assert bookkeeper2.paths.xcf_fname(absorber, region2).read_text() == "40"
        assert bookkeeper2.paths.xdmat_fname(absorber, region2).read_text() == "50"
        assert bookkeeper2.paths.xmetal_fname(absorber, region2).read_text() == "60"

        self.replace_paths_bookkeeper_output(bookkeeper2.paths)
        if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
            self.update_test_output(test_files, bookkeeper2.paths.run_path)
        self.compare_bookkeeper_output(test_files, bookkeeper2.paths.run_path)

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_run_only_correlation_on_another_Bookkeeper(self, mock_func_1, mock_func_2, read_mode=False):
        copy_config_substitute(self.files_path / "example_config_guadalupe_lyb.yaml")
        test_files = THIS_DIR / "test_files" / "second_correlation"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        # Now failing run
        copy_config_substitute(
            self.files_path / "example_config_guadalupe_only_corr_fail.yaml"
        )

        with open(THIS_DIR / "test_files" / "output" / "tmp.yaml", "r") as file:
            filedata = file.read()

        with self.assertRaises(ValueError) as cm:
            bookkeeper2 = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)
        self.assertEqual(
            "Incompatible configs:",
            str(cm.exception)[0:21],
        )

        # Now failing run (2)
        copy_config_substitute(
            self.files_path / "example_config_guadalupe_only_corr_fail_2.yaml"
        )

        with open(THIS_DIR / "test_files" / "output" / "tmp.yaml", "r") as file:
            filedata = file.read()

        with self.assertRaises(ValueError) as cm:
            bookkeeper2 = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)
        self.assertEqual(
            "Incompatible configs:",
            str(cm.exception)[0:21],
        )

        # Now main run:
        copy_config_substitute(
            self.files_path / "example_config_guadalupe_only_corr.yaml"
        )

        with open(THIS_DIR / "test_files" / "output" / "tmp.yaml", "r") as file:
            filedata = file.read()

        bookkeeper2 = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        cf = bookkeeper2.get_cf_tasker(
            region="lyb",
            region2="lya",
            wait_for=None,
        )

        cf.write_job()
        cf.send_job()
        
        self.replace_paths_bookkeeper_output(bookkeeper.paths)
        self.replace_paths_bookkeeper_output(bookkeeper2.paths)
        if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
            self.update_test_output(test_files, bookkeeper2.paths.run_path)
        self.compare_bookkeeper_output(test_files, bookkeeper2.paths.run_path)

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_run_only_fit_on_another_Bookkeeper(self, mock_func_1, mock_func_2, read_mode=False):
        copy_config_substitute(self.files_path / "example_config_guadalupe.yaml")
        test_files = THIS_DIR / "test_files" / "second_fit"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        self.replace_paths_bookkeeper_output(bookkeeper.paths)
        if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
            self.update_test_output(test_files, bookkeeper.paths.run_path)
        self.compare_bookkeeper_output(test_files, bookkeeper.paths.run_path)

        # Now failing run
        copy_config_substitute(
            self.files_path / "example_config_guadalupe_only_fit_fail.yaml"
        )

        with open(THIS_DIR / "test_files" / "output" / "tmp.yaml", "r") as file:
            filedata = file.read()

        with self.assertRaises(ValueError) as cm:
            bookkeeper2 = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)
        self.assertEqual(
            "Incompatible configs:",
            str(cm.exception)[0:21],
        )

        # Now failing run (2)
        copy_config_substitute(
            self.files_path / "example_config_guadalupe_only_fit_fail_2.yaml"
        )

        with open(THIS_DIR / "test_files" / "output" / "tmp.yaml", "r") as file:
            filedata = file.read()

        with self.assertRaises(ValueError) as cm:
            bookkeeper2 = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)
        self.assertEqual(
            "Incompatible configs:",
            str(cm.exception)[0:21],
        )

        # Now main run:
        copy_config_substitute(
            self.files_path / "example_config_guadalupe_only_fit.yaml"
        )

        with open(THIS_DIR / "test_files" / "output" / "tmp.yaml", "r") as file:
            filedata = file.read()

        bookkeeper2 = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        fit = bookkeeper2.get_fit_tasker()

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
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml")

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
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        with self.assertRaises(ValueError) as cm:
            calib = bookkeeper.get_calibration_extraction_tasker()
            calib.write_job()
            calib.send_job()

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
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

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

        write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml")

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
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        with self.assertRaises(ValueError) as cm:
            write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml")
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
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml")

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
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        with self.assertRaises(ValueError) as cm:
            write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml")
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
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        with self.assertRaises(ValueError) as cm:
            write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml")
        self.assertEqual("BalMask set by user with bal option !=0", str(cm.exception))

    @patch("picca_bookkeeper.tasker.run", side_effect=mock_run)
    @patch(
        "picca_bookkeeper.bookkeeper.get_quasar_catalog",
        side_effect=mock_get_3d_catalog,
    )
    def test_raw(self, mock_func_1, mock_func_2):
        copy_config_substitute(self.files_path / "example_config_raw.yaml")
        test_files = THIS_DIR / "test_files" / "raw"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

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
    def test_true(self, mock_func_1, mock_func_2):
        copy_config_substitute(self.files_path / "example_config_true.yaml")
        test_files = THIS_DIR / "test_files" / "true"
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

        write_full_analysis(THIS_DIR / "test_files" / "output" / "tmp.yaml")

        self.replace_paths_bookkeeper_output(bookkeeper.paths)
        if "UPDATE_TESTS" in os.environ and os.environ["UPDATE_TESTS"] == "True":
            self.update_test_output(test_files, bookkeeper.paths.run_path)
        self.compare_bookkeeper_output(test_files, bookkeeper.paths.run_path)

    def test_bookkeeper_paths(self):
        copy_config_substitute(self.files_path / "example_config_guadalupe.yaml")
        bookkeeper = Bookkeeper(THIS_DIR / "test_files" / "output" / "tmp.yaml", read_mode=False)

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
