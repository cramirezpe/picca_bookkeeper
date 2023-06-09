import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="picca_bookkeeper",
    version="0.1",
    author="",
    author_email="user@host.com",
    description="Tool to run picca scripts on early 3D DESI measurements",
    long_description=long_description,
    long_description_content="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    package_data={
        "picca_bookkeeper": [
            "resources/*",
            "resources/*/*",
        ]
    },
    # include_package_data=True,
    entry_points={
        "console_scripts": [
            "picca_bookkeeper_run_delta_extraction = picca_bookkeeper.scripts.run_delta_extraction:main",
            "picca_bookkeeper_run_cf = picca_bookkeeper.scripts.run_cf:main",
            "picca_bookkeeper_run_xcf = picca_bookkeeper.scripts.run_xcf:main",
            "picca_bookkeeper_run_convert_deltas = picca_bookkeeper.scripts.run_convert_deltas:main",
            "picca_bookkeeper_run_add_extra_deltas_data = picca_bookkeeper.scripts.run_add_extra_deltas_data:main",
            "picca_bookkeeper_run_fit = picca_bookkeeper.scripts.run_fit:main",
            "picca_bookkeeper_run_full_analysis = picca_bookkeeper.scripts.run_full_analysis:main",
            "picca_bookkeeper_convert_deltas = picca_bookkeeper.scripts.convert_deltas_format:main",
            "picca_bookkeeper_mix_DLA_catalogues = picca_bookkeeper.scripts.mix_DLA_catalogues:main",
            "picca_bookkeeper_add_last_nfight_column = picca_bookkeeper.scripts.add_last_night_column:main",
            "picca_bookkeeper_add_extra_deltas_data = picca_bookkeeper.scripts.add_extra_deltas_data:main",
            "picca_bookkeeper_search_runs = picca_bookkeeper.scripts.search_runs:main",
            "picca_bookkeeper_show_defaults = picca_bookkeeper.scripts.print_bookkeeper_defaults:main",
            "picca_bookkeeper_show_example = picca_bookkeeper.scripts.print_example:main",
            "picca_bookkeeper_cancel_jobids = picca_bookkeeper.scripts.cancel_jobids_from_file:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8.5",
)
