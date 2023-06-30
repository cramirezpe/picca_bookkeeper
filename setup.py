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
            "desi_bookkeeper_run_delta_extraction = picca_bookkeeper.scripts.run_delta_extraction:main",
            "desi_bookkeeper_run_cf = picca_bookkeeper.scripts.run_cf:main",
            "desi_bookkeeper_run_xcf = picca_bookkeeper.scripts.run_xcf:main",
            "desi_bookkeeper_run_convert_deltas = picca_bookkeeper.scripts.run_convert_deltas:main",
            "desi_bookkeeper_run_add_extra_deltas_data = picca_bookkeeper.scripts.run_add_extra_deltas_data:main",
            "desi_bookkeeper_run_fit = picca_bookkeeper.scripts.run_fit:main",
            "desi_bookkeeper_run_full_analysis = picca_bookkeeper.scripts.run_full_analysis:main",
            "desi_bookkeeper_convert_deltas = picca_bookkeeper.scripts.convert_deltas_format:main",
            "desi_bookkeeper_mix_DLA_catalogues = picca_bookkeeper.scripts.mix_DLA_catalogues:main",
            "desi_bookkeeper_add_last_night_column = picca_bookkeeper.scripts.add_last_night_column:main",
            "desi_bookkeeper_add_extra_deltas_data = picca_bookkeeper.scripts.add_extra_deltas_data:main",
            "desi_bookkeeper_generate_fit_config = picca_bookkeeper.scripts.generate_fit_config:main",
            "desi_bookkeeper_search_runs = picca_bookkeeper.scripts.search_runs:main",
            "desi_bookkeeper_show_defaults = picca_bookkeeper.scripts.print_bookkeeper_defaults:main",
            "desi_bookkeeper_show_example = picca_bookkeeper.scripts.print_example:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8.5",
)
