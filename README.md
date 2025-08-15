``picca_bookkeeper`` is a tool designed to help running analyses using the package [picca](https://github.com/igmhub/picca/). It allows for the computation of all the steps necessary for the Lyman-alpha 3D analysis using a unified configuration file, appropriately chaining all the jobs.

# Installation
Create a new conda environment, ensuring up to date dependencies for picca, vega, and the bookkeeper: 
```bash
conda create -n env_name numpy=1.24.4 scipy=1.15.3 matplotlib=3.10.3
```

Activate new environment, and install a stable version of Picca: 
```bash
conda activate evn_name

pip install picca --version=9.13.0
```

Install Vega:
```bash
git clone https://github.com/andreicuceu/vega.git Vega

cd vega

pip install -e
```

If using the Vega sampler, install MPI: 
```bash
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
```

To get the most up-to-date version of the bookkeper, clone the repo:
```bash
git clone https://github.com/cramirezpe/picca_bookkeeper.git picca_bookkeeper

cd picca_bookkeeper

pip install .
```

This package is also being published to PyPI, and therefore can be installed through:
```bash
pip install picca_bookkeeper
```

# File structure
``picca_bookkeeper`` relies in a particular file structure to work as intended. The parent folder is chosen by the user using the ``data/bookkeeper dir`` option in the config file.

Inside this folder, the internal structure is the following:
```
├── configs # Location of bookkeeper configuration file and defaults files
│   ├── bookkeeper_config.yaml 
│   └── defaults.yaml
├── correlations # Location of the different correlations
│   ├── configs # Configuration used for the correlation (usually empty)
│   ├── logs # Log output from slurm runs
│   ├── results # Correlations
│   └── scripts # Scripts to run the correlations
├── deltas
│   ├── configs
│   ├── logs
│   ├── results
│   └── scripts
└── fits
    ├── configs
    ├── logs
    ├── results
    └── scripts
```


# Configuration file
All the information needed to reproduce each of the run is (and should be) contained in the ``bookkeeper_config.yaml`` file. An example of a config file is stored under ``picca_bookkeeper.resources.example_config.yaml`` or can be retrieved in console by running  ``picca_bookkeeper_show_example`` anywhere.
# Scripts
There are multiple scripts associated with the package that are installed with the application, the most relevant are:
- ``picca_bookkeeper_run_full_analysis``: Can be used to run the full analysis. It can skip some of the steps (e.g. Deltas) if needed.
- ``picca_bookkeeper_run_delta_extraction``: Can be used to run deltas.
- ``picca_bookkeeper_run_cf`` and ``picca_bookkeeper_run_xcf``: Can be used to run correlations.
- ``picca_bookkeeper_run_fit``: Run fit.

For more information on how to run each of them use the ``--help`` command. (the scripts can be run directly from shell, e.g. ``picca_bookkeeper_run_full_analysis --help``.).

# Examples
> **¡¡Always check the terminal log to ensure the bookkeeper does what it is expected to do!!** The bookkeeper is designed to avoid rerunning something twice, this is done by checking sent jobs to slurm and preliminarily writing job ids in output files. This is very convenient, but the user needs to verify (**especially in runs with shared parts**) that everything is done as expected. 

## Run full analysis
``` bash
picca_bookkeeper_run_full_analysis /path_to_config.yaml
```

## Run full analysis if some steps where already computed (skipping them)
This will also check sent jobs, if they failed, they will be rerun automatically.
``` bash
picca_bookkeeper_run_full_analysis /path_to_config.yaml --skip-sent
```

## Write full analysis config files only (do not schedule to run)
Useful for just writing the config files, and checking the contents without scheduling jobs.
``` bash
picca_bookkeeper_run_full_analysis /path_to_config.yaml --only-write
```

## Run two different correlations 
If one wants to run two different correlation measurements for the same set of deltas, they can use the ``delta extraction/use existing`` option for the second set of correlations. The second set of correlations has to be run with the ``--skip-sent`` option, and it is not needed to wait for the first set to finish before launching the second one.

## Run two different fits
Similarly as in the case of correlations, one can use the same deltas and correlations using the option ``delta extraction/use existing`` and ``correlations/use existing`` at the same time.

## Run on mocks
To run on mocks, use as a defulat file  ``quickquasars``, ``raw`` or ``True``. Then use one of the examples as a base and modify it. Remember to run without distortion and metals, removing them in the fits section:
```
picca_bookkeeper_run_full_analysis \path_to_mock_config.yaml
```

## Run the Vega Sampler
To run the sampler, ensure that MPI is properly installed with NERSC specific settings in your conda environment: 
```bash
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
```
Note: for the bookkeeper config file, you will need to specify your conda environment again. (Otherwise the bookkeeper will default to sourcing a separate Vega installation.)
```bash
general:
  conda environment: /path/to/conda_environment

...

fits:
  sampler environment: /path/to/conda_environment
```
Then to run from the terminal (including arguments like ``--skip-sent`` if skipping deltas, etc.): 
```bash
picca_bookkeeper_run_sampler /path_to_config.yaml
```
