``picca_bookkeeper`` is a tool designed to help running analyses using the package [picca](https://github.com/igmhub/picca/). It allows for the computation of all the steps necessary for the Lyman-alpha 3D analysis using a unified configuration file, appropriately chaining all the jobs.

# Installation
The few needed requirements can be installed by using:
``` bash
pip install -r requirements.txt
```

Use: 
```bash
pip install .
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
- ``picca_bookkeeper_run_full_analysis``: Can be used to run the full analysis. It can skip some of the steps if needed.
- ``picca_bookkeeper_run_delta_extraction``: Can be used to run deltas.
- ``picca_bookkeeper_run_cf`` and ``picca_bookkeeper_run_xcf``: Can be used to run correlations.
- ``picca_bookkeeper_run_fit``: Run fit.

For more information on how to run each of them use the ``--help`` command. (the scripts can be run directly from shell, e.g. ``picca_bookkeeper_run_full_analysis --help``.).

# Examples
> **¡¡Always check the terminal log to ensure the bookkeeper does what it is expected to do!!** The bookkeeper is designed to avoid rerunning something twice, this is done by checking sent jobs to slurm and preliminarily writing job ids in output files. This is very convenient, but the user needs to verify (**especially in runs with shared parts**) that everything is done as expected. 

## Run full analysis
``` bash
picca_bookkeeper_run_full_analysis config.yaml
```

## Run full analysis if some steps where already computed (skipping them)
This will also check sent jobs, if they failed, they will be rerun automatically.
``` bash
picca_bookkeeper_run_full_analysis config.yaml --skip-sent
```

## Run two different correlations 
If one wants to run two different correlation measurements for the same set of deltas, they can use the ``delta extraction/use existing`` option for the second set of correlations. The second set of correlations has to be run with the ``--skip-sent`` option, and it is not needed to wait for the first set to finish before launching the second one.

## Run two different fits
Similarly as in the case of correlations, one can use the same deltas and correlations using the option ``delta extraction/use existing`` and ``correlations/use existing`` at the same time.

## Run on mocks
To run on mocks, use as a defulat file  ``quickquasars``, ``raw`` or ``True``. Then use one of the examples as a base and modify it. Remember to run without distortion and metals, removing them in the fits section:
```
picca_bookkeeper_run_full_analysis mock_config.yaml
```