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
``picca_bookkeeper`` relies in a particular file structure to work as intended. The parent folder structure where the bookkeeper is located depends on the input data to be used by the run:

```
{bookkeeper dir}/{release}/{survey}/{qso_cat}/
```

where:
- release (everest, fuji): specifies the version of the pipeline used.
- survey (sv1, sv2, sv3, sv, all, main): specifies the survey (or combinations of surveys) used.
- qso_cat (redrock_v0, redrock_v1, afterburner_v0, etc): specifies the catalogue used (can BAL information).

After the folder structure, we have one or many continuum fitting runs. The name of the folder will describe the contained fitting procedure in the following way:
```
{prefix}_{calib}.{calib_regions}.{dla}.{bal}_{suxfix}
```
- prefix (dMdB20, CRP23, raw, True): specifies the general configuration of the run.
    - dMdB20: fudge included, 2.4A bins, var lss mod=0 (see resources/dMdB20.yaml)
    - CRP23: analysis performed for early3D DESI analyses. nofudge, 0.8A bins, var lss mod = 7.5, lambda rest frame max=1205.
    - raw: use this for raw mocks.
    - true: use this for true continuum analyses.
    - Could also match any of the yaml files under picca_bookkeeper.resources.default_configs

- calib (0, 1, 2): Integer defining the number of calibration steps.
    - 0: no calibration.
    - 1: early3D calibration: One step calibration (flux).
    - 2: dMdB20 calibration: Two step calibration (first for flux, second for IVAR).
    - 10: Accept custom options for calibration in yaml file.

- calib region (any of the ones defined in picca_bookkeeper.Bookkeeper.forest_regions)
    - 0 if no calibration is used.

- dla (0, 1, 2, etc): integer identifying the version of DLA masking used
    - 0: no DLA masking.
    - 1: First version of DLA catalog.
    - N: Nth version of DLA catalog.

- bal (0, 1, 2, etc): integer identifying the version of BAL masking used 
    - 0: No BAL masking.
    - 1: Drop BAL objects (not implemented).
    - 2: Mask BAL objects.

- suffix (0, any string): any string identifying anything else particular
        in the analysis (0 means nothing). This is free for the user to select
        whatever they consider appropiately if it does not overlap with other 
        runs.

Inside each continuum fitting folder, the internal structure is the following:
```
├── configs
├── correlations
│   └── default
│       ├── configs
│       ├── results
│       ├── fits
│       │   └── default
│       │       ├── configs
│       │       ├── logs
│       │       ├── results
│       │       └── scripts
│       ├── logs
│       └── scripts
├── results
├── logs
└── scripts
```
In the first level, we have the different data associated with the delta extraction step:
- configs: Configuration files used for the analysis. This includes picca ``.ini`` files and line masking files. It also saves a copy of the bookkeeper configuration file used.
- results: ``{..}/results/{region}``, where:
  - region (lya, lyb, calibration_1, calibration_2): where the deltas is computed. 
    For each computed region we have two subfolders:
        - Delta: delta files.
        - Log: delta log files used for calibration.
- logs: shell logs for all the runs.e
- scripts: slurm scripts for all the runs

The second level defines the correlations run on this set of deltas. There can be multiple runs for different options, each of them labeled with a different name. In the case of the previous tree, this label is ``default``. The label can be set by the user using the ``correlations/run name`` field in the configuration file.

The internal structure of correlations is equivalent to the one for deltas. In config, a copy of the bookkeeper section used to run correlations is stored.

The third level defines the fits run on a specific correlation measurements. As before, there can be different fits for the same correlations measurement. The label identifying each fit can be set by the user by using the ``fits/run name`` field in the configuration file.

Again, the internal structure of fits is similar as the previous two. In configs, the ``.ini`` files needed to run vega are stored. Alongside a copy of the bookkeeper config file used.

# Configuration file
All the information needed to reproduce each of the run is (and should be) contained in the ``bookkeeper_config.yaml`` file. An example of a config file is stored under ``picca_bookkeeper.resources.example_config.yaml`` or can be retrieved in console by running  ``picca_bookkeeper_show_example`` anywhere.
# Scripts
There are multiple scripts associated with the package that are installed with the application, the most relevant are:
- ``picca_bookkeeper_run_delta_extraction``: Can be used to run deltas.
- ``picca_bookkeeper_run_cf`` and ``picca_bookkeeper_run_xcf``: Can be used to run correlations.
- ``picca_bookkeeper_run_full_analysis``: Can be used to run the full analysis.
- ``picca_bookkeeper_run_fit``: Run fit.

For more information on how to run each of them use the ``--help`` command. (the scripts can be run directly from shell, e.g. ``picca_bookkeeper_run_full_analysis --help``.).

# Examples
## Run full analysis
``` bash
picca_bookkeeper_run_full_analysis config.yaml --auto-correlations lya.lya-lya.lya lya.lya-lya.lyb --cross-correlations lya.lya lya.lyb 
```

## Run full analysis if deltas were already computed (skipping them)
``` bash
picca_bookkeeper_run_full_analysis config.yaml --auto-correlations lya.lya-lya.lya lya.lya-lya.lyb --cross-correlations lya.lya lya.lyb --no-deltas
```

## Run two different correlations 
If one wants to run two different correlation measurements for the same set of deltas, they will need to generate two config files ``config1.yaml``  ``config2.yaml``, config2 ``run name`` inside ``correlations`` section should be different than the one in config1:
```bash
picca_bookkeeper_run_full_analysis config1.yaml --auto-correlations lya.lya-lya.lya lya.lya-lya.lyb --cross-correlations lya.lya lya.lyb --no-fits
```
We can collect the jobid from the deltas steps (number returned in terminal), and use it for the next correlation measurements to wait for them:
```
# We don't need to rerun deltas
picca_bookkeeper_run_full_analysis config2.yaml --auto-correlations lya.lya-lya.lya lya.lya-lya.lyb --cross-correlations lya.lya lya.lyb --no-deltas --no-fits --wait-for {delta-lya-jobid} {delta-lyb-jobid}
```

## Run full analysis with modified fits
The same idea as for the correlations can be applied to generate two different fits. First we generate the full analysis:
``` bash
picca_bookkeeper_run_full_analysis config1.yaml --auto-correlations lya.lya-lya.lya lya.lya-lya.lyb --cross-correlations lya.lya lya.lyb 
```
and then we generate only the second fits
```bash
picca_bookkeeper_run_full_analysis --auto-correlations lya.lya-lya.lya lya.lya-lya.lyb --cross-correlations lya.lya lya.lyb --no-deltas --no-correlations --waitfor {auto-export-jobid} {auto-metal-jobid} {cross-export-jobid} {cross-metal-jobid}
```