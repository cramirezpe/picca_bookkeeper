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
/early-dir/{release}/{survey}/{qso_cat}/
```

where:
- release (everest, fuji): specifies the version of the pipeline used.
- survey (sv1, sv2, sv3, sv, all, main): specifies the survey (or combinations of surveys) used.
- qso_cat (redrock_v0, redrock_v1, afterburner_v0, etc): specifies the catalogue used (can BAL information).

After the folder structure, we have one or many continuum fitting runs. The name of the folder will describe the contained fitting procedure in the following way:
```
{prefix}_{calib}.{dla}.{bal}_{suxfix}
```
- prefix (dMdB20, pca, etc): specifies the type of continuum fit (dMdB20 is the         one used in eBOSS DR16, default in Picca).
- calib (0, 1, 2, etc): integer identifying the version of pre-calibration and weight estimation
    - 0: no calibration.
    - 1: dMdB20 calibration: Two step calibration (first for flux, second for IVAR). 
      - weighting scheme includes eta, var_lss, fudge
    - 2: early3D calibration: One step calibration (flux).
      - weighting scheme uses eta and var_lss
    - 3: dMdB20 1 step calibration: One step calibration (flux).
      - weighting scheme uses eta, var_lss, fudge
    - 10: Accept custom options for calibration in yaml file.
- calib region (any of the ones defined in picca_bookkeeper.Bookkeeper.forest_regions)
    - 0 if no calibration is used.
- dla (0, 1, 2, etc): integer identifying the version of DLA masking used
    - 0: no DLA masking.
    - 1: First version of DLA catalog.
- bal (0, 1, 2, etc): integer identifying the type of BAL masking used 
    - 0: No BAL masking.
    - 1: Drop BAL objects (not implemented).
    - 2: Mask BAL objects.
- suffix (0, any string): any string identifying anything else particular in the analysis (0 means nothing). This is free for the user to select whatever they consider appropriately if it does not overlap with other runs.

Inside each continuum fitting folder, the internal structure is the following:
```
├── configs
├── correlations
│   └── default
│       ├── configs
│       ├── correlations
│       ├── fits
│       │   └── default
│       │       ├── configs
│       │       ├── logs
│       │       ├── results
│       │       └── scripts
│       ├── logs
│       └── scripts
├── deltas
├── logs
└── scripts
```
In the first level, we have the different data associated with the delta extraction step:
- configs: Configuration files used for the analysis. This includes picca ``.ini`` files and line masking files. It also saves a copy of the bookkeeper configuration file used.
- deltas: ``{..}/deltas/{region}``, where:
  - region (lya, lyb, calibration_1, calibration_2): where the deltas is computed. 
    For each computed region we have two subfolders:
        - Delta: delta files.
        - Log: delta log files used for calibration.
- logs: shell logs for all the runs.
- scripts: slurm scripts for all the runs

The second level defines the correlations run on this set of deltas. There can be multiple runs for different options, each of them labeled with a different name. In the case of the previous tree, this label is ``default``. The label can be set by the user using the ``correlations/run name`` field in the configuration file.

The internal structure of correlations is equivalent to the one for deltas. In config, a copy of the bookkeeper section used to run correlations is stored.

The third level defines the fits run on a specific correlation measurements. As before, there can be different fits for the same correlations measurement. The label identifying each fit can be set by the user by using the ``fits/run name`` field in the configuration file.

Again, the internal structure of fits is similar as the previous two. In configs, the ``.ini`` files needed to run vega are stored. Alongside a copy of the bookkeeper config file used.

# Configuration file
All the information needed to reproduce each of the run is (and should be) contained in the ``bookkeeper_config.yaml`` file. An example of a config file is stored under ``picca_bookkeeper.resources.example_config.yaml`` or can be retrieved in console by running  ``picca_bookkeeper_show_defaults`` anywhere.
# Scripts
There are multiple scripts associated with the package that are installed with the application, the most relevant are:
- ``picca_bookkeeper_run_delta_extraction``: Can be used to run deltas.
- ``picca_bookkeeper_run_cf`` and ``picca_bookkeeper_run_xcf``: Can be used to run correlations.
- ``picca_bookkeeper_run_full_analysis``: This can be used to run the full analysis.
- ``picca_bookkeeper_run_fit``: Run fit.
- ``picca_bookkeeper_generate_fit_config``: Useful script to generate a valid bookkeeper config file for fits.

For more information on how to run each of them use the ``--help`` command. (the scripts can be run directly from shell, e.g. ``picca_bookkeeper_run_full_analysis --help``.).

# Example analysis.