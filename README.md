Repo to bookkeeping and run picca jobs. Currently supporting DESI.

# Installation
Use: 
```bash
pip install .
```

No specific dependencies are needed.

# File structure
The structure in the output folder will be the following:
``/global/cfs/cdirs/desi/science/lya/early-3d/{release}/{survey}/{qso_cat}/{prefix}_{calib}.{dla}.{bal}_{suxfix}``
where:
- release (everest, fuji): specifies the version of the pipeline used.
- survey (sv1, sv2, sv3, sv, all, main): specifies the survey (or combinations of surveys) used.
- qso_cat (redrock_v0, redrock_v1, afterburner_v0, etc): specifies the catalogue used (containing BAL information).
  
The last 5 elements define the continuum fitting used:
- prefix (dMdB20, pca, etc): specified the type of continuum fit (dMdB20 is the one used in eBOSS DR16, default in picca).
- calib (0, 1, 2, etc): integer identifying the version of pre-calibration using MgII (0 means no calibration).
- dla (0, 1, 2, etc): integer identifying the version of DLA masking used (0 mean no DLA masking).
- bal (0, 1, 2, etc): integer identifying the version of BAL masking used (0 mean no BAL masking).
- suffix (0, any string): any string identifying anything else particular in the analysis (0 means nothing).

The internal structure will be the following:
- correlations: ``{..}/correlations/{region}/{correlation}``, where:
  - region (lyalya_lyalya, lyalya_lyalyb, lyalya_qso, ...): region + type of correlation that is computed.
  - correlation (cf_lyalya_lyalya.fits.gz, xcf_lyalyb_qso.fits.gz, cf_exp_lyalya_lyalya.fits.gz, dmat_lyalya_lyalyb.fits.gz, ...): correlation files, including dmat and exports.
- deltas: ``{..}/deltas/{region}``, where:
  - region (lya, lyb, calibration_1, calibration_2): where the deltas is computed. 
    For each computed region we have two subfolders:
        - Delta: delta files.
        - Log: delta log files used for calibration.
- fits: ``{..}/fits/{type}``:
  - type (auto, combined, cross): Type of correlation used to make the fit. This folder will have the files:
    - chi2 for each fit with format chi2_{type}_{region}
    - config for each fit with format config_{type}_{region}
    - results for each fit with format results_{type}_{region}
- logs: shell logs for all the runs.
- scripts: slurm scripts for all the runs.