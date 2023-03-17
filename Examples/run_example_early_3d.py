from picca_bookkeeper.bookkeeper import Bookkeeper

bookkeeper = Bookkeeper("example_config_early_3d.yaml")

calib = bookkeeper.get_calibration_extraction_tasker(
    debug=False,
)
calib.write_job()
calib.send_job()

deltas = bookkeeper.get_delta_extraction_tasker(
    region="lya",
    debug=False,
    wait_for=calib,
)
deltas.write_job()
deltas.send_job()

cf = bookkeeper.get_cf_tasker(
    region="lya",
    wait_for=deltas,
)
cf.write_job()
cf.send_job()

cf_exp = bookkeeper.get_cf_exp_tasker(
    region="lya",
    wait_for=cf,
    no_dmat=True,
)
cf_exp.write_job()
cf_exp.send_job()

xcf = bookkeeper.get_xcf_tasker(
    region="lya",
    wait_for=xcf,
)
xcf.write_job()
xcf.send_job()

xcf_exp = bookkeeper.get_xcf_exp_tasker(
    region="lya",
    wait_for=xcf,
    no_dmat=True,
)
xcf_exp.write_job()
xcf_exp.send_job()
