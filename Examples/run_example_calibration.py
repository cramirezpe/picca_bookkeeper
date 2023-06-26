from numpy import nonzero
from picca_bookkeeper.bookkeeper import Bookkeeper

# Create bookkeeper instance
bookkeeper = Bookkeeper("example_config_early_3d.yaml")

# Run calibration
calib = bookkeeper.get_calibration_extraction_tasker(
    debug=True,
    wait_for=None,
)

calib.write_job()
calib.send_job()

# Run deltas
deltas = bookkeeper.get_delta_extraction_tasker(
    region="lya",  # Provide the region to use here
    debug=True,  # Use less specs and run in debug queue,
    wait_for=calib,  # Wait for previous steps
)

deltas.write_job()
deltas.send_job()

cf = bookkeeper.get_cf_tasker(
    region="lya",
    region2=None,  # None if same as region
    debug=False,
    wait_for=deltas,
)

dmat = bookkeeper.get_dmat_tasker(
    region="lya",
    region2=None,
    debug=False,
    wait_for=deltas,
)

metal = bookkeeper.get_metal_tasker(
    region="lya", region2=None, debug=False, wait_for=deltas
)

xcf = bookkeeper.get_xcf_tasker(
    region="lya",
    debug=False,
    wait_for=deltas,
)

xdmat = bookkeeper.get_xdmat_tasker(
    region="lya",
    debug=False,
    wait_for=deltas,
)

xmetal = bookkeeper.get_xmetal_tasker(
    region="lya",
    debug=False,
    wait_for=deltas,
)

for task in (cf, dmat, metal, xcf, xdmat, xmetal):
    task.write_job()
    task.send_job()

cf_exp = bookkeeper.get_cf_exp_tasker(
    region="lya",
    region2=None,
    wait_for=[cf, dmat],
    no_dmat=False,  # Don't use distortion matrix (set to true if dmat was not computed)
)

xcf_exp = bookkeeper.get_xcf_exp_tasker(
    region="lya",
    wait_for=[xcf, xdmat],
    no_dmat=False,
)

for task in (cf_exp, xcf_exp):
    task.write_job()
    task.send_job()

# Fits (under development)
# auto_fit = bookkeeper.get_cf_fit_tasker(
#     bao_mode="fixed",
#     region="lya",
#     region2=None,
#     wait_for=[cf_exp],
# )

# cross_fit = bookkeeper.get_xcf_fit_tasker(
#     region="lya", bao_mode="fixed", wait_for=[xcf_exp]
# )

# for task in (auto_fit, cross_fit):
#     task.write_job()
#     task.send_job()
