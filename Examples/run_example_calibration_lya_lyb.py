from sympy import N
from picca_bookkeeper.bookkeeper import Bookkeeper

# Create bookkeeper instance
bookkeeper = Bookkeeper("example_config_guadalupe.yaml")

calib = bookkeeper.get_calibration_extraction_tasker(
    debug=True,
    wait_for=None,
)

calib.write_job()
calib.send_job()

# Run deltas
deltas_a = bookkeeper.get_delta_extraction_tasker(
    region="lya",  # Provide the region to use here
    debug=True,  # Use less specs and run in debug queue,
    wait_for=calib,  # Wait for previous steps
)

deltas_a.write_job()
deltas_a.send_job()

deltas_b = bookkeeper.get_delta_extraction_tasker(
    region="lyb",  # Provide the region to use here
    debug=True,  # Use less specs and run in debug queue,
    wait_for=calib,  # Wait for previous steps
)

deltas_b.write_job()
deltas_b.send_job()

cf = bookkeeper.get_cf_tasker(
    region="lya",
    region2="lyb",
    debug=False,
    wait_for=[deltas_a, deltas_b],
)

dmat = bookkeeper.get_dmat_tasker(
    region="lya", region2="lyb", debug=False, wait_for=[deltas_a, deltas_b]
)

metal = bookkeeper.get_metal_tasker(
    region="lya", region2=None, debug=False, wait_for=[deltas_a, deltas_b]
)

xcf = bookkeeper.get_xcf_tasker(
    region="lyb",
    debug=False,
    wait_for=deltas_b,
)

xdmat = bookkeeper.get_xdmat_tasker(
    region="lyb",
    debug=False,
    wait_for=deltas_b,
)

xmetal = bookkeeper.get_xmetal_tasker(
    region="lya",
    debug=False,
    wait_for=deltas_b,
)

for task in (cf, dmat, metal, xcf, xdmat, xmetal):
    task.write_job()
    task.send_job()


cf_exp = bookkeeper.get_cf_exp_tasker(
    region="lya",
    region2="lyb",
    wait_for=[cf, dmat],
    no_dmat=False,  # Don't use distortion matrix (set to true if dmat was not computed)
)

xcf_exp = bookkeeper.get_xcf_exp_tasker(
    region="lyb",
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
