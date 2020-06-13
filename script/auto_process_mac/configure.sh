# Specifying files
script_folder="script"
video_folder="../fish_1"
cam_1_internal="database/cam_1_int.pkl"
cam_2_internal="database/cam_2_int.pkl"
cam_3_internal="database/cam_3_int.pkl"

# Calibration Spec
calib_folder="../calib-ext"
calib_format="tiff"
order_json="../calib-ext/calib-orders.json"
grid_size=16   # size of square on the calibration board, unit: mm
corner_number="23, 15"  # number of inner corners in the row, column

# video processing parameters
background_rolling_length=600
blur=4
local=7
binary_open_size=7

# 2D Tracking parameters
measure_roi=0
cam_1_kernels="database/shape_kernels.npy"
cam_2_kernels="database/shape_kernels.npy"
cam_3_kernels="database/shape_kernels.npy"
track_2d_frame_start=0
track_2d_frame_end=75
track_2d_size_min=5
track_2d_size_max=25
track_2d_orientation_number=36
track_2d_intensity_threshold=0.6
track_2d_want_plot="True"

# 3D Tracking parameters
track_3d_frame_start=0
track_3d_frame_end=75
track_3d_sample_size=10
track_3d_tol_2d=3  # tolarance of reprojection error
track_3d_water_depth=400
track_3d_water_level=0
track_3d_want_plot=0

# GReTA Tracking parameters
greta_frame_start=0
greta_frame_end=500
greta_water_depth=350
greta_search_range=40
greta_tol_2d=5
greta_tau_1=5
greta_tau_2=4
greta_overlap_num=5
greta_overlap_rtol=20
greta_relink_dx=50
greta_relink_dx_step=5
greta_relink_dt=40
greta_relink_blur=2
