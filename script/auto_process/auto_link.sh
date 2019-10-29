# Linking parameters
link_frame_start=0
link_frame_end=75
link_linker="trackpy"
link_range=40
link_dx_max=25
link_dt_max=10
link_blur=1
link_threshold=3

if [ ! -d "link_3d" ]; then
    mkdir link_3d
fi

python3 script/link.py track_3d/locations_3d.pkl $link_linker\
    $link_frame_start $link_frame_end $link_range\
    $link_dx_max $link_dt_max $link_blur $link_threshold\
    link_3d
