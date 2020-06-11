# retrieving parameters
source configure.sh

# create folders for 3D tracking
if [ ! -d "track_3d" ]; then
    mkdir track_3d
fi
if [ ! -d "track_3d/locations_3d" ]; then
    mkdir track_3d/locations_3d
fi

cp $script_folder/track_3d/* track_3d

# fill the configuration file
cd track_3d
sed -i '' "s~TRACK3DFRAMESTART~$track_3d_frame_start~" configure.ini
sed -i '' "s~TRACK3DFRAMEEND~$track_3d_frame_end~" configure.ini
sed -i '' "s~TRACK3DSAMPLESIZE~$track_3d_sample_size~" configure.ini
sed -i '' "s~TRACK3DTOL2D~$track_3d_tol_2d~" configure.ini
sed -i '' "s~TRACK3DWATERDEPTH~$track_3d_water_depth~" configure.ini
sed -i '' "s~TRACK3DWATERLEVEL~$track_3d_water_level~" configure.ini
sed -i '' "s~CALIBRATIONFOLDER~\.\./$calib_folder~" configure.ini
sed -i '' "s~CALIBRATIONFORMAT~$calib_format~" configure.ini
sed -i '' "s~GRIDSIZE~$grid_size~" configure.ini
sed -i '' "s~CORNERNUMBER~$corner_number~" configure.ini
sed -i '' "s~ORDERJSON~\.\./$order_json~" configure.ini
sed -i '' "s~TRACK3DWANTPLOT~$track_3d_want_plot~" configure.ini
sed -i '' "s~INTERNALCAM1~\.\./$cam_1_internal~" configure.ini
sed -i '' "s~INTERNALCAM2~\.\./$cam_2_internal~" configure.ini
sed -i '' "s~INTERNALCAM3~\.\./$cam_3_internal~" configure.ini
sed -i '' "s~VIDEOFILECAM1~\.\./$video_folder/cam-1.mp4~" configure.ini
sed -i '' "s~VIDEOFILECAM2~\.\./$video_folder/cam-2.mp4~" configure.ini
sed -i '' "s~VIDEOFILECAM3~\.\./$video_folder/cam-3.mp4~" configure.ini
sed -i '' "s~ORIENTATIONNUMBER~$track_2d_orientation_number~g" configure.ini

# Calibration, only calibrate if there is no calibrated camera file
if [ ! -e "cameras.pkl" ]; then
    python3 calibration.py
fi

make pos
make sort

tar -cf locations.tar locations_3d
rm -rf locations_3d

cd -
