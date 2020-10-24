#!/usr/bin/env bash
# copy the processing folder into a FishPy Tracking project folder
#
# How to use:
#   inside the project folder, execute the script
#
# What will happen
#    the folder [auto_process_*] to the project folder according
#    to the number of sub-folders whose name start with [video]
#    if the platform is darwin copy [auto_process_mac]
#    else copy the [auto_process_linux]
#    I don't have a windows machine to support windows
# 
# Example
#    working folder contains: video_1, video_2, video_3
#    created new folder:
#       auto_process_video_1,
#       auto_process_video_2,
#       auto_process_video_3
#    if the target folder is already created, then skip the existing folder
#    if you want to serach folders with different prefix, please provide the
#        script with such prefix as extra argument

# find the path of fishpy
SN=${0##*/}  # script name
fishpy=${0%/script/bin/$SN}  # path to fishpy

# detect the current platform, mac or linux
platform=$(python -mplatform)
platform=$(expr "$platform" : '\([A-Za-z]*\)-') # linux or darwin
platform="$(echo ${platform} | tr '[:upper:]' '[:lower:]')"
if [[ ${platform} == "darwin" ]]; then
    platform="mac"
fi
if [[ ! $platform == "mac" ]] && [[ ! $platform == "linux" ]]; then
    echo "the current platform is not supported: $platform"
    exit 1;
fi

# set the default prefix to video
if [[ -z $1 ]]; then
    prefix="video"
else
    prefix=$1
fi

target_folders=()
for path in "$PWD"/*; do
    fn=${path##*/}
    if [[ $fn =~ $prefix.+ ]]; then
        fn=${fn/-/_}
        target="$PWD/auto_process${fn/$prefix/}"
        target_folders+=("$target")
    fi
done

for target in "${target_folders[@]}"; do
    if [[ ! -e $target ]]; then
        cp -r "$fishpy/script/auto_process_$platform" "$PWD"
        mv "auto_process_$platform" ${target##*/}
    else
        echo "${target##*/} already exists, skip"
    fi
done
