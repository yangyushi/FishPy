#!/usr/bin/env bash

SN=${0##*/}  # script name
fishpy=${0%/script/bin/$SN}  # path to fishpy
platform=$(python -mplatform)
platform=$(expr "$platform" : '\([A-Za-z]*\)-') # linux or darwin
platform=${platform,,}  # all letters to lowercase

if [[ $platform == "darwin" ]]; then
    platform="mac"
fi

if [[ ! $platform == "mac" ]] && [[ ! $platform == "linux" ]]; then
    echo "not supporting current platform: $platform"
fi

# Check if PWD is a auto_process folder
# if not, set $should_update to 0 and do not update
should_update=1
needed=(
    'auto_link.sh' 'auto_track_2d.sh' 'auto_track_3d.sh'
    'configure.sh' 'see_trajs.py' 'clean'
)
for fn in "${needed[@]}"; do
    if [[ ! -e $fn ]]; then
        echo "$fn missing"
        should_update=0
    fi
done
if [[ ! -d script ]]; then
    echo "script missing"
    should_update=0
fi

if [[ $should_update -eq 0 ]]; then
    echo "current folder is not a valid FishPy auto process folder"
    exit 1
fi

if [[ $should_update -eq 1 ]]; then
    rm "$PWD"/*.sh
    rm "$PWD"/*.py
    rm "$PWD/"*.md
    rm "$PWD/clean"
    rm -rf "$PWD/script"

    cp "$fishpy/script/auto_process_$platform/"*.sh $PWD
    cp "$fishpy/script/auto_process_$platform/"*.py $PWD
    cp "$fishpy/script/auto_process_$platform/"*.md $PWD
    cp "$fishpy/script/auto_process_$platform/clean" $PWD
    cp -r "$fishpy/script/auto_process_$platform/script" $PWD
    echo "all scripts updated"
    exit 0
fi
