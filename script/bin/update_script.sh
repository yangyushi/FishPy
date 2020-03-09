#!/usr/bin/env bash
# update the script in these folders
#   * auto_proces_linux
#   * auto_proces_mac
#   * auto_analysis

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

should_update=0

# Check if PWD is a auto_process folder
update_process=1
needed=(
    'auto_link.sh' 'auto_track_2d.sh' 'auto_track_3d.sh'
    'see_trajs.py'
)
for fn in "${needed[@]}"; do
    if [[ ! -e $fn ]]; then
        update_process=0
    fi
done
if [[ ! -d script ]]; then
    update_process=0
fi

# Check if PWD is a auto_analysis folder
update_analysis=1
needed=(
    'auto_analysis.sh'
)
for fn in "${needed[@]}"; do
    if [[ ! -e $fn ]]; then
        update_analysis=0
    fi
done
if [[ ! -d script ]]; then
    update_analysis=0
fi

should_update=$(($update_analysis + $update_process))

if [[ $should_update -eq 0 ]]; then
    echo "not a valid FishPy project folder!"
    exit 1
fi

if [[ $update_process -eq 1 ]]; then
    for fn in "${needed[@]}"; do
        rm "$PWD/$fn"
        cp "$fishpy/script/auto_process_$platform/$fn" "$PWD"
    done
    cp "$fishpy/script/auto_process_$platform/collect.sh" "$PWD"
    rm -rf "$PWD/script"
    cp -r "$fishpy/script/auto_process_$platform/script" "$PWD"
    echo "Process scripts updated"
    exit 0
fi

if [[ $update_analysis -eq 1 ]]; then
    rm -rf "$PWD/script"
    cp -r "$fishpy/script/auto_analysis/script" "$PWD"
    echo "Analysis scripts updated"
    exit 0
fi
