#!/usr/bin/env bash
# update the script in these folders
#   * auto_proces_linux
#   * auto_proces_mac
#   * auto_analysis

SN=${0##*/}  # script name
fishpy=${0%/script/bin/$SN}  # path to fishpy
platform=$(python -mplatform)
platform=$(expr "$platform" : '\([A-Za-z]*\)-') # linux or darwin
#platform="${platform,,}"  # all letters to lowercase
platform="$(echo ${platform} | tr '[:upper:]' '[:lower:]')"

if [[ ${platform} == "darwin" ]]; then
    platform="mac"
fi

if [[ ! $platform == "mac" ]] && [[ ! $platform == "linux" ]]; then
    echo "not supporting current platform: $platform"
fi

should_update=0

# Check if PWD is a auto_process folder
update_process=1
needed_process=(
    'auto_link.sh' 'auto_track_2d.sh' 'auto_track_3d.sh'
    'see_trajs.py'
)
for fn in "${needed_process[@]}"; do
    if [[ ! -e $fn ]]; then
        update_process=0
    fi
done
if [[ ! -d script ]]; then
    update_process=0
fi

# Check if PWD is a auto_analysis folder
update_analysis=1
needed_analysis=(
    'auto_analysis.sh'
)
for fn in "${needed_analysis[@]}"; do
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
    echo "updating process scripts"
    for fn in "${needed_process[@]}"; do
        rm "$PWD/$fn"
        cp "$fishpy/script/auto_process_$platform/$fn" "$PWD"
    done
    cp "$fishpy/script/auto_process_$platform/auto_track_greta.sh" "$PWD"
    cp "$fishpy/script/auto_process_$platform/collect.sh" "$PWD"
    cp "$fishpy/script/auto_process_$platform/stat_trajs.py" "$PWD"
    rm -rf "$PWD/script"
    cp -r "$fishpy/script/auto_process_$platform/script" "$PWD"
    exit 0
fi

if [[ $update_analysis -eq 1 ]]; then
    echo "updating analysis scripts"
    rm -rf "$PWD/script"
    cp -r "$fishpy/script/auto_analysis/script" "$PWD"
    exit 0
fi
