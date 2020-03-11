#!/usr/bin/env bash
# Extract 1st frames of all videos
for fn in ./*.mp4; do
    ffmpeg -y -i ${fn} -vframes 1 ${fn%.mp4}.png  &> /dev/null
done
