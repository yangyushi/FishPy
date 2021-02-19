#!/usr/bin/env python3
import os
import shutil
"""
Update the script folder in auto_process folders
The newer version will be copied to the later version
"""

folders = (
    "auto_process_linux/script",
    "auto_process_mac/script"
)

def get_last_mtime(folder):
    mtimes = []
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        mtime = os.path.getmtime(fpath)
        mtimes.append(mtime)
    return max(mtimes)


mtimes = [get_last_mtime(f) for f in folders]

folders_sorted = [
    x for _, x in sorted(zip(mtimes, folders), key=lambda x: x[0])
]

print(f"Copying from {folders_sorted[1]} to {folders_sorted[0]}")

shutil.rmtree(folders_sorted[0])
shutil.copytree(folders_sorted[1], folders_sorted[0])
