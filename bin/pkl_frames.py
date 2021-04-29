#!/usr/bin/env python3
import pickle
import sys

fn = sys.argv[1]

f = open(fn, 'rb')
frames = 0

while f:
    try:
        pickle.load(f)
        frames += 1
    except EOFError:
        break

print(frames)
