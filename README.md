# SGY to GIF Comparison Tool

This Python script creates animated GIFs that flip between Pre- and Post-Processing seismic data. 
It processes all matching SEG-Y (.sgy) files in the specified input directories and saves the GIFs to an output folder.

## Features
- Automatically finds matching Pre/Post SGY files.
- Normalizes and clips seismic amplitudes.
- Creates labeled frames and an animated GIF for comparison.
- Configurable via the `CONFIG` dictionary in the script.

## Usage
1. Update the `CONFIG` paths in the script for your Pre, Post directories and output GIF folder.
2. Run the script:

```bash
python sgy_to_gif.py
