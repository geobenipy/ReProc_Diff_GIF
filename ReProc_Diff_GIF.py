#!/usr/bin/env python3
"""
SGY to GIF Comparison Tool
Creates an animated GIF that flips between Pre and Post Processing seismic data
Processes all matching SGY files in the specified directories
"""

import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import segyio
from tqdm import tqdm

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================
CONFIG = {
    # Input directories
    'pre_sgy_dir': r"D:\Haimerl\PhD\Vista\AL641\Export\PyAID\Data\Brutstack",
    'post_sgy_dir': r"D:\Haimerl\PhD\Vista\AL641\Export\PyAID\Label\Brutstack",
    
    # Output settings
    'output_gif_dir': r"C:\Users\u301640\Desktop\GIFS",
    'frame_duration_ms': 500,
    
    # Display settings
    'figure_dpi': 300,
    'figure_size_inches': (15, 10),
    'colormap': 'gray',  # seismics, gray, Greys
    'label_fontsize': 14,
    'label_color': 'white',
    'label_background': 'black',
    
    # Processing
    'normalize_data': True,
    'clip_percentile': 99,  # Clip extreme values at this percentile
    'sgy_extension': '.sgy',  # File extension to search for
}

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# FUNCTIONS
# ============================================================================

def find_sgy_files(directory, extension='.sgy'):
    """Find all SGY files in directory"""
    path = Path(directory)
    # Search for files case-insensitive
    files = []
    for ext in [extension.lower(), extension.upper()]:
        files.extend(path.glob(f'*{ext}'))
    return sorted(set(files))


def load_sgy_data(filepath):
    """Load seismic data from SGY file"""
    logger.info(f"Loading {filepath.name}")
    with segyio.open(str(filepath), ignore_geometry=True) as f:
        data = np.array([np.copy(trace) for trace in f.trace])
    logger.info(f"Loaded shape: {data.shape}")
    return data


def normalize_data(data, clip_percentile=99):
    """mean scaling + clipping"""
    # 1. global mean & std (trace-averaged)
    mean = np.mean(data)
    std = np.std(data)

    data = (data - mean) / std  # ge-mean scaling

    # 2. clip extreme amplitudes
    vmax = np.percentile(np.abs(data), clip_percentile)
    return np.clip(data, -vmax, vmax)


def create_frame(data, label, config):
    """Create a single frame with label"""
    fig = Figure(figsize=config['figure_size_inches'], dpi=config['figure_dpi'])
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    
    im = ax.imshow(
        data.T,
        aspect='auto',
        cmap=config['colormap'],
        interpolation='bilinear'
    )

    ax.text(
        0.02 if label == 'Pre' else 0.98,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=config['label_fontsize'],
        color=config['label_color'],
        verticalalignment='top',
        horizontalalignment='left' if label == 'Pre' else 'right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=config['label_background'], alpha=0.7)
    )

    ax.set_xlabel('Trace')
    ax.set_ylabel('Sample')
    fig.colorbar(im, ax=ax, label='Amplitude')

    # Render canvas
    canvas.draw()
    w, h = canvas.get_width_height()
    img = Image.frombuffer(
        'RGBA',
        (w, h),
        canvas.buffer_rgba(),
        'raw',
        'RGBA',
        0,
        1
    ).convert("RGB")

    plt.close(fig)
    return img


def create_gif(pre_data, post_data, output_path, config):
    """Create animated GIF flipping between Pre and Post"""
    
    # Normalize data if requested
    if config['normalize_data']:
        pre_data = normalize_data(pre_data, config['clip_percentile'])
        post_data = normalize_data(post_data, config['clip_percentile'])
    
    # Create frames
    frames = []
    for label, data in [('Pre', pre_data), ('Post', post_data)]:
        frame = create_frame(data, label, config)
        frames.append(frame)
    
    # Save as GIF
    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:],
        duration=config['frame_duration_ms'],
        loop=0
    )


def process_file_pair(pre_file, post_file, output_dir, config):
    """Process a single pair of SGY files"""
    try:
        # Load data
        pre_data = load_sgy_data(pre_file)
        post_data = load_sgy_data(post_file)
        
        # Validate shapes match
        if pre_data.shape != post_data.shape:
            logger.warning(f"Shape mismatch for {pre_file.name}: Pre {pre_data.shape} vs Post {post_data.shape}")
        
        # Create output filename
        output_filename = f"{pre_file.stem}_comparison.gif"
        output_path = output_dir / output_filename
        
        # Create GIF
        logger.info(f"Creating GIF: {output_filename}")
        create_gif(pre_data, post_data, output_path, config)
        logger.info(f"Saved: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {pre_file.name}: {e}")
        return False


def main():
    """Main execution function"""
    try:
        # Setup paths
        pre_dir = Path(CONFIG['pre_sgy_dir'])
        post_dir = Path(CONFIG['post_sgy_dir'])
        output_dir = Path(CONFIG['output_gif_dir'])
        
        # Validate directories
        if not pre_dir.exists():
            raise FileNotFoundError(f"Pre directory not found: {pre_dir}")
        if not post_dir.exists():
            raise FileNotFoundError(f"Post directory not found: {post_dir}")
        
        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Find all SGY files
        pre_files = find_sgy_files(pre_dir, CONFIG['sgy_extension'])
        post_files = find_sgy_files(post_dir, CONFIG['sgy_extension'])
        
        logger.info(f"Found {len(pre_files)} pre files and {len(post_files)} post files")
        
        if not pre_files:
            raise ValueError(f"No SGY files found in {pre_dir}")
        if not post_files:
            raise ValueError(f"No SGY files found in {post_dir}")
        
        # Match files by suffix (e.g., P101, P102)
        # Extract the last part after underscore (e.g., "P101" from "OnBoard_Brutstack_P101")
        def extract_id(filename):
            """Extract matching ID from filename (last part after underscore)"""
            return filename.stem.split('_')[-1]
        
        pre_dict = {extract_id(f): f for f in pre_files}
        post_dict = {extract_id(f): f for f in post_files}
        
        # Find matching pairs
        matching_ids = set(pre_dict.keys()) & set(post_dict.keys())
        
        if not matching_ids:
            logger.warning("No matching file pairs found!")
            logger.info(f"Pre files: {[f.name for f in pre_files]}")
            logger.info(f"Post files: {[f.name for f in post_files]}")
            return
        
        logger.info(f"Found {len(matching_ids)} matching file pairs")
        
        # Process all pairs
        success_count = 0
        for file_id in tqdm(sorted(matching_ids), desc="Processing file pairs"):
            pre_file = pre_dict[file_id]
            post_file = post_dict[file_id]
            
            if process_file_pair(pre_file, post_file, output_dir, CONFIG):
                success_count += 1
        
        logger.info(f"Successfully processed {success_count}/{len(matching_ids)} file pairs")
        logger.info("Process completed!")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
