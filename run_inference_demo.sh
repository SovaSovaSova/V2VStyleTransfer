#!/bin/bash
set -e

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Change to the script directory to ensure relative paths work
cd "$SCRIPT_DIR"

# --- Configuration ---
INPUT_VIDEO="data/test_video/test.mp4"
STYLE_IMAGE="data/styles/starry_night.jpg"
OUTPUT_VIDEO="outputs/result_demo_shape.mp4"
CHECKPOINT_DIR="checkpoints_adain"

# Create outputs directory if it doesn't exist
mkdir -p outputs

ALPHA=1.0

# 1. Check/Find Checkpoint
if [ -z "$CKPT_PATH" ]; then
    # Try to auto-detect latest checkpoint if not provided via env var
    CKPT_PATH=$(ls -t "$CHECKPOINT_DIR"/*.pth 2>/dev/null | head -n 1)
fi

if [ -z "$CKPT_PATH" ] || [ ! -f "$CKPT_PATH" ]; then
    echo "⚠️  Warning: Configured checkpoint not found: $CKPT_PATH"
    echo "Trying to auto-detect latest checkpoint again..."
    CKPT_PATH=$(ls -t "$CHECKPOINT_DIR"/*.pth 2>/dev/null | head -n 1)
    if [ -z "$CKPT_PATH" ]; then
        echo "Error: No checkpoints found at all in $CHECKPOINT_DIR."
        exit 1
    fi
fi

echo "========================================================"
echo "🎬 Starting Inference Demo"
echo "========================================================"
echo "checkpoint: $CKPT_PATH"
echo "input:      $INPUT_VIDEO"
echo "style:      $STYLE_IMAGE"
echo "output:     $OUTPUT_VIDEO"
echo "========================================================"

# 2. Run Inference Python Script
python inference.py \
    --input_video "$INPUT_VIDEO" \
    --style_image "$STYLE_IMAGE" \
    --output_video "$OUTPUT_VIDEO" \
    --checkpoint "$CKPT_PATH" \
    --alpha "$ALPHA"

echo ""
echo "✅ Done! Saved to $OUTPUT_VIDEO"
echo "You can now download or open this file manually to view the result."
