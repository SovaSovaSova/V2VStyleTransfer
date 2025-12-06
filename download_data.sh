#!/bin/bash
set -e

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Base directories
DATA_DIR="data"
CONTENT_DIR="$DATA_DIR/coco_content"
STYLE_DIR="$DATA_DIR/styles"
VIDEO_DIR="$DATA_DIR/test_video"

mkdir -p "$CONTENT_DIR"
mkdir -p "$STYLE_DIR"
mkdir -p "$VIDEO_DIR"

echo "========================================================"
echo "Step 1: Downloading COCO 2017 Validation Set (Content Images)"
echo "========================================================"
# We use Val2017 (1GB) instead of Train2017 (18GB) to save time/space.
# 5000 images is MORE than enough for style transfer training.
if [ ! -d "$CONTENT_DIR/val2017" ]; then
    echo "Downloading COCO Val2017 zip..."
    wget -c http://images.cocodataset.org/zips/val2017.zip -O "$CONTENT_DIR/val2017.zip"
    
    echo "Unzipping..."
    unzip -q "$CONTENT_DIR/val2017.zip" -d "$CONTENT_DIR"
    rm "$CONTENT_DIR/val2017.zip"
    echo "Content images ready at $CONTENT_DIR/val2017"
else
    echo "COCO Val2017 already exists. Skipping download."
fi

echo ""
echo "========================================================"
echo "Step 2: Downloading Style Image (Van Gogh - Starry Night)"
echo "========================================================"
STYLE_URL="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
if [ ! -f "$STYLE_DIR/starry_night.jpg" ]; then
    wget "$STYLE_URL" -O "$STYLE_DIR/starry_night.jpg"
    echo "Style image saved to $STYLE_DIR/starry_night.jpg"
else
    echo "Style image already exists."
fi

echo ""
echo "========================================================"
echo "Step 3: Downloading Test Video"
echo "========================================================"
# Downloading a short copyright-free sample video
VIDEO_URL="https://media.githubusercontent.com/media/jepsonxyz/deep-learning-data/master/video/test_video.mp4"
# If that link fails, we use a fallback or you can place your own. 
# Let's use a reliable sample from a public repo.
if [ ! -f "$VIDEO_DIR/test.mp4" ]; then
    # Try to download a sample video (e.g. from a public S3 or repo)
    # Using a specific short nature clip
    wget "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4" -O "$VIDEO_DIR/test.mp4"
    echo "Test video saved to $VIDEO_DIR/test.mp4"
else
    echo "Test video already exists."
fi

echo ""
echo "Done! All data is ready in $DATA_DIR"
