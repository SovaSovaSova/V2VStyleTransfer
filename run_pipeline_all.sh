#!/bin/bash
set -e

#############################
# 通用多风格多视频训练 + 推理
#############################

# 获取脚本所在目录并进入
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# 路径配置
CONTENT_DIR="data/coco_content/val2017"
STYLE_DIR="data/styles"
VIDEO_DIR="data/test_video"
OUTPUT_DIR="outputs_all"

mkdir -p "$OUTPUT_DIR"

#############################
# Step 0: 检查 / 下载数据
#############################
if [ ! -d "data/coco_content" ]; then
    echo "Data not found, running download script..."
    if [ -f "download_data.sh" ]; then
        bash "download_data.sh"
    else
        echo "Warning: download_data.sh not found."
    fi
fi

echo ""
echo "========================================================"
echo "开始：遍历所有风格图片并依次训练"
echo "风格图片目录：$STYLE_DIR"
echo "内容图像目录：$CONTENT_DIR"
echo "测试视频目录：$VIDEO_DIR"
echo "输出目录：$OUTPUT_DIR"
echo "========================================================"

#############################
# Step 1: 逐个风格训练
#############################
# 支持 jpg 和 png
for style_image in "$STYLE_DIR"/*.jpg "$STYLE_DIR"/*.png; do
    # 如果没有匹配文件，直接跳过
    if [ ! -f "$style_image" ]; then
        continue
    fi

    style_filename=$(basename "$style_image")    
    style_name="${style_filename%.*}"           

    CHECKPOINT_DIR="checkpoints_${style_name}"

    echo ""
    echo "--------------------------------------------------------"
    echo "正在训练风格：$style_name"
    echo "风格图片路径：$style_image"
    echo "保存权重目录：$CHECKPOINT_DIR"
    echo "--------------------------------------------------------"

    python train.py \
        --content_dir "$CONTENT_DIR" \
        --style_image "$style_image" \
        --save_dir "$CHECKPOINT_DIR" \
        --epochs 5 \
        --batch_size 8 \
        --image_size 256 \
        --content_weight 1.0 \
        --style_weight 10.0 \
        --temp_weight 10.0

    echo "✅ 风格 $style_name 训练完成！"

    #############################
    # Step 2: 对所有视频做推理
    #############################
    echo ""
    echo "========================================================"
    echo "开始对所有视频进行推理（当前风格：$style_name）"
    echo "========================================================"

    LATEST_CKPT="$CHECKPOINT_DIR/decoder_e4.pth"

    if [ ! -f "$LATEST_CKPT" ]; then
        echo "❌ 检查点未找到：$LATEST_CKPT，跳过该风格的推理。"
        continue
    fi

    # 遍历所有测试视频
    for input_video in "$VIDEO_DIR"/*.mp4; do
        if [ ! -f "$input_video" ]; then
            continue
        fi

        video_filename=$(basename "$input_video")    # e.g. test4.mp4
        video_name="${video_filename%.*}"           # e.g. test4

        output_video="$OUTPUT_DIR/${video_name}_${style_name}.mp4"

        echo ""
        echo ">> 正在推理视频：$input_video"
        echo "   风格图片：$style_image"
        echo "   输出视频：$output_video"

        python inference.py \
            --input_video "$input_video" \
            --style_image "$style_image" \
            --output_video "$output_video" \
            --checkpoint "$LATEST_CKPT"
    done

    echo ""
    echo "✅ 风格 $style_name 的所有视频推理完成！"
done

echo ""
echo "🎉 全部风格 & 全部视频的训练与推理已完成！"
