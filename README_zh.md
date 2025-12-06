# AdaIN + CCPL + Luminance Preservation（视频风格迁移）

该项目实现了基于 AdaIN 的风格迁移与 CCPL（Contrastive Coherence Preserving Loss）时序一致性约束，并在推理阶段通过 Lab 色彩空间融合保留原视频的亮度结构，使风格化结果兼具风格色彩与清晰形状。

## 特性

- AdaIN 自适应实例归一化进行风格对齐
- CCPL 对比时序一致性损失，提升视频连续帧的稳定性
- 亮度保留：在 Lab 空间融合 L 通道，最大化结构与形状保真度
- 仅训练解码器，VGG19 编码器固定（预训练 ImageNet 权重）
- 支持单风格训练、批量训练与批量视频推理流水线

## 仓库结构

```
.
├── models/
│   └── architecture.py        # VGG 编码器、解码器、AdaIN 前向
├── train.py                   # 训练入口（仅训练 decoder）
├── inference.py               # 视频推理（AdaIN + 亮度保留）
├── loss.py                    # CCPLLoss 实现
├── download_data.sh           # 示例数据下载与目录初始化
├── run_inference_demo.sh      # 单风格单视频推理示例
├── run_pipeline_all.sh        # 多风格多视频批量训练与推理流水线
├── requirements.txt           # 依赖列表
├── data/                      # 数据目录（脚本创建/下载）
├── checkpoints/               # 预训练的风格迁移解码器权重（.pth）
├── checkpoints_*/             # 各风格训练输出（权重与可视化）
└── outputs*/                  # 推理输出视频
```

## 环境要求与安装

- Python ≥ 3.8（建议 3.8–3.11）
- PyTorch ≥ 1.12.0，TorchVision ≥ 0.13.0
- CUDA 支持的 NVIDIA GPU（推理脚本强制要求 GPU；训练强烈建议使用 GPU）
- 其他依赖：`numpy`、`opencv-python`、`Pillow`

安装依赖：

```bash
pip install -r requirements.txt
```

## 数据准备

推荐使用脚本自动下载并组织数据：

```bash
bash download_data.sh
```

将生成如下目录结构：

```
data/
├── coco_content/
│   └── val2017/               # COCO 2017 验证集（约 5000 张）
├── styles/
│   └── starry_night.jpg       # 示例风格图
└── test_video/
    └── test.mp4               # 示例测试视频
```

也可手动准备同名目录与文件路径。

## 快速开始

### 单风格训练

```bash
python train.py \
  --content_dir data/coco_content/val2017 \
  --style_image data/styles/starry_night.jpg \
  --save_dir checkpoints_adain \
  --epochs 5 \
  --batch_size 4 \
  --image_size 256
```

- 仅训练 `decoder`，权重保存至 `--save_dir`（每个 epoch 保存一次）
- 训练过程中会在 `save_dir` 生成可视化图：`vis_e{epoch}_s{i}.jpg` 与 `validation_epoch_{epoch}.jpg`

### 视频推理（亮度保留）

```bash
python inference.py \
  --input_video data/test_video/test.mp4 \
  --style_image data/styles/starry_night.jpg \
  --checkpoint checkpoints/decoder_starry_night.pth \
  --output_video outputs/stylized_video.mp4 \
  --alpha 1.0
```

- `alpha` 控制风格强度（0.0–1.0）
- 推理阶段在 Lab 空间融合亮度通道，默认融合系数 `beta=0.45`（可在 `inference.py` 中调整）
- 推理脚本会验证 CUDA 是否可用，如不可用将退出

#### 使用预训练权重

- 可直接使用 `checkpoints/` 中任意 `.pth` 作为解码器权重进行推理。
- 为获得更稳定的色彩与纹理效果，建议与训练该权重时所用的 `--style_image` 保持一致。
- 示例：

```bash
python inference.py \
  --input_video data/test_video/test.mp4 \
  --style_image data/styles/ukiyoe.jpg \
  --checkpoint checkpoints/decoder_ukiyoe.pth \
  --output_video outputs/ukiyoe_result.mp4
```

### 推理演示脚本

```bash
bash run_inference_demo.sh
```

- 自动从 `checkpoints_adain/` 中选择最新的 `.pth` 作为权重
- 输出至 `outputs/result_demo_shape.mp4`

### 批量训练与推理流水线

```bash
bash run_pipeline_all.sh
```

- 遍历 `data/styles` 中所有风格图进行训练（默认 5 个 epoch）
- 对 `data/test_video/*.mp4` 进行全量推理，结果保存到 `outputs_all/`

## 脚本参数说明

### train.py

- `--content_dir`：内容图目录（必填）
- `--style_image`：风格图路径（必填）
- `--save_dir`：检查点保存目录，默认 `checkpoints_adain`
- `--epochs`：训练轮数，默认 `5`
- `--batch_size`：批大小，默认 `4`
- `--lr`：学习率，默认 `1e-4`
- `--image_size`：训练输入尺寸，默认 `256`
- `--content_weight`：内容损失权重，默认 `1.0`
- `--style_weight`：风格损失权重，默认 `10.0`
- `--temp_weight`：时序一致性（CCPL）损失权重，默认 `10.0`

### inference.py

- `--input_video`：输入视频路径（必填）
- `--style_image`：风格图路径（必填）
- `--output_video`：输出视频路径，默认 `outputs/output_adain.mp4`
- `--checkpoint`：训练得到的解码器权重（必填，`decoder_e*.pth`）
- `--alpha`：风格强度（0.0–1.0），默认 `1.0`

## 工作原理（简述）

- 编码器：固定 `VGG19` 预训练特征，切片至 `relu4_1`
- 解码器：与 VGG 对称的反卷积结构，将风格化深层特征还原到图像空间
- AdaIN：对齐内容特征的均值与方差至风格特征，实现风格迁移与强度插值（`alpha`）
- 损失构成：
  - 内容损失：编码器输出与目标特征 `t` 的 MSE
  - 风格损失：多层特征的均值/方差 MSE（统计匹配）
  - CCPL：对比式时序一致性，训练中通过仿射网格合成运动并对上一帧进行对齐，鼓励连续帧的一致性
- 亮度保留：推理阶段将原始视频的 L 通道与风格化视频的 A/B 通道融合，平衡结构与风格色彩

## 输出与日志

- 训练权重：`{save_dir}/decoder_e{epoch}.pth`
- 训练可视化：`{save_dir}/vis_e{epoch}_s{i}.jpg`
- 验证可视化：`{save_dir}/validation_epoch_{epoch}.jpg`
- 推理视频：`outputs/*.mp4`、`outputs_all/*.mp4`

## 常见问题（FAQ）

- CUDA 不可用或报错：请确认已安装支持的 NVIDIA 驱动与 CUDA，PyTorch 能正常使用 GPU；推理脚本在无 CUDA 时会直接退出。
- 导入错误（`models.architecture` 找不到）：请在项目根目录执行命令，确保相对导入路径正确。
- OpenCV 视频编码失败：请确认系统 OpenCV 支持 `mp4v` 编码器，或修改 `inference.py` 中的 fourcc 设置与输出后缀。
- 风格过强或形状失真：下调 `--alpha`（如 0.6–0.8），或在 `inference.py` 中适当提高 `beta` 保留更多原始亮度。
- 是否必须使用仓库自带的权重？不必。`checkpoints/` 提供快速演示。若需自定义风格，请用你的风格图运行 `train.py` 训练得到 `decoder_e*.pth`，再用于推理。

## 参考文献

- AdaIN: Xun Huang, Serge Belongie. Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization (ICCV 2017)
- CCPL: Contrastive Coherence Preserving Loss for Video Style Transfer (ECCV 2022)

## 许可证

MIT license。

## 致谢

感谢开源社区与相关论文作者的贡献。本项目使用了 PyTorch 与 TorchVision 等优秀开源库。
