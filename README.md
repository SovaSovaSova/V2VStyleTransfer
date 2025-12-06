# AdaIN + CCPL + Luminance Preservation (Video Style Transfer)

This project implements video style transfer using AdaIN together with CCPL (Contrastive Coherence Preserving Loss). During inference, it preserves luminance via Lab color space fusion so the results keep stylistic colors while maintaining clear structure.

## Features

- AdaIN for aligning content and style statistics
- CCPL for temporal coherence across consecutive frames
- Luminance preservation: fuse L channel in Lab space to retain structure
- Train decoder only; VGG19 encoder is fixed (ImageNet weights)
- Single-style training, batch training, and batch video inference pipeline

## Repository Structure

```
.
├── models/
│   └── architecture.py        # VGG encoder, decoder, AdaIN forward
├── train.py                   # Training entry (decoder-only)
├── inference.py               # Video inference (AdaIN + luminance preservation)
├── loss.py                    # CCPLLoss implementation
├── download_data.sh           # Sample data download and setup
├── run_inference_demo.sh      # Single-style single-video demo
├── run_pipeline_all.sh        # Multi-style multi-video pipeline
├── requirements.txt           # Dependencies
├── results                    # v2v demo
└── styles                     # style picture 
```

## Requirements & Setup

- Python ≥ 3.8 (recommended 3.8–3.11)
- PyTorch ≥ 1.12.0, TorchVision ≥ 0.13.0
- NVIDIA GPU with CUDA (GPU required for inference; training strongly recommended on GPU)
- Other deps: `numpy`, `opencv-python`, `Pillow`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

Use the helper script to download and organize sample data:

```bash
bash download_data.sh
```

It creates the following structure:

```
data/
├── coco_content/
│   └── val2017/               # COCO 2017 validation set (~5000 images)
├── styles/
│   └── starry_night.jpg       # Example style image
└── test_video/
    └── test.mp4               # Example test video
```

You can also prepare equivalent folders and files manually.

## Quick Start

### Single-Style Training

```bash
python train.py \
  --content_dir data/coco_content/val2017 \
  --style_image data/styles/starry_night.jpg \
  --save_dir checkpoints_adain \
  --epochs 5 \
  --batch_size 4 \
  --image_size 256
```

- Trains the `decoder` only; checkpoints are saved to `--save_dir` per epoch
- Visualizations are written to `save_dir`: `vis_e{epoch}_s{i}.jpg`, `validation_epoch_{epoch}.jpg`

### Video Inference (Luminance Preservation)

```bash
python inference.py \
  --input_video data/test_video/test.mp4 \
  --style_image data/styles/starry_night.jpg \
  --checkpoint checkpoints/decoder_starry_night.pth \
  --output_video outputs/stylized_video.mp4 \
  --alpha 1.0
```

- `alpha` controls style strength (0.0–1.0)
- Inference fuses luminance in Lab space with coefficient `beta=0.45` (modifiable in `inference.py`)
- Script checks CUDA availability and exits if no GPU

#### Using Pretrained Checkpoints

- You can directly use any `.pth` in `checkpoints/` as the decoder weights.
- For best results, use the matching `--style_image` that was used to train the chosen checkpoint.
- Example:

```bash
python inference.py \
  --input_video data/test_video/test.mp4 \
  --style_image data/styles/ukiyoe.jpg \
  --checkpoint checkpoints/decoder_ukiyoe.pth \
  --output_video outputs/ukiyoe_result.mp4
```

### Demo Script

```bash
bash run_inference_demo.sh
```

- Auto-detects the latest `.pth` in `checkpoints_adain/`
- Outputs to `outputs/result_demo_shape.mp4`

### Batch Pipeline

```bash
bash run_pipeline_all.sh
```

- Iterates all style images in `data/styles` (default 5 epochs)
- Runs inference for `data/test_video/*.mp4`, saves results to `outputs_all/`

## Script Arguments

### train.py

- `--content_dir`: content images directory (required)
- `--style_image`: style image path (required)
- `--save_dir`: checkpoint output directory, default `checkpoints_adain`
- `--epochs`: training epochs, default `5`
- `--batch_size`: batch size, default `4`
- `--lr`: learning rate, default `1e-4`
- `--image_size`: input image size, default `256`
- `--content_weight`: content loss weight, default `1.0`
- `--style_weight`: style loss weight, default `10.0`
- `--temp_weight`: temporal (CCPL) loss weight, default `10.0`

### inference.py

- `--input_video`: input video path (required)
- `--style_image`: style image path (required)
- `--output_video`: output video path, default `outputs/output_adain.mp4`
- `--checkpoint`: decoder weights from training (required, `decoder_e*.pth`)
- `--alpha`: style strength (0.0–1.0), default `1.0`

## How It Works (Brief)

- Encoder: fixed `VGG19` pretrained features up to `relu4_1`
- Decoder: symmetric deconvolutional stack that maps stylized deep features back to image space
- AdaIN: aligns mean/std of content features to style features; `alpha` interpolates strength
- Losses:
  - Content loss: MSE between encoder output and target feature `t`
  - Style loss: MSE on mean/std across multiple layers (statistical matching)
  - CCPL: contrastive temporal coherence; synthetic motion via affine grid and alignment to encourage frame-to-frame consistency
- Luminance preservation: fuse original L with stylized A/B channels in Lab space for structure-color balance

## Outputs & Logs

- Checkpoints: `{save_dir}/decoder_e{epoch}.pth`
- Training visuals: `{save_dir}/vis_e{epoch}_s{i}.jpg`
- Validation visuals: `{save_dir}/validation_epoch_{epoch}.jpg`
- Inference videos: `outputs/*.mp4`, `outputs_all/*.mp4`

## FAQ

- CUDA not available: ensure NVIDIA driver and CUDA are installed and PyTorch can access GPU; inference exits if no CUDA.
- Import errors (`models.architecture`): run commands from project root to keep relative imports valid.
- OpenCV codec errors: ensure system OpenCV supports `mp4v`, or change fourcc/output extension in `inference.py`.
- Style too strong or shape distorted: reduce `--alpha` (e.g., 0.6–0.8) or increase `beta` in `inference.py` to preserve more luminance.
- Are checkpoints mandatory? No. They are provided for quick demo. For custom styles, train your own decoder with `train.py` using your style image, then use the resulting `decoder_e*.pth` in inference.

## References

- AdaIN: Xun Huang, Serge Belongie. Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization (ICCV 2017)
- CCPL: Contrastive Coherence Preserving Loss for Video Style Transfer (ECCV 2022)

## License

MIT License

## Acknowledgements

Thanks to the open-source community and the authors of related papers. This project uses PyTorch, TorchVision, and other excellent libraries.

---

For the Chinese version, see `README_zh.md`.
