import argparse
import torch
import cv2
import numpy as np
from torchvision import transforms
from models.architecture import VGGEncoder, AdaINStyleTransferNetwork
from PIL import Image
import sys


def inference(args):
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires a GPU.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Load Models
    vgg = VGGEncoder().to(device)
    network = AdaINStyleTransferNetwork(vgg).to(device)

    if args.checkpoint:
        # We only load the decoder weights
        # Note: train.py saves only decoder state dict
        network.decoder.load_state_dict(
            torch.load(args.checkpoint, map_location=device))
        print(f"Loaded decoder from {args.checkpoint}")
    else:
        print("Warning: No checkpoint loaded")

    network.eval()

    # Prepare Style Image
    transform = transforms.Compose([
        transforms.Resize(512),  # High res style
        transforms.ToTensor()
    ])
    style = transform(Image.open(args.style_image).convert(
        "RGB")).unsqueeze(0).to(device)

    # Video Input
    cap = cv2.VideoCapture(args.input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(
        *'mp4v'), fps, (width, height))

    # Processing transform (maintain aspect ratio but downscale for speed if needed)
    process_h = 480
    process_w = int(process_h * (width/height))
    video_transform = transforms.Compose([
        transforms.Resize((process_h, process_w)),
        transforms.ToTensor()
    ])

    print("Processing video with Lab Color Space Preservation...")
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            input_tensor = video_transform(pil_img).unsqueeze(0).to(device)

            # Inference with AdaIN
            g_t, _, _, _ = network(input_tensor, style, alpha=args.alpha)

            g_t = g_t.squeeze(0).cpu().clamp(0, 1)
            res_img = transforms.ToPILImage()(g_t)
            res_img = res_img.resize((width, height), Image.BILINEAR)

            # === Lab Color Space Strategy ===
            # 1. Convert original and stylized frames to float [0, 1]
            original_img = frame.astype(np.float32) / 255.0
            stylized_bgr = cv2.cvtColor(np.array(res_img), cv2.COLOR_RGB2BGR)
            stylized_img = stylized_bgr.astype(np.float32) / 255.0

            # 2. Convert to LAB color space
            original_lab = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)
            stylized_lab = cv2.cvtColor(stylized_img, cv2.COLOR_BGR2LAB)

            # 3. Recombine Channels
            # L (Lightness): Contains structure/shape info -> From Original
            # A, B (Color): Contains style color info -> From Stylized

            # beta controls how much original lightness is preserved
            # 1.0 = Fully original lightness (perfect shape, but maybe dull color)
            # 0.0 = Fully stylized lightness (good style, but shape distorted)
            # 0.5 = Balanced
            beta = 0.45

            final_lab = stylized_lab.copy()
            final_lab[:, :, 0] = original_lab[:, :, 0] * \
                beta + stylized_lab[:, :, 0] * (1 - beta)

            # 4. Convert back to BGR
            out_frame = cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)
            out_frame = (np.clip(out_frame, 0, 1) * 255).astype(np.uint8)

            out.write(out_frame)

    cap.release()
    out.release()
    print(f"Done. Saved to {args.output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--style_image", type=str, required=True)
    parser.add_argument("--output_video", type=str, default="output_adain.mp4")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Stylization strength (0.0-1.0).")
    args = parser.parse_args()
    inference(args)
