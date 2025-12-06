import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import torch.nn.functional as F

from models.architecture import VGGEncoder, AdaINStyleTransferNetwork, calc_mean_std
from loss import CCPLLoss

# --- Custom Dataset ---


class FlatImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = [os.path.join(root, f) for f in os.listdir(root)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        try:
            img = Image.open(path).convert('RGB')
        except:
            return self.__getitem__(random.randint(0, len(self) - 1))
        if self.transform:
            img = self.transform(img)
        return img


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    # Content Transform: Random Crop is good for data augmentation
    content_transform = transforms.Compose([
        transforms.Resize(int(args.image_size * 1.1)),
        transforms.RandomCrop(args.image_size),
        transforms.ToTensor()
    ])

    # Style Transform: We want global style, so Resize + CenterCrop or just Resize
    # RandomCrop on style might lose important textures if the style image is not uniform.
    style_transform = transforms.Compose([
        # Force resize to match input dim
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])

    # Load Data
    if not os.path.exists(args.content_dir):
        print("Content dir not found.")
        return
    dataset = FlatImageFolder(args.content_dir, transform=content_transform)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Style Image
    style_img = style_transform(Image.open(args.style_image).convert(
        "RGB")).unsqueeze(0).to(device)
    # Repeat style to match batch size
    style_batch = style_img.repeat(args.batch_size, 1, 1, 1)

    # Models
    vgg = VGGEncoder().to(device)
    network = AdaINStyleTransferNetwork(vgg).to(device)

    # Optimizer: We only train the decoder!
    optimizer = optim.Adam(network.decoder.parameters(), lr=args.lr)
    ccpl_loss = CCPLLoss()

    # --- Validation Setup: Load a FIXED image to monitor progress ---
    val_img_path = os.path.join(
        args.content_dir, os.listdir(args.content_dir)[0])
    val_img = Image.open(val_img_path).convert('RGB')
    # Keep aspect ratio but resize small side to image_size
    val_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()
    ])
    val_tensor = val_transform(val_img).unsqueeze(0).to(device)  # [1, 3, H, W]
    print(f"Validation image loaded: {val_img_path}")

    print("Starting AdaIN + CCPL training...")

    for epoch in range(args.epochs):
        for i, content_batch in enumerate(dataloader):
            content_batch = content_batch.to(device)

            # --- 1. Synthetic Motion (for Temporal Loss) ---
            B, C, H, W = content_batch.shape

            # Generate affine grid
            angle = random.uniform(-5, 5)
            translate = (random.uniform(-0.05, 0.05),
                         random.uniform(-0.05, 0.05))
            scale = random.uniform(0.95, 1.05)
            theta = torch.tensor([
                [scale * torch.cos(torch.tensor(angle)), -
                 torch.sin(torch.tensor(angle)), translate[0]],
                [torch.sin(torch.tensor(angle)), scale *
                 torch.cos(torch.tensor(angle)), translate[1]]
            ]).unsqueeze(0).repeat(B, 1, 1).to(device)
            grid = F.affine_grid(
                theta, content_batch.size(), align_corners=False)

            # Frame 1 (Prev) and Frame 2 (Curr = Warped Prev)
            prev_frame = content_batch
            curr_frame_warped = F.grid_sample(
                prev_frame, grid, align_corners=False)

            # Mask
            mask = torch.ones_like(prev_frame)
            mask_warped = F.grid_sample(mask, grid, align_corners=False)
            mask_warped = (mask_warped > 0.9).float()

            # --- 2. Forward Pass ---
            # We stylize BOTH frames
            # g_t: generated image, t: adain target feature
            g_t_prev, t_prev, _, _ = network(prev_frame, style_batch)
            g_t_curr, t_curr, _, _ = network(curr_frame_warped, style_batch)

            # --- 3. Loss Calculation ---

            # A. Content Loss: MSE(Encoder(g_t), t)
            # We pass generated image back through VGG to get its features
            g_t_curr_feats = vgg(g_t_curr)
            loss_c = F.mse_loss(g_t_curr_feats[-1], t_curr)

            # B. Style Loss: MSE(MeanStd(Encoder(g_t)), MeanStd(Style))
            # We check multiple layers for style consistency
            loss_s = 0.0
            style_feats = vgg(style_batch)  # Fixed style features
            for g_f, s_f in zip(g_t_curr_feats, style_feats):
                m_g, s_g = calc_mean_std(g_f)
                m_s, s_s = calc_mean_std(s_f)
                loss_s += F.mse_loss(m_g, m_s) + F.mse_loss(s_g, s_s)

            # C. CCPL (Temporal) Loss
            # Warp the previous stylized image to current position
            g_t_prev_warped = F.grid_sample(
                g_t_prev, grid, align_corners=False)

            # Get features of warped previous output
            g_t_prev_warped_feats = vgg(g_t_prev_warped)

            # Apply CCPL on deep features (relu4_1)
            mask_feat = F.interpolate(
                mask_warped, size=g_t_curr_feats[-1].shape[-2:], mode='nearest')
            loss_temp = ccpl_loss(
                g_t_curr_feats[-1], g_t_prev_warped_feats[-1], mask=mask_feat)

            # Total Loss
            loss = (args.content_weight * loss_c +
                    args.style_weight * loss_s +
                    args.temp_weight * loss_temp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- Logging ---
            if i % 50 == 0:
                print(
                    f"Epoch {epoch} [{i}/{len(dataloader)}] Loss: {loss.item():.4f} (C:{loss_c.item():.4f}, S:{loss_s.item():.4f}, T:{loss_temp.item():.4f})")

                vis = torch.cat(
                    [curr_frame_warped, g_t_curr.clamp(0, 1)], dim=3)
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                save_image(vis, os.path.join(
                    args.save_dir, f"vis_e{epoch}_s{i}.jpg"))

        # Save
        torch.save(network.decoder.state_dict(), os.path.join(
            args.save_dir, f"decoder_e{epoch}.pth"))

        # --- Validation: Inference on fixed image ---
        if val_tensor is not None:
            with torch.no_grad():
                network.eval()
                # Match validation style batch size to 1
                val_style = style_img  # [1, 3, H, W]
                val_out, _, _, _ = network(val_tensor, val_style, alpha=1.0)

                # Visualize: [Input | Output]
                # Resize output to match input if needed (usually they match due to architecture)
                val_vis = torch.cat([val_tensor, val_out.clamp(0, 1)], dim=3)
                save_image(val_vis, os.path.join(
                    args.save_dir, f"validation_epoch_{epoch}.jpg"))
                network.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", type=str, required=True)
    parser.add_argument("--style_image", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints_adain")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--content_weight", type=float, default=1.0)
    # AdaIN needs much smaller style weight than Gram!
    parser.add_argument("--style_weight", type=float, default=10.0)
    parser.add_argument("--temp_weight", type=float, default=10.0)
    args = parser.parse_args()

    train(args)
