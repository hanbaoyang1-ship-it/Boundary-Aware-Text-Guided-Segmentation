from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bacl.config import load_config
from bacl.runtime import build_model, load_checkpoint
from bacl.utils import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BACL inference on one 2-D image")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device(config.get("device", "auto"))
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    original_size = image.size
    image_size = int(config.get("data", {}).get("image_size", 224))
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = ((tensor - 0.5) / 0.5).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor, [args.text])["logits"]
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = Image.fromarray((probability >= args.threshold).astype(np.uint8) * 255)
    mask = mask.resize(original_size, Image.Resampling.NEAREST)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output_path)


if __name__ == "__main__":
    main()
