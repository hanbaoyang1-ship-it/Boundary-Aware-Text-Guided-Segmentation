from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bacl.config import load_config
from bacl.metrics import binary_segmentation_metrics, normal_confidence_interval
from bacl.runtime import build_dataset, build_model, load_checkpoint
from bacl.utils import choose_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BACL on a labeled split")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--output-dir", default="results/evaluation")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device(config.get("device", "auto"))
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()
    dataset = build_dataset(config, args.split, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("evaluation", {}).get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config.get("evaluation", {}).get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            outputs = model(images, list(batch["text"]))
            dice, iou = binary_segmentation_metrics(outputs["logits"], masks, args.threshold)
            for image_id, dice_value, iou_value in zip(
                batch["id"], dice.cpu().tolist(), iou.cpu().tolist(), strict=True
            ):
                rows.append({"id": image_id, "dice": dice_value, "iou": iou_value})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_case.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "dice", "iou"))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "threshold": args.threshold,
        "dice": normal_confidence_interval([float(row["dice"]) for row in rows]),
        "iou": normal_confidence_interval([float(row["iou"]) for row in rows]),
    }
    save_json(summary, output_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
