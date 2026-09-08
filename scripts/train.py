from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bacl.config import load_config
from bacl.losses import bce_dice_loss
from bacl.metrics import binary_segmentation_metrics
from bacl.runtime import build_contrastive_loss, build_dataset, build_model, contrastive_weight
from bacl.utils import choose_device, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BACL text-guided segmentation")
    parser.add_argument("--config", required=True, help="Dataset/training YAML file")
    parser.add_argument("--output-dir", default=None, help="Override the run output directory")
    parser.add_argument("--resume", default=None, help="Resume from a release-format checkpoint")
    return parser.parse_args()


def run_epoch(model, boundary_loss, loader, device, optimizer, boundary_lambda, amp_enabled, scaler=None):
    training = optimizer is not None
    model.train(training)
    boundary_loss.train(training)
    total_loss = total_seg = total_boundary = 0.0
    dice_values: list[float] = []
    iou_values: list[float] = []
    sample_count = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        texts = list(batch["text"])
        batch_size = images.shape[0]
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, enabled=amp_enabled
        ):
            outputs = model(images, texts, mask_images=masks if training else None)
            segmentation_loss = bce_dice_loss(outputs["logits"], masks)
            if training:
                contrastive = boundary_loss(
                    outputs["image_features"], outputs["mask_features"], masks
                )
                loss = segmentation_loss + boundary_lambda * contrastive
            else:
                contrastive = segmentation_loss.detach() * 0.0
                loss = segmentation_loss

        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(boundary_loss.parameters()), max_norm=1.0
            )
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        dice, iou = binary_segmentation_metrics(outputs["logits"].detach(), masks)
        dice_values.extend(dice.cpu().tolist())
        iou_values.extend(iou.cpu().tolist())
        total_loss += float(loss.detach()) * batch_size
        total_seg += float(segmentation_loss.detach()) * batch_size
        total_boundary += float(contrastive.detach()) * batch_size
        sample_count += batch_size

    return {
        "loss": total_loss / sample_count,
        "segmentation_loss": total_seg / sample_count,
        "boundary_loss": total_boundary / sample_count,
        "dice": sum(dice_values) / len(dice_values),
        "iou": sum(iou_values) / len(iou_values),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_config = config.get("training", {})
    seed_everything(
        int(config.get("seed", 42)), deterministic=bool(config.get("deterministic", False))
    )
    device = choose_device(config.get("device", "auto"))
    model = build_model(config).to(device)
    boundary_loss = build_contrastive_loss(config).to(device)
    parameters = list(model.parameters()) + list(boundary_loss.parameters())
    optimizer = torch.optim.Adam(
        parameters,
        lr=float(train_config.get("learning_rate", 0.001)),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(train_config.get("lr_factor", 0.5)),
        patience=int(train_config.get("lr_patience", 15)),
        min_lr=float(train_config.get("min_lr", 1e-6)),
    )
    amp_enabled = bool(train_config.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    train_dataset = build_dataset(config, "train", augment=True)
    val_dataset = build_dataset(config, "val", augment=False)
    loader_options = {
        "batch_size": int(train_config.get("batch_size", 8)),
        "num_workers": int(train_config.get("num_workers", 4)),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or train_config.get("output_dir", f"runs/{timestamp}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json({key: value for key, value in config.items() if key != "_config_path"}, output_dir / "config.json")

    start_epoch = 0
    best_dice = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        boundary_loss.load_state_dict(checkpoint["contrastive_loss"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint.get("best_dice", -1.0))

    history_path = output_dir / "history.jsonl"
    epochs = int(train_config.get("epochs", 120))
    for epoch in range(start_epoch, epochs):
        boundary_lambda = contrastive_weight(epoch, config)
        train_metrics = run_epoch(
            model, boundary_loss, train_loader, device, optimizer, boundary_lambda, amp_enabled, scaler
        )
        val_metrics = run_epoch(model, boundary_loss, val_loader, device, None, 0.0, amp_enabled)
        scheduler.step(val_metrics["dice"])
        record = {
            "epoch": epoch + 1,
            "boundary_weight": boundary_lambda,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record, indent=2))

        checkpoint = {
            "epoch": epoch,
            "best_dice": max(best_dice, val_metrics["dice"]),
            "model": model.state_dict(),
            "contrastive_loss": boundary_loss.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": {key: value for key, value in config.items() if key != "_config_path"},
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            checkpoint["best_dice"] = best_dice
            torch.save(checkpoint, output_dir / "best.pt")


if __name__ == "__main__":
    main()
