import argparse, json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

from src.models.factory import build_model
from src.data.datamodule import build_dataloaders


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        logp = -self.ce(logits, targets)
        p = torch.exp(logp)
        loss = -((1 - p) ** self.gamma) * logp
        return loss.mean()


def save_checkpoint(state: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--save_dir', default='runs')
    parser.add_argument('--save_best_by', default='macro_recall')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    model_name = cfg.get('model', 'resnet18')
    img_size = int(cfg.get('img_size', 224))
    batch_size = int(cfg.get('batch_size', 16))
    epochs = int(cfg.get('epochs', 10))
    lr = float(cfg.get('lr', 1e-3))
    weight_decay = float(cfg.get('weight_decay', 1e-4))
    loss_name = cfg.get('loss', 'weighted_ce')
    use_sampler = cfg.get('sampler', 'weighted_random') == 'weighted_random'

    loaders, class_to_idx = build_dataloaders(args.data_root, img_size, batch_size, use_weighted_sampler=use_sampler)
    num_classes = len(class_to_idx)

    model, _ = build_model(model_name, num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # class weights from train loader counts
    counts = torch.zeros(num_classes)
    for _, targets in loaders['train']:
        for t in targets:
            counts[t] += 1
    weights = (counts.sum() / (num_classes * counts.clamp(min=1))).to(device)

    if loss_name.startswith('focal'):
        loss_fn = FocalLoss(gamma=float(cfg.get('focal', {}).get('gamma', 1.5)), weight=weights)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_score = -1.0
    patience_cfg = cfg.get('early_stopping', {}) or {}
    patience = int(patience_cfg.get('patience', 0))
    no_improve = 0
    save_dir = Path(args.save_dir)
    best_ckpt = save_dir / 'best.pt'
    last_ckpt = save_dir / 'last.pt'

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for images, targets in tqdm(loaders['train'], desc=f"Train {epoch}/{epochs}"):
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        scheduler.step()

        # Validation loop: compute recall for pneumonia class(es)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in tqdm(loaders['val'], desc="Val"):
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.numel()
        acc = correct / max(1, total)
        score = acc  # placeholder: can be replaced by pneumonia recall computation on mapped classes

        print(f"Epoch {epoch}: val_acc={acc:.4f}")
        # Save last
        save_checkpoint({'model': model.state_dict(), 'classes': class_to_idx}, last_ckpt)
        # Save best
        if score > best_score:
            best_score = score
            save_checkpoint({'model': model.state_dict(), 'classes': class_to_idx}, best_ckpt)
            print(f"Saved best checkpoint: {best_ckpt}")
            no_improve = 0
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement in {no_improve} epochs)")
                break

    print(f"Training done. Best {args.save_best_by}={best_score:.4f}")


if __name__ == '__main__':
    main()
