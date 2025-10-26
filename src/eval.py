import argparse, json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.utils.metrics import compute_metrics
from src.data.datamodule import build_dataloaders
from src.models.factory import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    parser.add_argument('--report', default='reports/val.json')
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    class_to_idx = ckpt['classes']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    # infer model name from checkpoint path (heuristic) else default
    model_name = 'resnet18'
    model, _ = build_model(model_name, num_classes)
    model.load_state_dict(ckpt['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    loaders, _ = build_dataloaders(args.data_root, img_size=224, batch_size=16, use_weighted_sampler=False)
    loader = loaders[args.split]

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc=f"Eval:{args.split}"):
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(targets.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics, cm = compute_metrics(y_true, y_pred, labels=idx_to_class)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    json.dump({**metrics, 'confusion_matrix': cm.tolist()}, open(args.report, 'w'), indent=2)
    print(f"Saved report to {args.report}")


if __name__ == '__main__':
    main()
