from __future__ import annotations

raise SystemExit(
    "scripts/train_head_count.py is obsolete: the density-map HeadCountNet "
    "it trained was replaced by APGCC, a pretrained point detector. There is "
    "no density head left here to train.\n"
    "To fine-tune APGCC on your own footage, use apgcc_finetune.py in the "
    "crowd-model-eval repo instead."
)

"""
Train the density-map head counter on hand-labelled patches.

    python scripts/train_head_count.py --patches data/nashik/patches
    python scripts/train_head_count.py --patches data/nashik/patches \
        --epochs 200 --batch 8 --crop 512

Reports MAE in heads per patch — the number you actually care about — rather
than only the pixel loss, which can look converged while the count is wrong.

The best checkpoint is chosen on validation MAE, not on the last epoch: with
a few hundred patches the model overfits well before the schedule ends, and
saving the final weights would ship the overfitted ones.

Expect to need a few hundred labelled patches. Fewer will train, but the
result will be specific to your footage in ways that are hard to see — which
is fine if that is the only camera it will ever run on, and misleading if it
is not.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.head_count.dataset import Patches
from models.head_count.model import HeadCountNet, CountLoss
from pipeline.device import resolve_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("train_head_count")


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    """Returns (MAE, RMSE) in heads per patch."""
    model.eval()
    errs = []
    for img, target, n_true in loader:
        img = img.to(device, non_blocking=True)
        pred = model(img).sum(dim=(1, 2, 3)).cpu().numpy()
        errs.append(pred - n_true.numpy())
    if not errs:
        return float("nan"), float("nan")
    e = np.concatenate(errs)
    return float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patches", required=True, help="dir with images/, labels/, *.list")
    ap.add_argument("--out", default="model_weights/head_count.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone-lr", type=float, default=1e-5,
                    help="Lower LR for the pretrained frontend; the default "
                         "head LR would destroy its features in a few steps.")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    train_ds = Patches(args.patches, "train", crop=args.crop, augment=True)
    val_ds = Patches(args.patches, "val", crop=args.crop, augment=False)

    log.info("train %d patches (%d heads, %d hard negatives)",
             len(train_ds), train_ds.total_heads(), train_ds.negatives)
    log.info("val   %d patches (%d heads, %d hard negatives)",
             len(val_ds), val_ds.total_heads(), val_ds.negatives)
    if train_ds.negatives == 0:
        log.warning(
            "No hard negatives in the training split.  Without patches "
            "labelled as containing zero heads, the model has never been "
            "told what is NOT a crowd and will fire on roofs, tarpaulins and "
            "foliage.  Label some with `c` in scripts/annotate_heads.py."
        )

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, drop_last=len(train_ds) > args.batch)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers)

    model = HeadCountNet(pretrained=True).to(device)
    criterion = CountLoss()
    backbone = list(model.stem.parameters()) + list(model.layer1.parameters()) \
        + list(model.layer2.parameters()) + list(model.layer3.parameters())
    head = list(model.backend.parameters()) + list(model.head.parameters())
    opt = torch.optim.AdamW(
        [{"params": backbone, "lr": args.backbone_lr},
         {"params": head, "lr": args.lr}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    best = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = pix = cnt = 0.0
        for img, target, _n in train_dl:
            img = img.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            loss, p, c = criterion(model(img), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss); pix += float(p); cnt += float(c)
        sched.step()
        n = max(1, len(train_dl))

        if epoch % 5 == 0 or epoch == args.epochs or epoch == 1:
            mae, rmse = evaluate(model, val_dl, device)
            flag = ""
            if mae < best:
                best = mae
                torch.save({"model": model.state_dict(), "val_mae": mae,
                            "epoch": epoch}, args.out)
                flag = "  <- saved"
            log.info("epoch %3d/%d  loss %.4f (pix %.4f count %.2f)  "
                     "val MAE %.2f  RMSE %.2f%s",
                     epoch, args.epochs, tot / n, pix / n, cnt / n, mae, rmse, flag)

    log.info("done in %.1f min; best val MAE %.2f heads/patch -> %s",
             (time.time() - t0) / 60, best, args.out)
    if best == float("inf"):
        log.error("No checkpoint was saved — validation never produced a "
                  "finite MAE.  Check that val.list is non-empty.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
