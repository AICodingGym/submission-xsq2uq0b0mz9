"""Train a masked-pressure bidirectional GRU on the supplied Gym data."""
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset


SEED = 20261116
MAX_EPOCHS = 20
BATCH_SIZE = 512
HIDDEN_SIZE = 128
LAYERS = 2
CHECKPOINT_DIR = Path("checkpoints/bigru")
SEQ_LEN = 80

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.backends.cudnn.benchmark = True


def build_sequences(path: str, labeled: bool):
    columns = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
    if labeled:
        columns.append("pressure")
    frame = pd.read_csv(path, usecols=columns)
    frame.sort_values(["breath_id", "time_step", "id"], kind="mergesort", inplace=True)
    nrows = len(frame)
    if nrows % SEQ_LEN:
        raise ValueError(f"{path} has {nrows} rows, not a multiple of {SEQ_LEN}")
    nbreaths = nrows // SEQ_LEN
    breath_ids = frame["breath_id"].to_numpy().reshape(nbreaths, SEQ_LEN)
    if not np.all(breath_ids == breath_ids[:, :1]):
        raise ValueError(f"{path} contains a breath that is not exactly {SEQ_LEN} rows")

    u = frame["u_in"].to_numpy(dtype=np.float32).reshape(nbreaths, SEQ_LEN)
    lag = np.zeros_like(u)
    lag[:, 1:] = u[:, :-1]
    diff = u - lag
    cum = np.cumsum(u, axis=1, dtype=np.float32)
    x = np.empty((nbreaths, SEQ_LEN, 8), dtype=np.float32)
    for j, col in enumerate(["R", "C", "time_step", "u_in", "u_out"]):
        x[:, :, j] = frame[col].to_numpy(dtype=np.float32).reshape(nbreaths, SEQ_LEN)
    x[:, :, 5] = lag
    x[:, :, 6] = diff
    x[:, :, 7] = cum
    ids = frame["id"].to_numpy().reshape(nbreaths, SEQ_LEN)
    mask = x[:, :, 4] == 0
    y = None
    if labeled:
        y = frame["pressure"].to_numpy(dtype=np.float32).reshape(nbreaths, SEQ_LEN)
    del frame
    return breath_ids[:, 0].copy(), ids, x, mask, y


class BiGRU(nn.Module):
    def __init__(self, input_size: int, hidden: int, layers: int):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.15 if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        z, _ = self.rnn(x)
        return self.head(z).squeeze(-1)


def evaluate(model, loader, device):
    model.eval()
    abs_sum = 0.0
    n_scored = 0
    with torch.no_grad():
        for xb, yb, mb in loader:
            xb = xb.to(device, non_blocking=True)
            pred = model(xb)
            mb = mb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            abs_sum += torch.abs(pred - yb)[mb].sum().item()
            n_scored += int(mb.sum().item())
    return abs_sum / n_scored


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to silently train on CPU")
    device = torch.device("cuda")
    print(f"Using {torch.cuda.get_device_name(0)}; torch {torch.__version__}", flush=True)

    train_breath, train_row_ids, x, mask, y = build_sequences("data/train.csv", True)
    valid_ids = pd.read_csv("valid_breath_ids.csv")["breath_id"].to_numpy()
    is_valid = np.isin(train_breath, valid_ids)
    if int(is_valid.sum()) != len(valid_ids):
        raise ValueError("Validation breath list does not match the training data")
    train_ix = np.flatnonzero(~is_valid)
    valid_ix = np.flatnonzero(is_valid)

    # Normalize using only the training breaths.
    mean = x[train_ix].reshape(-1, x.shape[-1]).mean(axis=0)
    std = x[train_ix].reshape(-1, x.shape[-1]).std(axis=0)
    std[std < 1e-6] = 1.0
    x -= mean.astype(np.float32)
    x /= std.astype(np.float32)

    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    mt = torch.from_numpy(mask)
    dataset = TensorDataset(xt, yt, mt)
    train_loader = DataLoader(
        Subset(dataset, train_ix.tolist()), batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    valid_loader = DataLoader(
        Subset(dataset, valid_ix.tolist()), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )

    model = BiGRU(x.shape[-1], HIDDEN_SIZE, LAYERS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_mae = float("inf")
    best_epoch = 0
    patience = 5
    since_best = 0
    started = time.monotonic()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb, mb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = torch.abs(pred - yb)[mb].mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))

        val_mae = evaluate(model, valid_loader, device)
        elapsed = time.monotonic() - started
        state = {
            "epoch": epoch,
            "validation_mae": val_mae,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "seed": SEED,
        }
        checkpoint = CHECKPOINT_DIR / f"bigru_epoch_{epoch:02d}.pt"
        torch.save(state, checkpoint)
        if val_mae < best_mae:
            best_mae, best_epoch, since_best = val_mae, epoch, 0
            torch.save(state, CHECKPOINT_DIR / "bigru_best.pt")
        else:
            since_best += 1
        print(
            f"epoch {epoch:02d}/{MAX_EPOCHS} train_l1={np.mean(losses):.5f} "
            f"validation_mae(u_out==0)={val_mae:.6f} best={best_mae:.6f} "
            f"elapsed={elapsed/60:.1f}m checkpoint={checkpoint}",
            flush=True,
        )
        if since_best >= patience and epoch >= 8:
            print(f"Early stopping; best epoch was {best_epoch}.", flush=True)
            break
        if elapsed >= 22 * 60:
            print("Stopping at the 22-minute safety limit.", flush=True)
            break

    # Inference uses the best complete-breath holdout checkpoint.
    best = torch.load(CHECKPOINT_DIR / "bigru_best.pt", map_location=device)
    model.load_state_dict(best["model_state"])
    del dataset, xt, yt, mt, x, y, mask
    test_breath, test_row_ids, tx, tmask, _ = build_sequences("data/test.csv", False)
    tx -= best["feature_mean"].astype(np.float32)
    tx /= best["feature_std"].astype(np.float32)
    test_tensor = torch.from_numpy(tx)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    predictions = []
    with torch.no_grad():
        for (xb,) in test_loader:
            predictions.append(model(xb.to(device, non_blocking=True)).cpu().numpy())
    pressure = np.concatenate(predictions).reshape(-1)
    submission = pd.DataFrame({"id": test_row_ids.reshape(-1), "pressure": pressure})
    submission.sort_values("id", kind="mergesort", inplace=True)
    sample = pd.read_csv("data/sample_submission.csv")
    if list(submission.columns) != list(sample.columns):
        raise ValueError(f"Wrong columns: {submission.columns.tolist()}")
    if len(submission) != len(sample) or not np.array_equal(
        submission["id"].to_numpy(), sample["id"].to_numpy()
    ):
        raise ValueError("Submission ids or row count do not match sample_submission.csv")
    if not np.isfinite(submission["pressure"].to_numpy()).all():
        raise ValueError("Submission contains non-finite predictions")
    submission.to_csv("submission_v2.csv", index=False)
    print(
        f"Wrote submission_v2.csv: {len(submission)} rows, columns id,pressure; "
        f"best epoch {best_epoch}, validation MAE {best_mae:.6f}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
