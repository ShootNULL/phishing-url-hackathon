# cnn по байтам url, oof + тест -> cache/cnn_url.npz
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

SEED, N_FOLDS, MAXLEN, EPOCHS, BS = 42, 5, 256, 8, 256
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def encode(urls):
    X = np.zeros((len(urls), MAXLEN), dtype=np.int64)
    for i, u in enumerate(urls):
        b = u.encode("utf-8", "replace")[:MAXLEN]
        X[i, :len(b)] = np.frombuffer(b, dtype=np.uint8).astype(np.int64) + 1  # 0 = pad
    return torch.from_numpy(X)


class Net(nn.Module):
    def __init__(self, emb=48, ch=192):
        super().__init__()
        self.emb = nn.Embedding(257, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, ch, k, padding=k // 2) for k in (2, 3, 4, 5, 7)])
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(ch * 5 * 2, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        mask = (x > 0).unsqueeze(1).float()
        h = self.emb(x).transpose(1, 2)
        pooled = []
        for c in self.convs:
            z = F.gelu(c(h))[:, :, : x.shape[1]]
            pooled.append((z - (1 - mask) * 1e4).max(-1).values)
            pooled.append((z * mask).sum(-1) / mask.sum(-1).clamp(min=1))
        return self.fc2(self.drop(F.gelu(self.fc1(self.drop(torch.cat(pooled, 1)))))).squeeze(-1)


def predict(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            out.append(torch.sigmoid(model(X[i:i + 1024].to(DEV))).cpu().numpy())
    return np.concatenate(out)


def train_fold(Xtr, ytr, seed):
    seed_all(seed)
    model = Net().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = EPOCHS * ((len(Xtr) + BS - 1) // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    yt = torch.tensor(ytr, dtype=torch.float32)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BS):
            idx = perm[i:i + BS]
            loss = F.binary_cross_entropy_with_logits(model(Xtr[idx].to(DEV)), yt[idx].to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model


if __name__ == "__main__":
    tr = pd.read_csv("train_6dd2c989.csv")
    te = pd.read_csv("test_5013633d.csv")
    y = tr.result.values.astype(int)
    X, Xte = encode(tr.url.tolist()), encode(te.url.tolist())
    oof, test = np.zeros(len(tr)), np.zeros(len(te))
    for k, (trn, val) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(tr, y)):
        m = train_fold(X[trn], y[trn], SEED + k)
        oof[val] = predict(m, X[val])
        test += predict(m, Xte) / N_FOLDS
        print(f"fold {k}: F1={f1_score(y[val], oof[val] > .5):.4f}", flush=True)
    print(f"CNN oof F1@.5={f1_score(y, oof > .5):.4f}")
    np.savez("cache/cnn_url.npz", oof=oof, test=test)
