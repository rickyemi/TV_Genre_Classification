"""
Deep multilayer perceptron for multi-class tabular data (PyTorch), wrapped as a
scikit-learn classifier so it can be tuned with RandomizedSearchCV, sit inside
a Pipeline, be permutation-tested and saved with joblib like the other models.

Architecture: [Linear -> BatchNorm -> SiLU -> Dropout] x n_layers -> Linear(3)
Training:    AdamW, cross-entropy (optionally class-weighted for the genre
             imbalance), mini-batches, cosine learning-rate decay, early stopping
             on a stratified validation slice of the training data with the best
             weights restored.
"Optimised" = depth, width, dropout, learning rate, weight decay, batch size and
class weighting are all searched by RandomizedSearchCV (see train_model.py).
"""
import copy

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn


class _MLP(nn.Module):
    def __init__(self, n_in, n_out, width, n_layers, dropout):
        super().__init__()
        layers, d = [], n_in
        for _ in range(n_layers):
            layers += [nn.Linear(d, width), nn.BatchNorm1d(width), nn.SiLU(), nn.Dropout(dropout)]
            d = width
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, width=64, n_layers=2, dropout=0.2, lr=1e-3, weight_decay=1e-4, batch_size=64,
                 class_weight="balanced", max_epochs=300, patience=30, val_fraction=0.15, random_state=42):
        self.width = width
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state); np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=self.val_fraction, stratify=y,
                                              random_state=self.random_state)
        w = (compute_class_weight("balanced", classes=self.classes_, y=ytr) if self.class_weight == "balanced"
             else np.ones(len(self.classes_)))
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
        self.model_ = _MLP(X.shape[1], len(self.classes_), int(self.width), int(self.n_layers), float(self.dropout))
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
        Xv, yv = torch.from_numpy(Xva), torch.from_numpy(yva)
        gen = torch.Generator().manual_seed(self.random_state)
        best, best_state, bad = np.inf, None, 0
        self.history_ = {"train_loss": [], "val_loss": [], "val_acc": []}
        for epoch in range(self.max_epochs):
            self.model_.train()
            perm = torch.randperm(len(Xt), generator=gen)
            tot = 0.0
            for i in range(0, len(perm), int(self.batch_size)):
                idx = perm[i:i + int(self.batch_size)]
                if len(idx) < 2:          # BatchNorm needs >= 2 rows
                    continue
                opt.zero_grad()
                loss = loss_fn(self.model_(Xt[idx]), yt[idx])
                loss.backward(); opt.step()
                tot += loss.item() * len(idx)
            sched.step()
            self.model_.eval()
            with torch.no_grad():
                out = self.model_(Xv)
                vl = loss_fn(out, yv).item()
                va = (out.argmax(1) == yv).float().mean().item()
            self.history_["train_loss"].append(tot / len(Xt))
            self.history_["val_loss"].append(vl); self.history_["val_acc"].append(va)
            if vl < best - 1e-4:
                best, bad, best_state, self.best_epoch_ = vl, 0, copy.deepcopy(self.model_.state_dict()), epoch
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict_proba(self, X):
        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.model_.eval()
        with torch.no_grad():
            return torch.softmax(self.model_(X), dim=1).numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
