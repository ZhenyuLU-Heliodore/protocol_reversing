import numpy as np
import torch

from torch.utils.data import TensorDataset
from vrae import open_data


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = open_data('data', ratio_train=0.9)
    num_classes = len(np.unique(y_train))
    base = np.min(y_train)
    if base != 0:
        y_train -= base
    y_val -= base

    x_train = torch.tensor(X_train).to(torch.float)
    y_train = torch.tensor(y_train).to(torch.int)
    x_val = torch.tensor(X_val).to(torch.float)
    y_val = torch.tensor(y_val).to(torch.int)

    # train_set = TensorDataset(x_train, y_train)
    # valid_set = TensorDataset(x_val, y_val)
    #
    # torch.save(train_set, "./dataset_ECG/train_set.pt")
    # torch.save(valid_set, "./dataset_ECG/valid_set.pt")

    x_train_pad = torch.zeros_like(x_train, dtype=torch.bool).squeeze(-1)
    x_val_pad = torch.zeros_like(x_val, dtype=torch.bool).squeeze(-1)

    train_set = TensorDataset(x_train, y_train.squeeze(-1), x_train_pad)
    valid_set = TensorDataset(x_val, y_val.squeeze(-1), x_val_pad)

    torch.save(train_set, "../vae_clustering/dataset_ECG/train_set.pt")
    torch.save(valid_set, "../vae_clustering/dataset_ECG/valid_set.pt")
