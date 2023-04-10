import torch
import torch.nn.functional as F
import numpy as np

from src.criterion import cluster_eval_metric
from pathlib import Path
from torch.utils.data import DataLoader


if __name__ == "__main__":
    # model = torch.load("./models/demo_3.17/VAE_epoch_500.pt")
    # model = model.to("cpu")
    #
    # data1 = torch.cat((torch.arange(50), torch.mul(torch.ones(206), 256)), 0)
    # data2 = torch.cat((torch.arange(100), torch.mul(torch.ones(156), 256)), 0)
    # data3 = torch.cat((torch.arange(150), torch.mul(torch.ones(106), 256)), 0)
    # data = torch.stack((data1, data2, data3), dim=0).to(torch.int)
    #
    # mask1 = torch.cat((torch.zeros(50), torch.ones(206)), 0)
    # mask2 = torch.cat((torch.zeros(100), torch.ones(156)), 0)
    # mask3 = torch.cat((torch.zeros(150), torch.ones(106)), 0)
    # mask = torch.stack((mask1, mask2, mask3), dim=0).to(torch.bool)
    #
    # label = torch.tensor([0, 1, 2])
    #
    # print(model(token_seq=data, key_padding_mask=mask)[0].argmax(dim=-1))
    #
    # training_dataset = torch.load("./dataset/training_set.pt")
    # print(training_dataset.__len__())

    device = "cuda:0"
    dataset = torch.load("./dataset/eval_inside.pt")
    model = torch.load("./models/3.29/config6/VAE_epoch_80.pt").to(device)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for _, batch in enumerate(dataloader):
        inputs, labels, padding_mask = map(lambda x: x.to(device), batch)
        _, _, _, _, _, y, y_one_hot = model(token_seq=inputs, key_padding_mask=padding_mask)
        print((y.argmax(dim=-1) == y_one_hot.argmax(dim=-1)).to(torch.float).mean())
