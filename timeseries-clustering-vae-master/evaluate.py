import torch
import argparse

from torch.utils.data import DataLoader
from vrae import inference
from vrae import plot_clustering


def evaluate(args):
    print(args)

    dataset = torch.load(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=True)

    z, labels = inference(args, dataloader)
    plot_clustering(z, labels, folder_name=args.fig_folder_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument("--dataset", default="protocol", type=str)
    parser.add_argument("--dataset_path", default="./dataset/eval_inside.pt", type=str)

    # eval args
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)

    # model args
    parser.add_argument("--model_path", default="./models/4.7/vrae_epoch_9.pt", type=str)

    # plotting args
    parser.add_argument("--fig_folder_name", default="./figures/temp", type=str)

    args = parser.parse_args()

    args.dataset = "protocol"
    args.dataset_path = "./dataset/eval_inside.pt"
    args.model_path = "./models/4.9/vrae_protocol_epoch_30.pt"
    args.fig_folder_name = "./figures/protocol_dim=8"

    evaluate(args)


