import argparse
import torch
import matplotlib as mpl

from src import clus_visualization, kmeans_visualization, inference
from torch.utils.data import DataLoader


def evaluate(args):
    print(args)

    dataset = torch.load(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=True)

    _, _, labels, clus_centers, z = inference(args, dataloader)

    if args.visual_type is None:
        pass
    elif args.visual_type == "vanilla":
        clus_visualization(args, labels, clus_centers, z)
    elif args.visual_type == "kmeans":
        kmeans_visualization(args, clus_centers, z)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Evaluation args
    parser.add_argument("--max_num_batches", default=None, type=int)
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--eval_metric", default="rand_score", type=str)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    # Path args
    parser.add_argument("--dataset_path", default="./dataset/eval_inside.pt", type=str)
    parser.add_argument("--model_path", default="./models/temp", type=str)
    parser.add_argument("--figure_prefix", default="./figures/temp", type=str)

    # Plotting args
    parser.add_argument("--visual_type", default=None, type=str, help="Chosen in None/vanilla/kmeans")
    parser.add_argument("--max_num_points", default=1500, type=int)

    # Fitting args
    parser.add_argument("--tsne_perplexity", default=30., type=float)
    parser.add_argument("--kmeans_max_itr", default=300, type=int)

    args = parser.parse_args()

    args.visual_type = "vanilla"
    args.dataset_path = "./dataset_ECG/valid_set.pt"
    args.model_path = "./models/4.9/ECG_config2/VAE_epoch_50.pt"
    args.figure_prefix = "./figures/4.9/ECG_config2_epoch50"
    args.device = "cuda:1"
    args.perplexity = 50
    args.num_classes = 5

    evaluate(args)
