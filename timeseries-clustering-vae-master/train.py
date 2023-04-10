import argparse
import torch

from vrae import VRAE


def train(args):
    print(args)
    training_set = torch.load(args.training_set)

    vrae = VRAE(
        args.seq_len,
        args.feat_dim,
        n_epochs=args.epochs,
        device=args.device,
        dataset=args.dataset,
    )

    vrae.fit(
        dataset=training_set,
        save=True,
        model_prefix=args.model_prefix,
        save_model_every = args.save_model_every,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument("--seq_len", default=256, type=int)
    parser.add_argument("--feat_dim", default=256, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)

    # dataset args
    parser.add_argument("--dataset", default="protocol", type=str, help="must be chosen in protocol/ECG")
    parser.add_argument("--training_set", default="./dataset/training_set.pt", type=str)

    # saving args
    parser.add_argument("--model_prefix", default="./models/4.7/vrae", type=str)
    parser.add_argument("--save_model_every", default=10, type=int)

    # training args
    parser.add_argument("--epochs", default=100, type=int)

    args = parser.parse_args()

    args.seq_len = 256
    args.feat_dim = 8
    args.dataset = "protocol"
    args.training_set = "./dataset/training_set.pt"
    args.model_prefix = "./models/4.9/vrae_protocol"

    train(args)

