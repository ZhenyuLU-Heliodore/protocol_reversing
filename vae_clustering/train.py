import argparse
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from src import Trainer
from torch.utils.data import TensorDataset


def train(args):
    print(args)
    path = Path(args.txt_result_prefix + ".txt")
    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    with open(path, 'w') as file:
        file.write('args:\n' + str(args) + '\n\n')

    training_dataset = torch.load(args.train_set_path)
    training_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)

    if args.valid_set_path is not None:
        validation_dataset = torch.load(args.valid_set_path)
        validation_loader = DataLoader(validation_dataset, batch_size=args.valid_batch_size, shuffle=True)
    else:
        validation_loader = None

    trainer = Trainer(args, training_loader, validation_loader)

    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        trainer.train(epoch)
        if validation_loader is not None:
            trainer.validate(epoch)

        if epoch % args.model_saving_step == 0:
            trainer.save(epoch, args.model_prefix, args.optimizer_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model args
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument("--encoder_dim", default=256, type=int)
    parser.add_argument("--decoder_dim", default=256, type=int)
    parser.add_argument("--num_encoder_layers", default=4, type=int)
    parser.add_argument("--num_decoder_layers", default=4, type=int)
    parser.add_argument("--dim_ffn", default=512, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--activation", default="relu", type=str)
    parser.add_argument("--layer_norm_eps", default=1e-5, type=float)
    parser.add_argument("--encoder_norm", default=None, type=torch.nn.Module)
    parser.add_argument("--decoder_norm", default=None, type=torch.nn.Module)

    # Loss function args
    parser.add_argument("--lambda1", default=1., type=float)
    parser.add_argument("--lambda2", default=3., type=float)
    parser.add_argument("--label_smooth", default=0.5, type=float)
    parser.add_argument("--recon_loss_type", default="cross_entropy", type=str, help="chosen in cross_entropy/mse")
    parser.add_argument("--loss3_type", default="triplet", type=str, help="chosen in category/triplet")
    parser.add_argument("--triplet_margin", default=1., type=float)

    # Optimizer args
    parser.add_argument("--lr", default=1e-4, type=float)

    # Dataset args
    parser.add_argument("--dataset", default="protocol", type=str, help="chosen in protocol/ECG")
    parser.add_argument("--train_set_path", default="../dataset/training_set.pt", type=str)
    parser.add_argument("--valid_set_path", default="../dataset/validation_set.pt", type=str)
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--num_tokens", default=256, type=int)
    parser.add_argument("--pad_id", default=256, type=int)
    parser.add_argument("--seq_len", default=256, type=int)

    # Trainer args
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--valid_batch_size", default=64, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--eval_metric", default="rand_score", type=str)

    # Retraining args
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--start_model_path", default=None, type=str)
    parser.add_argument("--start_optimizer_path", default=None, type=str)

    # Saving args
    parser.add_argument("--txt_result_prefix", default="../logs/txt/temp", type=str)
    parser.add_argument("--tb_result_dir", default="../logs/tb/temp", type=str)
    parser.add_argument("--model_prefix", default=None, type=str)
    parser.add_argument("--model_saving_step", default=10, type=str)
    parser.add_argument("--optimizer_prefix", default=None, type=str)

    args = parser.parse_args()

    train(args)
