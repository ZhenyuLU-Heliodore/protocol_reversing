import torch
import torch.optim as optim
import numpy as np

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from .model import TransformerVAE
from .criterion import vae_clustering_loss, cluster_eval_metric


class Trainer:
    def __init__(self, args, train_loader, validation_loader):
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.result_path = args.txt_result_prefix + ".txt"
        self.eval_metric = args.eval_metric
        self.writer = SummaryWriter(args.tb_result_dir)
        self.start_epoch = args.start_epoch
        self.device = args.device
        self.triplet_margin = args.triplet_margin
        self.label_smooth = args.label_smooth
        self.recon_loss_type = args.recon_loss_type
        self.loss3_type = args.loss3_type
        self.dataset = args.dataset

        if self.start_epoch == 0:
            self.model = TransformerVAE(
                num_classes=args.num_classes,
                num_tokens=args.num_tokens,
                pad_id=args.pad_id,
                seq_len=args.seq_len,
                num_heads=args.num_heads,
                encoder_dim=args.encoder_dim,
                decoder_dim=args.decoder_dim,
                num_encoder_layers=args.num_encoder_layers,
                num_decoder_layers=args.num_decoder_layers,
                dim_ffn=args.dim_ffn,
                dropout=args.dropout,
                activation=args.activation,
                layer_norm_eps=args.layer_norm_eps,
                encoder_norm=args.encoder_norm,
                decoder_norm=args.decoder_norm,
                dataset=args.dataset
            )
            self.optimizer = optim.Adam(self.model.parameters(), args.lr)

        else:
            self.model = torch.load(args.start_model_path)
            self.optimizer = torch.load(args.start_optimizer_path)

        self.model.to(self.device)

    def train(self, epoch):
        losses, recon_losses, kl_losses, losses_3 = 0., 0., 0., 0.
        ep1_losses, ep1_recon_losses, ep1_kl_losses, ep1_losses_3 = 0., 0., 0., 0.
        n_batches, n_samples = len(self.train_loader), len(self.train_loader.dataset)

        self.model.train()
        for i, batch in enumerate(self.train_loader):
            inputs, labels, padding_mask = map(lambda x: x.to(self.device), batch)
            x_recon, z_mean, z_log_var, z_prior_mean, z, y, y_one_hot = self.model(
                token_seq=inputs, key_padding_mask=padding_mask,
            )

            loss, recon_loss, kl_loss, loss_3 = vae_clustering_loss(
                x_recon, inputs, z_log_var, z_prior_mean, z,
                y, y_one_hot, self.model.clus_centers,
                padding_mask,
                lambda1=self.lambda1, lambda2=self.lambda2,
                label_smooth=self.label_smooth,
                recon_loss_type=self.recon_loss_type,
                loss3_type=self.loss3_type,
                triplet_margin=self.triplet_margin,
            )

            losses += loss
            recon_losses += recon_loss
            kl_losses += kl_loss
            losses_3 += loss_3

            # losses per 100 itrs in epoch 1
            if epoch == 1:
                itr = i + 1
                ep1_losses += loss
                ep1_recon_losses += recon_loss
                ep1_kl_losses += kl_loss
                ep1_losses_3 += loss_3
                if itr % 100 == 0:
                    epoch1_result = (
                        "Train Epoch 1 Itr: {}\t>\tLoss: {:.6f}\n recon_loss: {}, kl_loss: {}, {}: {}"
                        .format(
                            itr, ep1_losses / 100, ep1_recon_losses / 100,
                            ep1_kl_losses / 100, self.loss3_type, ep1_losses_3 / 100
                        )
                    )
                    print(epoch1_result)
                    with open(self.result_path, 'a') as file:
                        file.write(epoch1_result + '\n')

                    ep1_losses, ep1_recon_losses, ep1_kl_losses, ep1_losses_3 = 0., 0., 0., 0.

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.writer.add_scalar("training_loss", losses / n_batches, epoch)
        self.writer.add_scalar("recon_loss", recon_losses / n_batches, epoch)
        self.writer.add_scalar("kl_loss", kl_losses / n_batches, epoch)
        self.writer.add_scalar(self.loss3_type, losses_3 / n_batches, epoch)
        self.writer.close()

        training_result = (
            "Train Epoch: {}\t>\tLoss: {:.6f}\n recon_loss: {}, kl_loss: {}, {}: {}"
            .format(
                epoch, losses / n_batches, recon_losses / n_batches,
                       kl_losses / n_batches, self.loss3_type, losses_3 / n_batches
            )
        )
        print(training_result)
        with open(self.result_path, 'a') as file:
            file.write(training_result + '\n')

    def validate(self, epoch):
        losses = 0
        n_batches = len(self.validation_loader)
        y_arrays, label_arrays = [], []

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.validation_loader):
                inputs, labels, padding_mask = map(lambda x: x.to(self.device), batch)

                x_recon, z_mean, z_log_var, z_prior_mean, z, y, y_one_hot = self.model(
                    token_seq=inputs, key_padding_mask=padding_mask,
                )

                loss, _, _, _ = vae_clustering_loss(
                    x_recon, inputs, z_log_var, z_prior_mean, z,
                    y, y_one_hot, self.model.clus_centers,
                    padding_mask,
                    lambda1=self.lambda1, lambda2=self.lambda2,
                    label_smooth=self.label_smooth,
                    recon_loss_type=self.recon_loss_type,
                    loss3_type=self.loss3_type,
                    triplet_margin=self.triplet_margin,
                )
                # # another method to choose predicted y
                # y = z_prior_mean.square().mean(dim=-1).argmin(dim=1)

                y_arrays.append(y.cpu().numpy())
                label_arrays.append(labels.cpu().numpy())

                losses += loss

        score = eval_global_score(y_arrays, label_arrays, self.eval_metric)

        self.writer.add_scalar("validation_loss", losses / n_batches, epoch)
        self.writer.add_scalar("score", score, epoch)
        self.writer.close()

        validation_result = (
            "Validation Epoch: {}\t>\tLoss: {:.6f}\tScore: {:.6f}"
            .format(epoch, losses / n_batches, score)
        )
        print(validation_result)
        with open(self.result_path, 'a') as file:
            file.write(validation_result + '\n\n')

    def save(self, epoch, model_prefix, optimizer_prefix):
        if model_prefix is not None:
            path1 = Path(model_prefix + "_epoch_" + str(epoch) + ".pt")
            if not path1.parent.exists():
                path1.parent.mkdir(parents=True)
            torch.save(self.model, path1)

        if optimizer_prefix is not None:
            path2 = Path(optimizer_prefix + "_epoch_" + str(epoch) + ".pt")
            if not path2.parent.exists():
                path2.parent.mkdir(parents=True)
            torch.save(self.optimizer, path2)


def eval_global_score(
        predicts: list, labels: list, eval_metric: str
):
    predicts_all = np.concatenate(predicts, axis=0)
    # If input predicted probs instead of labels
    if predicts_all.shape.__len__() > 1:
        predicts_all = np.argmax(predicts_all, axis=-1)
    labels_all = np.concatenate(labels, axis=0)
    score = cluster_eval_metric(predicts_all, labels_all, metric=eval_metric)

    return score
