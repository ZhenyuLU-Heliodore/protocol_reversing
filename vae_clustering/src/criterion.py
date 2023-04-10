import torch

from torch.nn import CrossEntropyLoss, TripletMarginLoss, MSELoss
from sklearn.metrics.cluster import rand_score, adjusted_rand_score


def vae_clustering_loss(
        x_recon, x, z_log_var, z_prior_mean, z,
        y, y_one_hot, clus_center,
        padding_mask,
        lambda1=1., lambda2=3.,
        label_smooth=0.5,
        recon_loss_type="cross_entropy",
        loss3_type="triplet",
        triplet_margin=1.,
):
    if recon_loss_type == "cross_entropy":
        recon_loss = recon_loss_func(x_recon, x, padding_mask, label_smooth)
    elif recon_loss_type == "mse":
        mse_func = MSELoss()
        recon_loss = mse_func(x, x_recon)
    else:
        raise ValueError("Illegal input of recon loss type.")

    kl_loss = kl_loss_func(y, z_log_var, z_prior_mean)

    if loss3_type == "triplet":
        loss_3 = triplet_loss_func(z, y_one_hot, clus_center, triplet_margin)
    elif loss3_type == "category":
        loss_3 = cat_loss_func(y)
    else:
        raise ValueError("Illegal input of loss3 type.")

    vae_loss = lambda1 * recon_loss + lambda2 * kl_loss + loss_3

    return vae_loss, recon_loss, kl_loss, loss_3


def recon_loss_func(x_recon, x, padding_mask, label_smooth):
    batch_size = x_recon.size(0)
    recon_loss = 0
    for i in range(batch_size):
        length = padding_mask.size(-1) - int(torch.sum(padding_mask[i]).item())
        cross_entropy = CrossEntropyLoss(label_smoothing=label_smooth)
        recon_loss += cross_entropy(
            x_recon[i, 0:length, :], x[i, 0:length].to(torch.int64)
        )

    # print((x_recon[i, 0:length, :].argmax(-1) == x[i, 0:length]).to(torch.float).mean())  # recon acc of the last seq
    return recon_loss / batch_size


def kl_loss_func(y, z_log_var, z_prior_mean):
    kl = -0.5 * (z_log_var.unsqueeze(1) - torch.square(z_prior_mean))
    kl = torch.mean(torch.matmul(y.unsqueeze(1), kl))

    return kl


def triplet_loss_func(z, y_one_hot, clus_center, triplet_margin):
    batch_size = z.size(0)
    num_classes = y_one_hot.size(dim=-1)
    dim = z.size(dim=-1)
    trip_loss_func = TripletMarginLoss(margin=triplet_margin, p=2)

    # [batch_size, dim] -> [batch_size, num_classes, dim]
    anchor = torch.stack([z] * num_classes, dim=1)
    # [batch_size, num_classes] * [num_classes, dim] -> [batch_size, dim]; y is one-hot
    positive = torch.mul(
        y_one_hot.unsqueeze(1), clus_center.permute(1, 0)
    ).sum(-1)
    # [batch_size, dim] -> [batch_size, num_classes, dim]
    positive = torch.stack([positive] * num_classes, dim=1)

    # [num_classes, dim] -> [batch_size, num_classes, dim]
    negative = torch.stack([clus_center] * batch_size, dim=0)

    return trip_loss_func(
        anchor.view(-1, dim), positive.view(-1, dim), negative.view(-1, dim)
    )


def cat_loss_func(y):
    return torch.mean(y * torch.log(y + 1e-10))


def cluster_eval_metric(predict, label, metric="rand_score"):
    if metric == "rand_score":
        return rand_score(label, predict)
    if metric == "adjusted_rand_score":
        return adjusted_rand_score(label, predict)
    else:
        raise ValueError("Illegal input of score metric.")
