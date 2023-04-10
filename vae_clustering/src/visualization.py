import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from .criterion import cluster_eval_metric


def clus_visualization(args, labels, clus_centers, z):
    # clustering_center: [num_classes, dim]
    # z: [n_samples, dim]
    num_classes = args.num_classes
    max_num_points = args.max_num_points

    # embedding: [num_classes + n_samples, dim]
    embedding = TSNE(
        n_components=2, learning_rate="auto", init="random", perplexity=args.tsne_perplexity
    ).fit_transform(
        np.concatenate((clus_centers, z), axis=0)
    )

    centers = embedding[0:num_classes, :]

    if z.shape[0] > max_num_points:
        points = embedding[num_classes:num_classes+max_num_points, :]
        colors = labels[0: max_num_points].astype(float)
    else:
        points = embedding[num_classes:, :]
        colors = labels.astype(float)

    if num_classes > 12:
        plt.scatter(points[:, 0], points[:, 1], c="b", s=3)
    else:
        plt.scatter(points[:, 0], points[:, 1], c=colors * 10, alpha=0.5, cmap="viridis", s=3)

    plt.scatter(centers[:, 0], centers[:, 1], c="black", s=5)

    path = Path(args.figure_prefix + ".png")
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    plt.savefig(path, dpi=750, bbox_inches='tight')


def kmeans_visualization(args, clus_centers, z, label_truth=None):
    num_classes = args.num_classes

    kmeans = KMeans(n_clusters=num_classes, max_iter=args.kmeans_max_itr).fit(z)
    kmeans_labels = kmeans.labels_
    embedding = TSNE(
        n_components=2, learning_rate="auto", init="random", perplexity=args.tsne_perplexity
    ).fit_transform(
        np.concatenate((clus_centers, z), axis=0)
    )
    centers = embedding[0:num_classes, :]
    points = embedding[num_classes:, :]

    # colors = ["b", "r", "g", "y", "c", "m"]
    # shapes = ["o", "s", "v", "p", "v", "^"]

    # for class_idx in range(num_classes):
    #     for idx in range(len(kmeans_labels)):
    #         if int(kmeans_labels[idx]) == class_idx:
    #             plt.scatter(
    #                 points[idx, 0], points[idx, 1], c=colors[class_idx],
    #             )

    colors = kmeans_labels.astype(float)
    plt.scatter(points[:, 0], points[:, 1], c=colors * 10, alpha=0.5, cmap="viridis", s=3)
    plt.scatter(centers[:, 0], centers[:, 1], c="black", s=5)

    path = Path(args.figure_prefix + ".png")
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    plt.savefig(path, dpi=750, bbox_inches='tight')
