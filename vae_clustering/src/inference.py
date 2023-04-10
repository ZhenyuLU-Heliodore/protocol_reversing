import torch
import numpy as np

from .trainer import eval_global_score


def inference(args, dataloader):
    model = torch.load(args.model_path).to(args.device)
    model.eval()

    with torch.no_grad():
        try:
            clus_centers = model.clus_centers.detach().cpu().numpy()
        except AttributeError:  # Attribute name of clustering vectors got modified
            clus_centers = model.clustering_center.detach().cpu().numpy()

        y_arrays, label_arrays, z_arrays = [], [], []
        max_num_batches = args.max_num_batches

        for i, batch in enumerate(dataloader):
            if max_num_batches is not None and i >= max_num_batches:
                print("Number of batches capped")
                break

            inputs, labels, padding_mask = map(lambda x: x.to(args.device), batch)
            _, _, _, _, z, y, _ = model(token_seq=inputs, key_padding_mask=padding_mask)

            y_arrays.append(y.cpu().numpy())
            label_arrays.append(labels.cpu().numpy())
            z_arrays.append(z.cpu().numpy())

        score = eval_global_score(y_arrays, label_arrays, args.eval_metric)

    predicts = np.concatenate(y_arrays, axis=0)
    labels = np.concatenate(label_arrays, axis=0)
    z = np.concatenate(z_arrays, axis=0)

    return score, predicts, labels, clus_centers, z
