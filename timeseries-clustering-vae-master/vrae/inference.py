import torch
import numpy as np


def inference(args, dataloader):
    model = torch.load(args.model_path).to(args.device)
    model.eval()

    with torch.no_grad():
        label_arrays, z_arrays = [], []
        for i, batch in enumerate(dataloader):
            if args.dataset == "protocol":
                inputs, labels, _ = map(lambda x: x.to(args.device), batch)
                x = inputs.permute(1, 0)
            elif args.dataset == "ECG":
                inputs, labels = map(lambda x: x.to(args.device), batch)
                x = inputs.permute(1, 0, 2)
            else:
                raise ValueError("Dataset should be chosen in protocol/ECG.")

            if x.size(dim=1) != args.eval_batch_size:
                break

            _, z = model(x)

            label_arrays.append(labels.cpu().numpy())
            z_arrays.append(z.cpu().numpy())

        labels = np.concatenate(label_arrays, axis=0)
        z = np.concatenate(z_arrays, axis=0)

    return z, labels
