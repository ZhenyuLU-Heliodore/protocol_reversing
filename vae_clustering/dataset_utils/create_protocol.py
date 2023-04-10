import os
import random
import torch
import torch.nn.functional as F

from torch.utils.data import TensorDataset, random_split


def create_dataset(
        protocol_list: list,
        max_sample_num=10000,
        max_seq_len=256,
        pad_id=256,
        src_root_path='.../data/dec_tensor'
):
    x_list, y_list, mask_list = [], [], []

    for protocol_idx, protocol in enumerate(protocol_list):
        tensor_dir = os.listdir(src_root_path+'/'+protocol)
        dir_len = len(tensor_dir)

        # filename starting from 1
        if dir_len > max_sample_num:
            sample_idx_list = random.sample(range(1, dir_len+1), max_sample_num)
        else:
            sample_idx_list = range(1, dir_len+1)

        for sample_idx in sample_idx_list:
            sequence = torch.load(src_root_path+'/'+protocol+'/'+tensor_dir[sample_idx-1])
            seq_len = sequence.size(dim=0)
            if seq_len >= max_seq_len:
                sequence = sequence[0: max_seq_len]
                msk = torch.zeros_like(sequence, dtype=torch.bool)
            else:
                sequence = F.pad(sequence, (0, max_seq_len - seq_len), "constant", pad_id)
                msk = torch.cat(
                    (torch.zeros(seq_len), torch.ones(max_seq_len - seq_len)), dim=0
                ).to(torch.bool)

            # exclude blank tensor
            if seq_len > 1:
                x_list.append(sequence)
                y_list.append(torch.tensor([protocol_idx], dtype=torch.int))
                mask_list.append(msk)

        print(protocol + ' done')

    inputs = torch.stack(x_list, dim=0).to(torch.int)
    labels = torch.stack(y_list, dim=0).squeeze(1)
    padding_mask = torch.stack(mask_list, dim=0)

    print(inputs)
    print(labels)
    print(padding_mask)

    return TensorDataset(inputs, labels, padding_mask)


def dataset_split(dataset, split_prop=None):
    if split_prop is None:
        split_prop = [0.8, 0.1, 0.1]
    data_size = dataset.__len__()
    training_size = int(split_prop[0] * data_size)
    validation_size = int(split_prop[1] * data_size)
    test_size = data_size - training_size - validation_size

    return random_split(
        dataset, [training_size, validation_size, test_size]
    )


def create_eval_dataset(
        protocol_list,
        n=300,
        train_set_path="../dataset/training_set.pt",
        test_set_path="../dataset/test_set.pt"
):
    eval_inside = torch.load(train_set_path)
    eval_outside = torch.load(test_set_path)
    k = len(protocol_list)

    torch.save(sample(eval_inside, n, k), "../dataset/eval_inside.pt")
    torch.save(sample(eval_outside, n, k), "../dataset/eval_outside.pt")


def sample(dataset, n=300, k=4):
    tensor_list = [[]] * k
    l = len(dataset)
    for data in dataset:
        _, label, _ = data
        label_item = int(label.item())

        tensor_list[label_item].append(data)

    list1, list2, list3 = [], [], []

    for i in range(k):
        sample_list = random.sample(range(l), n)
        for j in sample_list:
            data = tensor_list[i][j]
            inputs, labels, padding_mask = data
            list1.append(inputs)
            list2.append(labels)
            list3.append(padding_mask)

    inputs = torch.stack(list1, dim=0)
    labels = torch.stack(list2, dim=0)
    padding_mask = torch.stack(list3, dim=0)

    print(inputs.shape)
    print(labels.to(torch.float).mean())
    print(padding_mask.shape)

    return TensorDataset(inputs, labels, padding_mask)


if __name__ == '__main__':
    protocol_list = [
        'ISCX_Botnet',  # 544k
        'SMIA',  # 47k
        'dnp3',  # 26k
        # 'dhcp',  # 1.1k
        # 'dns',  # 2.2k
        'modbus',  # 13k
        # 'nbns',  # 1.1k
        # 'ntp',  # 0.1k
        # 'smb',  # 1.1k
        # 'tftp',  # 0.5k
        ]
    # dataset = create_dataset(protocol_list=protocol_list)
    # training_set, validation_set, test_set = dataset_split(dataset)
    #
    # torch.save(training_set, './dataset/training_set.pt')
    # torch.save(validation_set, './dataset/validation_set.pt')
    # torch.save(test_set, './dataset/test_set.pt')

    create_eval_dataset(protocol_list)