import torch
import torchvision.datasets
import torchvision.transforms
import torchvision.transforms.functional as F
import sys
import os
import datetime
import numpy as np
import torch
from GetParams import get_args
import matplotlib.pyplot as plt
import random




def load_bound_dataset(dataset, batch_size, shuffle=False, start=None, end=None, **kwargs):
    def _bound_dataset(dataset, start, end):
        if start is None:
            start = 0
        if end is None:
            end = len(dataset)
        return torch.utils.data.Subset(dataset, range(start, end))

    dataset = _bound_dataset(dataset, start, end)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle, **kwargs)


def fetch_mnist(root, train=False, transform=None, target_transform=None):
    transform = transform if transform is not None else torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root, train=train, transform=transform, target_transform=target_transform, download=True)
    return dataset


def load_mnist(root, batch_size, train=False, transform=None, target_transform=None, **kwargs):
    dataset = fetch_mnist(root, train, transform, target_transform)
    return load_bound_dataset(dataset, batch_size, **kwargs)


def move_to_type_device(x, y, device):
    print('X:', x.shape)
    print('y:', y.shape)
    x = x.to(torch.get_default_dtype())
    y = y.to(torch.get_default_dtype())
    x, y = x.to(device), y.to(device)
    return x, y


def create_labels(y0):
    labels_dict = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1}
    y0 = torch.stack([torch.tensor(labels_dict[int(cur_y)]) for cur_y in y0])
    return y0


def get_balanced_data(args, data_loader, data_amount, train):
    print('BALANCING DATASET...')
    # get balanced data
    data_amount_per_class = data_amount // 2

    labels_counter = {1: 0, 0: 0}
    x0, y0 = [], []
    got_enough = False
    for bx, by in data_loader:
        by = create_labels(by)
        for i in range(len(bx)):
            if labels_counter[int(by[i])] < data_amount_per_class:
                labels_counter[int(by[i])] += 1
                x0.append(bx[i])
                y0.append(by[i])
            if (labels_counter[0] >= data_amount_per_class) and (labels_counter[1] >= data_amount_per_class):
                got_enough = True
                break
        if got_enough:
            break
    x0, y0 = torch.stack(x0), torch.stack(y0)


    if train == True:
        n_images = y0.shape[0]
        if args.bias_type == 'square':
            n_noisy_images = int(args.noise_perc*n_images)
            if args.noise_mode == 'fixed_squares':
                for ii in range(n_noisy_images):
                    if y0[ii] == 0:
                        x0[ii, :, 0:3, 0:3] = 0.25

                    elif y0[ii] == 1:
                        x0[ii, :, 0:3, 0:3] = 0.75
            elif args.noise_mode == 'non_fixed_squares':
                random.seed(92.48) # Hala madrid
                clue_pos = random.sample(range(1, 25), 2*args.n_squares)
                class1_pos = clue_pos[0:args.n_squares]
                print('class 1 clue positions: ', class1_pos)
                class2_pos = clue_pos[args.n_squares:]
                print('class 2 clue positions: ', class2_pos)
                for ii in range(n_noisy_images):
                    if y0[ii] == 0:
                        pos = class1_pos[ii % args.n_squares]
                        x0[ii, :, pos:pos+3, pos:pos+3] = 0.25

                    elif y0[ii] == 1:
                        pos = class2_pos[ii % args.n_squares]
                        x0[ii, :, pos:pos+3, pos:pos+3] = 0.75

        elif args.bias_type == 'contrast':
            for ii in range(n_images):
                    if y0[ii] == 0:
                        x0[ii] = F.adjust_contrast(x0[ii], args.contrast_factor_1)
                    elif y0[ii] == 1:
                        x0[ii] = F.adjust_contrast(x0[ii], args.contrast_factor_2)   

    return x0, y0


def load_mnist_data(args):
    # Get Train Set
    data_loader = load_mnist(root=args.datasets_dir, batch_size=100, train=True, shuffle=False, start=0, end=50000)
    x0, y0 = get_balanced_data(args, data_loader, args.data_amount, train=True)

    # Get Test Set
    print('LOADING TESTSET')
    assert not args.data_use_test or (args.data_use_test and args.data_test_amount >= 2), f"args.data_use_test={args.data_use_test} but args.data_test_amount={args.data_test_amount}"
    data_loader = load_mnist(root=args.datasets_dir, batch_size=100, train=False, shuffle=False, start=0, end=10000)
    x0_test, y0_test = get_balanced_data(args, data_loader, args.data_test_amount, train=False)

    # move to cuda and double
    x0, y0 = move_to_type_device(x0, y0, args.device)
    x0_test, y0_test = move_to_type_device(x0_test, y0_test, args.device)

    print(f'BALANCE: 0: {y0[y0 == 0].shape[0]}, 1: {y0[y0 == 1].shape[0]}')

    return [(x0, y0)], [(x0_test, y0_test)], None


def get_dataloader(args):
    args.input_dim = 28 * 28
    args.num_classes = 2
    args.output_dim = 1
    args.dataset = 'mnist'

    if args.run_mode == 'reconstruct':
        args.extraction_data_amount = args.extraction_data_amount_per_class * args.num_classes

    # for legacy:
    args.data_amount = args.data_per_class_train * args.num_classes
    args.data_use_test = True
    args.data_test_amount = 1000

    data_loader = load_mnist_data(args)
    return data_loader

  