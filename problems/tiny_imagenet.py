import torch
import torchvision.datasets as datasets
import torchvision.transforms


def load_bound_dataset(dataset, batch_size, shuffle=False, start=None, end=None, **kwargs):
    def _bound_dataset(dataset, start, end):
        if start is None:
            start = 0
        if end is None:
            end = len(dataset)
        return torch.utils.data.Subset(dataset, range(start, end))

    dataset = _bound_dataset(dataset, start, end)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle, **kwargs)


def fetch_tiny_imagenet(root, train, transform=None, target_transform=None):
    transform = transform if transform is not None else torchvision.transforms.ToTensor()
    if train:
        root+= "/train"
    else:
        root+= "/val"
    # Load the Tiny ImageNet dataset
    dataset = datasets.ImageFolder(root=root, transform=transform, target_transform=target_transform,)
    return dataset


def load_tiny_imagenet(root, batch_size, train=False, transform=None, target_transform=None, **kwargs):
    dataset = fetch_tiny_imagenet(root, train, transform, target_transform)
    return load_bound_dataset(dataset, batch_size, **kwargs)


def move_to_type_device(x, y, device):
    print('X:', x.shape)
    print('y:', y.shape)
    x = x.to(torch.get_default_dtype())
    y = y.to(torch.get_default_dtype())
    x, y = x.to(device), y.to(device)
    return x, y


def create_labels(x0, y0):
    labels_dict = {0: 0, 82: 1}
    x0 = torch.stack([x0[i] for i in range(len(x0)) if y0[i].item() in labels_dict])
    y0 = torch.stack([torch.tensor(labels_dict[int(cur_y)]) for cur_y in y0 if cur_y.item() in labels_dict])
    return x0, y0


def get_balanced_data(args, data_loader, data_amount):
    print('BALANCING DATASET...')
    # get balanced data
    data_amount_per_class = data_amount // 2

    labels_counter = {1: 0, 0: 0}
    label_classes = [0, 82]
    x0, y0 = [], []
    got_enough = False
    for bx, by in data_loader:
        if not label_classes[0] in by and not label_classes[1] in by:
            continue
        bx, by = create_labels(bx, by)
        for i in range(len(by)):
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
    return x0, y0


def load_tiny_imagenet_data(args):
    # Get Train Set
    print('TRAINSET BALANCED')
    data_loader = load_tiny_imagenet(root=args.datasets_dir, batch_size=100, train=True, shuffle=False, start=0, end=50000)
    x0, y0 = get_balanced_data(args, data_loader, args.data_amount)

    # Get Test Set (balanced)
    print('LOADING TESTSET')
    assert not args.data_use_test or (args.data_use_test and args.data_test_amount >= 2), f"args.data_use_test={args.data_use_test} but args.data_test_amount={args.data_test_amount}"
    data_loader = load_tiny_imagenet(root=args.datasets_dir, batch_size=100, train=False, shuffle=False, start=0, end=10000)
    x0_test, y0_test = get_balanced_data(args, data_loader, args.data_test_amount)

    # move to cuda and double
    x0, y0 = move_to_type_device(x0, y0, args.device)
    x0_test, y0_test = move_to_type_device(x0_test, y0_test, args.device)

    print(f'TRAIN BALANCE: 0: {y0[y0 == 0].shape[0]}, 1: {y0[y0 == 1].shape[0]}')
    print(f'TEST BALANCE: 0: {y0[y0 == 0].shape[0]}, 1: {y0[y0 == 1].shape[0]}')

    return [(x0, y0)], [(x0_test, y0_test)], None


def get_dataloader(args):
    args.input_dim = 64*64*3
    args.num_classes = 2
    args.output_dim = 1
    args.dataset = 'tiny_imagenet'

    if args.run_mode == 'reconstruct':
        args.extraction_data_amount = args.extraction_data_amount_per_class * args.num_classes

    # for legacy:
    args.data_amount = args.data_per_class_train * args.num_classes
    args.data_use_test = True
    args.data_test_amount = 100

    data_loader = load_tiny_imagenet_data(args)
    return data_loader