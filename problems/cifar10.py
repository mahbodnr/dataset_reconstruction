import torch
import torchvision.datasets
import torchvision.transforms as transforms


def get_dataloader(args):
    args.input_dim = 32 * 32 * 3
    args.num_classes = args.n_class
    args.output_dim = 10
    args.dataset = "cifar10"

    if args.run_mode == "reconstruct":
        args.extraction_data_amount = (
            args.extraction_data_amount_per_class * args.num_classes
        )

    # for legacy:
    args.data_amount = args.data_per_class_train * args.num_classes
    args.data_use_test = True
    args.data_test_amount = 1000

    data_loader = load_cifar10_data(args)
    return data_loader


def load_cifar10_data(args):
    """
    loads cifar10 dataloader and selects ´args.data_per_class_train' data per class
    """
    # Transformation to normalize data and convert to tensors
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(
            #     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            # ),  # Normalize to [-1, 1]
        ]
    )

    # Load the full CIFAR-10 dataset
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=args.datasets_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.datasets_dir, train=False, download=True, transform=transform
    )

    # Initialize lists to hold limited samples
    limited_train_samples = []
    limited_train_labels = []

    # Count of samples per class
    samples_per_class = [0] * 10

    for data, label in full_train_dataset:
        if samples_per_class[label] < args.data_per_class_train:
            limited_train_samples.append(data)
            limited_train_labels.append(label)
            samples_per_class[label] += 1

    # Convert lists to tensors
    X_train = torch.stack(limited_train_samples)
    Y_train = torch.tensor(limited_train_labels)

    # Convert test set to tensors
    X_test = torch.stack([data for data, _ in test_dataset])
    Y_test = torch.tensor([label for _, label in test_dataset])

    return [(X_train, Y_train)], [(X_test, Y_test)], None
