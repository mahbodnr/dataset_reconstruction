
def setup_problem(args):
    if False:
        pass
    elif args.problem == 'cifar10_vehicles_animals':
        from problems.cifar10_vehicles_animals import get_dataloader
        return get_dataloader(args)
    elif args.problem == 'mnist_odd_even':
        from problems.mnist_odd_even import get_dataloader
        return get_dataloader(args)
    elif args.problem == 'mnist_odd_even_noisy_background':
        from problems.mnist_odd_even_noisy_background import get_dataloader
        return get_dataloader(args)
    elif args.problem == 'cifar10_noisy_background':
        from problems.cifar10_noisy_background import get_dataloader
        return get_dataloader(args)
    elif args.problem == 'tiny_imagenet':
        from problems.tiny_imagenet import get_dataloader
    else:
        raise ValueError(f'Unknown args.problem={args.problem}')
    return data_loader

