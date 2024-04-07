"""
作者：hp
日期：2022年11月01日
"""
from torchvision import datasets, transforms
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, cifar_noniid_shard


def Load_Data(args):
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
        dataset_train = datasets.MNIST('D:/山大/专业学习/Python相关/Federated-Learning-PyTorch-master/data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('D:/山大/专业学习/Python相关/Federated-Learning-PyTorch-master/data/mnist/', train=False, download=True, transform=trans_mnist)
        # dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        # dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.M_Prime)
        else:
            dict_users = mnist_noniid(dataset_train, args.M_Prime)

    elif args.dataset == 'fmnist':
        trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
        dataset_train = datasets.FashionMNIST('D:/山大/专业学习/Python相关/Federated-Learning-PyTorch-master/data/fashion_mnist/', train=True, download=True, transform=trans_fmnist)
        dataset_test = datasets.FashionMNIST('D:/山大/专业学习/Python相关/Federated-Learning-PyTorch-master/data/fashion_mnist/', train=False, download=True, transform=trans_fmnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.M_Prime)
        else:
            dict_users = mnist_noniid(dataset_train, args.M_Prime)
    elif args.dataset == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('D:/山大/专业学习/Python相关/Federated-Learning-PyTorch-master/data/cifar/', train=True, download=True, transform=transform_train)
        # dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10('D:/山大/专业学习/Python相关/Federated-Learning-PyTorch-master/data/cifar/', train=False, download=True, transform=transform_test)
        # dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.M_Prime)
        else:
            # dict_users = cifar_noniid(dataset_train, args.M_Prime)
            dict_users = cifar_noniid_shard(dataset_train, args.M_Prime)
    else:
        exit('Error: unrecognized dataset')

    return dict_users, dataset_train, dataset_test
