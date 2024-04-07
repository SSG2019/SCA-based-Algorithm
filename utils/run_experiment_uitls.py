"""
作者：hp
日期：2024年01月17日
"""


def set_RIS_cifar_minibatch_iid(args):
    args.dataset = 'cifar'
    args.EsN0dB = -9.0
    args.lr = 0.05
    args.epochs = 100
    args.iid = True
    args.local_ep = 5
    args.local_bs = 128
    return "ris_cifar_minibatch_iid"


def set_RIS_cifar_minibatch_niid(args):
    args.dataset = 'cifar'
    args.EsN0dB = -11.0
    args.lr = 0.05
    args.epochs = 100
    args.iid = False
    args.local_ep = 5
    args.local_bs = 128
    return "ris_cifar_minibatch_niid"


def set_RIS_fmnist_minibatch_iid(args):
    args.dataset = 'fmnist'
    args.EsN0dB = -20.0
    args.lr = 0.1
    args.epochs = 200
    args.iid = True
    args.local_ep = 5
    args.local_bs = 0
    return "ris_fmnist_minibatch_iid"


def set_RIS_fmnist_minibatch_niid(args):
    args.dataset = 'fmnist'
    args.EsN0dB = -17.0
    args.lr = 0.1
    args.epochs = 200
    args.iid = False
    args.local_ep = 5
    args.local_bs = 0
    return "ris_fmnist_minibatch_niid"


def set_RIS_mnist_minibatch_iid(args):
    args.dataset = 'mnist'
    args.EsN0dB = -21.0
    args.lr = 0.1
    args.epochs = 200
    args.iid = True
    args.local_ep = 5
    args.local_bs = 0
    return "ris_mnist_minibatch_iid"


def set_RIS_mnist_minibatch_niid(args):
    args.dataset = 'mnist'
    args.EsN0dB = -17.0
    args.lr = 0.1
    args.epochs = 200
    args.iid = False
    args.local_ep = 5
    args.local_bs = 0
    return "ris_mnist_minibatch_niid"


def set_RIS_mnist_fullbatch_iid(args):
    args.dataset = 'mnist'
    args.EsN0dB = -26.0
    args.lr = 0.1
    args.epochs = 200
    args.iid = True
    args.local_ep = 1
    args.local_bs = 0
    return "ris_mnist_fullbatch_iid"


def set_RIS_mnist_fullbatch_niid(args):
    args.dataset = 'mnist'
    args.EsN0dB = -20.0
    args.lr = 0.1
    args.epochs = 200
    args.iid = False
    args.local_ep = 1
    args.local_bs = 0
    return "ris_mnist_fullbatch_niid"


def set_RIS_EsN0(args):
    # mnist_minibatch_iid
    args.dataset = 'mnist'
    args.lr = 0.1
    args.epochs = 100
    args.iid = True
    args.local_ep = 5
    args.local_bs = 0
    return "ris_EsN0"


def set_RIS_ana_mnist_minibatch_iid(args):
    args.dataset = 'mnist'
    args.EsN0dB = -33.0
    args.lr = 0.1
    args.epochs = 100
    args.iid = True
    args.local_ep = 5
    args.local_bs = 0
    return "ris_ana_mnist_minibatch_iid"


def set_cifar_minibatch_iid(args):
    args.dataset = 'cifar'
    args.EsN0dB = -8.0
    args.lr = 0.05
    args.epochs = 100
    args.iid = True
    args.local_ep = 5
    args.local_bs = 128
    return "cifar_minibatch_iid"


def set_cifar_minibatch_niid(args):
    args.dataset = 'cifar'
    args.EsN0dB = -10.0
    args.lr = 0.05
    args.epochs = 100
    args.iid = False
    args.local_ep = 5
    args.local_bs = 128
    return "cifar_minibatch_niid"


def set_mnist_minibatch_iid(args):
    args.dataset = 'mnist'
    args.EsN0dB = -20.0
    args.lr = 0.1
    args.epochs = 200
    args.iid = True
    args.local_ep = 5
    args.local_bs = 0
    return "mnist_minibatch_iid"


def set_mnist_minibatch_niid(args):
    args.dataset = 'mnist'
    args.EsN0dB = -16.0
    args.lr = 0.1
    args.epochs = 200
    args.iid = False
    args.local_ep = 5
    args.local_bs = 0
    return "mnist_minibatch_niid"


def set_mnist_fullbatch_iid(args):
    args.dataset = 'mnist'
    args.EsN0dB = -26.0
    args.lr = 0.1
    args.epochs = 200
    args.iid = True
    args.local_ep = 1
    args.local_bs = 0
    return "mnist_fullbatch_iid"


def set_mnist_fullbatch_niid(args):
    args.dataset = 'mnist'
    args.EsN0dB = -20.0
    args.lr = 0.1
    args.epochs = 200
    args.iid = False
    args.local_ep = 1
    args.local_bs = 0
    return "mnist_fullbatch_niid"
