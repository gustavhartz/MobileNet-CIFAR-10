import argparse
import random
import string
import numpy as np
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Function that generates random string of length nª


def random_string(n):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(n))


def set_seed(args):
    """Create a random seed for reproducibility

    Args:
        args (_type_): Argparse
    """
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
