

from dqd.ribs.archives import GridArchive
from dqd.ribs.emitters import ImprovementEmitter
from dqd.ribs.optimizers import Optimizer


def create_optimizer(algorithm, dim, seed):
    """Creates an optimizer based on the algorithm name.

    Args:
        algorithm (str): Name of the algorithm passed into sphere_main.
        dim (int): Dimensionality of the sphere function.
        seed (int): Main seed or the various components.
    Returns:
        Optimizer: A ribs Optimizer for running the algorithm.
    """