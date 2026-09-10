import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
        Seeds every RNG the pipeline draws from, so two runs of the same
        configuration produce identical results.

        Parameters
        ----------
        seed : int
            Seed applied to python's random, numpy and torch (cpu + cuda).
        deterministic : bool
            If True, disables the cuDNN autotuner and requests deterministic
            kernels. Costs some throughput but is required for run-to-run
            equality on the same hardware.

        Returns
        -------
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # CUBLAS workspace config is required for deterministic matmuls on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def seeded_generator(seed: int) -> torch.Generator:
    """
        Returns a torch.Generator for DataLoader shuffling, so batch order is
        reproducible independently of global RNG consumption elsewhere.

        Parameters
        ----------
        seed : int
            Seed for the generator.

        Returns
        -------
        torch.Generator
            Generator seeded with the given value.
    """

    g = torch.Generator()
    g.manual_seed(seed)
    return g
