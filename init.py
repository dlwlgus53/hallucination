import torch
import numpy as np
import random
from log_conf import init_logger
from config import *


def init_experiment(args):
    seed = args.seed
    init_logger(args.output_dir)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
