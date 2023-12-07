import json
from collections import defaultdict
import time
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from prettytable import PrettyTable
from itertools import islice
import numpy as np
from ..data import JSONBatcher, JSONDataset
from ..utils.postprocessing import *

class JSONGenerator:
    def __init__(self, dataset, batcher):
        self.dataset = dataset
        self.batcher = batcher

    def run(self, generate, **sampling_params):

        self.outputs = []
        self.run_times = []

        start_time = time.time()

        for out in tqdm(generate(self.dataset, batch_size = self.batcher.batch_size, **sampling_params)):
            time_taken = round(time.time() - start_time, 3)
            self.run_times.append(time_taken)
            self.outputs.append(out)
            start_time = time.time()
        
        return self.outputs, self.run_times