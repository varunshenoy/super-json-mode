import json
import time
from tqdm import tqdm
from prettytable import PrettyTable
import numpy as np


class JSONBatcher:
  def __init__(self, dataset_file, batch_size = 4):
    self.dataset = self.load_dataset(dataset_file)
    self.batch_size = batch_size
  
  def load_dataset(self, dataset_file):
      with open(dataset_file, "r") as f:
          dataset = [json.loads(line) for line in f.readlines()]
      return dataset

  def get_num_levels_examplewise(self, schema):
     return max(self.get_num_levels_examplewise(value) if isinstance(value,dict) else 0 for value in schema.values()) + 1

  def get_num_levels(self):
     num_levels = 0
     for data in self.dataset:
         schema = data['schema']
         num_levels = max(num_levels, self.get_num_levels_examplewise(schema))
        
     return num_levels
