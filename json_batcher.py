import json
from collections import defaultdict
import time
from tqdm import tqdm
from prettytable import PrettyTable
from itertools import islice 
import numpy as np


class JSONBatcher:
  def __init__(self, dataset_file, batch_size = 4):
    self.dataset = self.load_dataset(dataset_file)
    self.batch_size = batch_size

  def load_dataset(self, dataset_file):
      with open(dataset_file, "r") as f:
          dataset = [json.loads(line) for line in f.readlines()]
      return dataset

  def get_num_levels(self, schema):
     return max(self.get_num_levels_examplewise(value) if isinstance(value,dict) else 0 for value in schema.values()) + 1
  
  def get_all_examples_recurse(self, schema, all_properties, prefix):
    for key in schema:
      value = schema[key]
      if isinstance(value, dict):
          self.get_all_examples_recurse(value, all_properties, prefix + "{}_".format(key))
      else:
          all_properties.append({
             prefix + key : value
          })

  def get_batches_recurse(self, schema, levelwise_properties, level, prefix):
    for key in schema:
      value = schema[key]
      if isinstance(value, dict):
          self.get_batches_recurse(value, levelwise_properties, level + 1, prefix + "{}_".format(key))
      else:
          levelwise_properties[level].append(str({
             prefix + key : value
          }))

  def split_schema(self, levelwise_properties):
    levelwise_batches = defaultdict(list)
    for level in levelwise_properties:
       properties = levelwise_properties[level]
       levelwise_batches[level] = np.array_split(np.array(levelwise_properties[level]), max(1, len(properties)/self.batch_size))

    return levelwise_batches
  
  def get_dataset(self, generate_prompt):
      data = []
      schemas = []

      for passage_idx in range(len(self.dataset)):
        passage = self.dataset[passage_idx]["passage"]
        schema = self.dataset[passage_idx]["schema"]
        all_properties = []
        self.get_all_examples_recurse(schema, all_properties, prefix = "")
        for schema in all_properties:
          data.append(generate_prompt(passage, schema))
          schemas.append(schema)
      
      return data, schemas

  def get_batches(self, generate_prompt):
      batches = []
      for passage_idx in range(len(self.dataset)):
        passage = self.dataset[passage_idx]["passage"]
        schema = self.dataset[passage_idx]["schema"]
        levelwise_properties = defaultdict(list)
        self.get_batches_recurse(schema, levelwise_properties, level = 0, prefix = "")
        levelwise_batches = self.split_schema(levelwise_properties)

        for level in levelwise_batches:
           for batch in levelwise_batches[level]:
              prompted_batch = []
              prompted_schemas = []

              for example in batch:
                 prompted_batch.append(generate_prompt(passage, example))
                 prompted_schemas.append(example)
                
              if len(prompted_batch) > 0: 
                batches.append([prompted_batch, prompted_schemas])

      return batches
