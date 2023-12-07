import json
from collections import defaultdict
import time
from tqdm import tqdm
from prettytable import PrettyTable
from itertools import islice 
import numpy as np
import copy
from .jsondataset import *


class JSONBatcher:
  def __init__(self, dataset_file, generate_prompt, batch_size = 4):
    self.dataset = self.load_dataset(dataset_file)
    self.generate_prompt = generate_prompt
    self.batch_size = batch_size

  def load_dataset(self, dataset_file):
      with open(dataset_file, "r") as f:
          dataset = [json.loads(line) for line in f.readlines()]
      return dataset

  def get_num_levels(self, schema):
     return max(self.get_num_levels_examplewise(value) if isinstance(value,dict) else 0 for value in schema.values()) + 1
  
  def get_all_examples_recurse(self, schema, all_properties, prompt_id, prefix, traversed_keys : list = []):
    for key in schema:
      value = schema[key]

      updated_traversed_keys = copy.deepcopy(traversed_keys)
      updated_traversed_keys.append(key)

      if isinstance(value, dict):

          self.get_all_examples_recurse(value, all_properties, prompt_id, prefix + "{}_".format(key), updated_traversed_keys)
      else:
          all_properties.append({
             "collapsed_property" : 
             {
                prefix + key : value
             }, 
             "original_property" : 
             updated_traversed_keys,
             "prompt_id" : prompt_id
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
  
  def get_data(self):
      data = []
      schemas = []
      original_properties = []
      prompt_ids = []

      for prompt_id in range(len(self.dataset)):
        passage = self.dataset[prompt_id]["passage"]
        schema = self.dataset[prompt_id]["schema"]
        all_properties = []
        self.get_all_examples_recurse(schema, all_properties, prompt_id, prefix = "")

        for properties in all_properties:
          schema = properties["collapsed_property"]
          original_property = properties["original_property"]
          prompt_id = properties["prompt_id"]

          data.append(self.generate_prompt(passage, schema))
          schemas.append(schema)
          original_properties.append(original_property)
          prompt_ids.append(prompt_id)
          
      self.data = data
      self.schemas = schemas
      self.original_properties = original_properties
      self.prompt_ids = prompt_ids

      return 
    
  def get_dataset(self):
     self.get_data()
     return JSONDataset(self.data) 


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
