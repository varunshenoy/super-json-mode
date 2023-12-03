import json
from collections import defaultdict
import time
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from prettytable import PrettyTable
from itertools import islice
import numpy as np
from utils import JSONBatcher, JSONDataset

class JSONGenerator:
    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
        self.dataset = self.load_dataset(dataset_file)

    def load_dataset(self, dataset_file):
        with open(dataset_file, "r") as f:
            dataset = [json.loads(line) for line in f.readlines()]
        return dataset

    def has_matching_schema(self, output, target):

        if type(output) is not type(target):
          return False

        output_keys = output.keys()
        target_keys = target.keys()

        if output_keys != target_keys:
          return False

        else:
          for key in output_keys:
            if type(output[key]) is dict:
              if not self.has_matching_schema(output[key], target[key]):
                return False

        return True

    def generate_prompt(self, passage, schema):
        user_message = f"""{passage}
        From the above passage, extract the following schema: {schema}

        Only output JSON with the allowed types."""

        prompt = f"""<s><<SYS>>You only respond in JSON. You do not add text before. You do not add text after. Only JSON. <</SYS>>[INST] {user_message} [/INST]"""
        return prompt
    
    def clean_outputs(outputs):
  
        generated_texts = [outputs[i][0]["generated_text"] for i in range(len(outputs))]

        clean_outputs = []

        for generated_text in generated_texts:

          clean_generated_text = generated_text.replace("\n", "")
          open_bracket_index = clean_generated_text.find("{")
          close_bracket_index = clean_generated_text.find("}")
          
          clean_generated_text = clean_generated_text[open_bracket_index : close_bracket_index + 1]
          clean_outputs.append(clean_generated_text)

        return clean_outputs
  
    def write_outputs(self, out_dir, outputs, batch_size, **sampling_params):

      if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

      outfile = os.path.join(out_dir, f'batch-size={batch_size}_\
                                        max-length={sampling_params["max_length"]}_\
                                        {self.dataset_file.split(".")[0]}_outputs.txt')

      generated_texts = [outputs[i][0]["generated_text"] for i in range(len(outputs))]

      with open(outfile, "a") as f:
        f.writelines(generated_texts)
        f.close()

    def run(self, generate, batch_sizes, out = True, out_dir = os.getcwd(), **sampling_params):

        evals = []

        #dataset generator object from raw JSON file
        batcher = JSONBatcher(self.dataset_file)
        data, schemas = batcher.get_dataset(self.generate_prompt)

        #Initialize Hugging Face Dataset object
        dataset = JSONDataset(data)

        for batch_size in batch_sizes:
          outputs = []
          run_times = []

          start_time = time.time()

          for out in tqdm(generate(dataset, batch_size = batch_size, **sampling_params)):
              time_taken = round(time.time() - start_time, 3)
              run_times.append(time_taken)
              outputs.append(out)
              start_time = time.time()

          for output, run_time, schema in zip(outputs, run_times, schemas):
              evaluation = {}

              result = output[0]["generated_text"].strip()
              result = result.replace("\'", "\"")

              evaluation["generation"] = result
              evaluation["time_taken"] = time_taken

              # check if result is valid JSON
              try:
                  json_result = json.loads(result)
                  evaluation["is_valid"] = True

                  # check if result matches schema
                  # JSON might have erroneous keys
                  evaluation["matches_schema"] = self.has_matching_schema(json_result, schema)
                  evaluation["error_type"] = None
              except ValueError:
                  evaluation["is_valid"] = False
                  evaluation["matches_schema"] = False

                  if result[0] != "{":
                      evaluation["error_type"] = "prefix"
                  elif result[-1] != "}":
                      evaluation["error_type"] = "suffix"
                  else:
                      evaluation["error_type"] = "invalid"

              evaluation["batch_size"] = batch_size
              evals.append(evaluation)


          if out:
            self.write_outputs(out_dir, outputs, batch_size, **sampling_params)
          
          clean_outputs = self.clean_outputs(outputs)

        return outputs, evals

    def print(self, evals, show_generation=False):
        table = PrettyTable()

        # Define the table columns
        table.field_names = [
            "Valid (✅/❌)",
            "Matches Schema (✅/❌)",
            "Batch Size",
            "Time (s)",
            "Error",
        ]
        if show_generation:
            table.add_column("Generation")

        valid_counter, schema_counter, total_time = 0, 0, 0

        for eval in evals:
            is_valid = "✅" if eval["is_valid"] else "❌"
            matches_schema = "✅" if eval["matches_schema"] else "❌"
            error_type = eval["error_type"]
            batch_size = eval["batch_size"]

            valid_counter += eval["is_valid"]
            schema_counter += eval["matches_schema"]
            total_time += eval["time_taken"]

            row = [is_valid, matches_schema, batch_size, eval["time_taken"], error_type]
            if show_generation:
                row.append(eval["generation"])

            table.add_row(row)

        valid_accuracy = valid_counter / len(evals)
        schema_accuracy = schema_counter / len(evals)
        average_time = round(total_time / len(evals), 3)

        table.add_row(["-", "-", "-", "-", "-"])
        table.add_row(
            [
                f"Accuracy: {valid_accuracy}",
                f"Accuracy: {schema_accuracy}",
                "-",
                f"Average: {average_time}",
                "-",
            ]
        )

        print(table)