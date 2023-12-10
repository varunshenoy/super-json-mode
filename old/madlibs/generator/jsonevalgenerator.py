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
from ..utils import *
from .jsongenerator import JSONGenerator

class JSONEvalGenerator(JSONGenerator):
    def __init__(self, dataset, batcher):
        super().__init__(dataset, batcher)

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
    
    
    def run_eval(self):

        evals = []

        for output, run_time, schema in zip(self.outputs, self.run_times, self.batcher.schemas):
            evaluation = {}

            result = output[0]["generated_text"].strip()
            result = result.replace("\'", "\"")

            evaluation["generation"] = result
            evaluation["time_taken"] = run_time

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

            evaluation["batch_size"] = self.batcher.batch_size
            evals.append(evaluation)

        #TODO ADD STUFF TO WRITE 
        self.print(evals)
