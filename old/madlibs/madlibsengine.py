import json
from collections import defaultdict
import time
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from prettytable import PrettyTable
from itertools import islice
import numpy as np
import os
from transformers import AutoTokenizer
import transformers
import torch
from .data import *
from .utils import *
from .generator import *

class MadlibsEngine:
    def __init__(self, 
                 dataset_file):
        self.dataset_file = dataset_file
        #TODO VARUN add more args about corresponding to model pipeline model here as you see fit. 

    def generate_prompt(self, passage, schema):
        user_message = f"""{passage}
        From the above passage, extract the following schema: {schema}

        Only output JSON with the allowed types."""

        prompt = f"""<s><<SYS>>You only respond in JSON. You do not add text before. You do not add text after. Only JSON. <</SYS>>[INST] {user_message} [/INST]"""
        return prompt
    
    def run(self, 
            batch_size, 
            eval = False, 
            out = True, 
            out_dir = os.getcwd(), 
            **sampling_params):

        #TODO replace with pipeline module
        model = "NousResearch/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model)

        pipeline = transformers.pipeline(
          "text-generation",
          model=model,
          torch_dtype=torch.float16,
          device_map="auto",
        )

        sampling_params = {
            "num_return_sequences": 1,
            "eos_token_id": tokenizer.eos_token_id,
            "max_length": 512,
            "return_full_text": False
        }

        #Run inference
        batcher = JSONBatcher(self.dataset_file, self.generate_prompt, batch_size)
        dataset = batcher.get_dataset()
        generator = JSONEvalGenerator(dataset, batcher) if eval else JSONGenerator(dataset, batcher)
        outputs, run_times = generator.run(pipeline, **sampling_params)

        # #Post processing and reconstruction
        post_processor = PostProcessor(outputs, batcher, out, out_dir, **sampling_params)
        output_jsons = post_processor.run()

        #Eval 
        if eval:
            generator.run_eval()
        
        return output_jsons




        
    