import os
import numpy as np
import json


class PostProcessor:
  def __init__(self, 
               outputs,
               batcher,
               out, 
               out_dir,
               **sampling_params):

    self.outputs = outputs
    self.batcher = batcher
    self.out = out
    self.out_dir = out_dir
    self.sampling_params = sampling_params

def clean_outputs(self):

  generated_texts = [self.outputs[i][0]["generated_text"] for i in range(len(self.outputs))]

  clean_generated_texts = []

  for generated_text in generated_texts:

    clean_generated_text = generated_text.replace("\n", "")
    open_bracket_index = clean_generated_text.find("{")
    
    close_bracket_instances = [i for i in range(len(clean_generated_text)) if clean_generated_text[i] == "}"]
    close_bracket_index = close_bracket_instances[-1]
    
    clean_generated_text = clean_generated_text[open_bracket_index : close_bracket_index + 1]
    clean_generated_texts.append(clean_generated_text)
  
  return clean_generated_texts

def write_outputs(self):

  if not os.path.isdir(self.out_dir):
    os.mkdir(self.out_dir)

  outfile = 'PLACEHOLDER.txt' #TODO FIX
  outfile_global = os.path.join(self.out_dir, outfile)

  with open(outfile_global, "a") as f:
    f.writelines([self.outputs[i] + "\n" for i in range(len(self.outputs))])
    f.close()

def build_json(self):
  unique_prompt_ids = set(self.batcher.prompt_ids)
  output_jsons = []

  for prompt_id in unique_prompt_ids:
    output_json = {}
    promptwise_filter = np.equal(self.batcher.prompt_ids, prompt_id)
    promptwise_outputs = np.array(self.outputs)[promptwise_filter]
    promptwise_properties = np.array(self.batcher.original_properties)[promptwise_filter]

    for i in range(len(promptwise_outputs)):
      current_level = {}

      promptwise_output = promptwise_outputs[i]
      promptwise_property = promptwise_properties[i]

      #TODO MAKE MORE NUANCED
      promptwise_output = promptwise_output.replace("\'", "\"")
      promptwise_output_dict = json.loads(promptwise_output)

      for j in range(len(promptwise_property) -1, -1, -1):
        level = promptwise_property[j]
        if j == len(promptwise_property) - 1:
          current_level[level] = promptwise_output_dict[list(promptwise_output_dict.keys())[0]]
        if j == 0:
          output_json[level] = current_level[list(current_level.keys())[0]]
        else:
          current_level = {level : current_level}

    #print(json.dumps(output_json, indent=4))
    #TODO ADD JSON DUMP TO WRITE TO SOME OUTPUT .json FILE
    output_jsons.append(output_json)
  
  return output_jsons

def run(self):

    cleaned_outputs = self.clean_outputs(self.outputs)

    if self.out:
      self.write_outputs()
    
    output_jsons = self.build_json()

    return output_jsons
