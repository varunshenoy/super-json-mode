import os
import numpy as np
import json

def clean_outputs(outputs):

  generated_texts = [outputs[i][0]["generated_text"] for i in range(len(outputs))]

  clean_generated_texts = []

  for generated_text in generated_texts:

    clean_generated_text = generated_text.replace("\n", "")
    open_bracket_index = clean_generated_text.find("{")
    
    close_bracket_instances = [i for i in range(len(clean_generated_text)) if clean_generated_text[i] == "}"]
    close_bracket_index = close_bracket_instances[-1]
    
    clean_generated_text = clean_generated_text[open_bracket_index : close_bracket_index + 1]
    clean_generated_texts.append(clean_generated_text)
  
  return clean_generated_texts

def write_outputs(output_dir, outputs, batch_size, **sampling_params):

  if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

  outfile = os.path.join(output_dir, f'batch-size={batch_size}_max-length={sampling_params["max_length"]}_jsonbench_outputs.txt')

  with open(outfile, "a") as f:
    f.writelines([outputs[i] + "\n" for i in range(len(outputs))])
    f.close()

def build_json(outputs, original_properties, prompt_ids):
  unique_prompt_ids = set(prompt_ids)
  output_jsons = []

  for prompt_id in unique_prompt_ids:
    output_json = {}
    promptwise_filter = np.equal(prompt_ids, prompt_id)
    promptwise_outputs = np.array(outputs)[promptwise_filter]
    promptwise_properties = np.array(original_properties)[promptwise_filter]

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
    output_jsons.append(output_json)
  
  return output_jsons
  