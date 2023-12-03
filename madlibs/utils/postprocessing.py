import os

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

def write_outputs(output_dir, outputs, batch_size, **sampling_params):

  if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

  outfile = os.path.join(output_dir, f'batch-size={batch_size}_max-length={sampling_params["max_length"]}_jsonbench_outputs.txt')

  with open(outfile, "a") as f:
    f.writelines([outputs[i] + "\n" for i in range(len(outputs))])
    f.close()