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

def write_outputs(out_dir, outputs, batch_size, **sampling_params):

  if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

  outfile = os.path.join(out_dir, f'batch-size={batch_size}_\
                                    max-length={sampling_params["max_length"]}_\
                                    {self.dataset_file.split(".")[0]}_outputs.txt')

  generated_texts = [outputs[i][0]["generated_text"] for i in range(len(outputs))]

  with open(outfile, "a") as f:
    f.writelines(generated_texts)
    f.close()
