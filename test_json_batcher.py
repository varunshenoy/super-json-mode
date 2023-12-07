from madlibs import *

dataset_file = 'example-jsons/jsonbench.jsonl'
engine = MadlibsEngine(dataset_file)
output_jsons = engine.run(batch_size = 4)