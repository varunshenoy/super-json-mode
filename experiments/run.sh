python run.py --batch_size 2 --input-json ../benchmark.jsonbench.jsonl
python run.py --batch_size 4 --input-json ../benchmark.jsonbench.jsonl
python run.py --batch_size 6 --input-json ../benchmark.jsonbench.jsonl
python run.py --batch_size 8 --input-json ../benchmark.jsonbench.jsonl

#unbatched/serial
python run.py --input-json ../benchmark.jsonbench.jsonl