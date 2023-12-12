import argparse
import tqdm
import csv
import os
import sys

madlibs_dir = os.pardir
sys.path.append(madlibs_dir)

from madlibs.evals.run_benchmarks import BenchmarkRunner, Backend

def run_iters(num_iters,
              bm, 
              batch_size, 
              input_json,
              out_file):
    """Run inference and write outputs to CSV"""
    
    all_outputs = []
    for _ in tqdm.tqdm(range(num_iters)):

        if batch_size:
            bm.run_json_benchmark(
                input_json, 
                batch_size=batch_size,
                run_batching=True
            )

        else:
            bm.run_json_benchmark(
                input_json, 
                run_batching=False
            )

        bm.evaluator.run_eval()
        all_outputs.append(bm.evaluator.evals)

    #Write to CSV
    with open(out_file, 'w') as f:
        writer = csv.writer(f)
        column_names = list(all_outputs[0][0].keys())
        writer.writerow(column_names)
        for iteration in all_outputs:
            for output in iteration:
                writer.writerow(output.values())
        f.close()


def main():
    bm = BenchmarkRunner(args.model_name, Backend.VLLM)

    out_file = None
    if args.batch_size:
        out_file = os.path.join(args.out_dir, 'batched-json-dolly-{num_iters}_iters-{batch_size}_batchsize.csv')
    else:
        out_file = os.path.join(args.out_dir, 'unbatched-json-dolly-{num_iters}_iters.csv')
    
    run_iters(args.num_iters, 
              bm, 
              args.batch_size,
              args.input_json,
              out_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iters', type = int, default = 10)
    parser.add_argument('--batch_size', type = int)
    parser.add_argument('--out-dir', type = str, default = os.getcwd())
    parser.add_argument('--input-json', type = str)
    parser.add_argument('--model-name', type = str, default = "databricks/dolly-v2-3b")
    args = parser.parse_args()
    main()
