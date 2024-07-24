"""
Converts a multi-run results directory to be checkpoint-separated. Example:
run_1/eval_checkpoint-1_temp0.json.gz
run_1/eval_checkpoint-2_temp0.json.gz
run_1/eval_checkpoint-3_temp0.json.gz
run_2/eval_checkpoint-1_temp0.json.gz
run_2/eval_checkpoint-2_temp0.json.gz
run_2/eval_checkpoint-3_temp0.json.gz
run_3/eval_checkpoint-1_temp0.json.gz
run_3/eval_checkpoint-2_temp0.json.gz
run_3/eval_checkpoint-3_temp0.json.gz

To:
checkpoint_1/
checkpoint_1/run_1_eval_checkpoint-1_temp0.json.gz
checkpoint_1/run_2_eval_checkpoint-1_temp0.json.gz
checkpoint_1/run_3_eval_checkpoint-1_temp0.json.gz
checkpoint_2/
checkpoint_2/run_1_eval_checkpoint-2_temp0.json.gz
checkpoint_2/run_2_eval_checkpoint-2_temp0.json.gz
checkpoint_2/run_3_eval_checkpoint-2_temp0.json.gz
checkpoint_3/
checkpoint_3/run_1_eval_checkpoint-3_temp0.json.gz
checkpoint_3/run_2_eval_checkpoint-3_temp0.json.gz
checkpoint_3/run_3_eval_checkpoint-3_temp0.json.gz
"""
import os
import shutil
from pathlib import Path
import argparse


def normalize_multirun(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    for run_dir in input_path.iterdir():
        if not run_dir.is_dir():
            continue

        run_name = run_dir.name

        for checkpoint_file in run_dir.glob('*eval_checkpoint-*.json.gz'):
            checkpoint_num = checkpoint_file.stem.split('eval_checkpoint-')[-1].split('_')[0]
            checkpoint_dir = output_path / f"checkpoint-{checkpoint_num}"
            checkpoint_dir.mkdir(exist_ok=True)

            new_filename = f"{run_name}_{checkpoint_file.name}"
            print(f"Copying {checkpoint_file} to {checkpoint_dir / new_filename}")
            shutil.copy2(checkpoint_file, checkpoint_dir / new_filename)

    print(f"Normalized multi-run results saved to {output_path}")

def main(args):
    normalize_multirun(args.input_dir, args.output_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize multi-run evaluation results.")
    parser.add_argument('--input-dir', type=str, required=True, help="Input directory containing run folders")
    parser.add_argument('--output-dir', type=str, required=True, help="Output directory for normalized results")
    args = parser.parse_args()
    main(args)

