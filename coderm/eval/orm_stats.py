"""
Takes in a result file and spits out the pass@k metrics.
"""
from pathlib import Path
from typing import Optional
import numpy as np
from coderm.utils import gunzip_json_read


def per_file_metrics(file: Path, tresh=0.9) -> Optional[str]:
    if file.is_dir():
        import datasets
        ds = datasets.load_from_disk(file)
        items = ds.to_list()
    else:
        obj = gunzip_json_read(file)
        assert obj is not None, f"Failed to read {file}"
        items = obj["items"]

    tp = 0
    tp_rates = []
    tp_ur = 0
    tn = 0
    tn_rates = []
    fp = 0
    fp_rates = []
    fp_or = 0
    fn = 0
    fn_rates = []

    size = 0
    for item in items:
        for res in item["results"]:
            if res["orm_label"] == 1:
                s = res["orm_1_score"]
                if res["passing"]:
                    tp += 1
                    if s < tresh:
                        tp_ur += 1
                    tp_rates.append(s)
                else:
                    fp += 1
                    if s >= tresh:
                        fp_or += 1
                    fp_rates.append(s)
            else:
                s = res["orm_0_score"]
                if not res["passing"]:
                    tn += 1
                    tn_rates.append(s)
                else:
                    fn += 1
                    fn_rates.append(s)

            size += 1

    acc = (tp + tn) / size
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / \
        (precision + recall) if precision + recall > 0 else 0

    fp_or = round((fp_or / fp) * 100, 2) # fp_or is relative to fp
    fp = round((fp / size) * 100, 2)
    fp_r = round(np.mean(fp_rates) * 100, 2)
    fn = round((fn / size) * 100, 2)
    fn_r = round(np.mean(fn_rates) * 100, 2)
    tp_ur = round((tp_ur / tp) * 100, 2) # tp_ur is relative to tp
    tp = round((tp / size) * 100, 2)
    tp_r = round(np.mean(tp_rates) * 100, 2)
    tn = round((tn / size) * 100, 2)
    tn_r = round(np.mean(tn_rates) * 100, 2)
    acc = round(acc * 100, 2)
    precision = round(precision * 100, 2)
    recall = round(recall * 100, 2)
    f1 = round(f1 * 100, 2)

    return f"{file.stem},{tp},{tp_r},{tp_ur},{tn},{tn_r},{fp},{fp_r},{fp_or},{fn},{fn_r},{acc},{precision},{recall},{f1}"


def main(args):
    header = "name,tp,tp_r,tp_ur,tn,tn_r,fp,fp_r,fp_or,fn,fn_r,acc,precision,recall,f1"
    print(header)
    for file in args.inputs:
        print(per_file_metrics(Path(file), args.t))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        type=str,
        nargs="+",
        help="Input files"
    )
    parser.add_argument(
        "-t",
        type=float,
        default=0.9,
        help="Threshold for positive class"
    )
    args = parser.parse_args()
    main(args)
