from pathlib import Path
from tqdm import tqdm
from codeprm.model import OutcomeRewardModel
from codeprm.prompts import py_prompt
from codeprm.utils import chunkify, gunzip_json_read, gunzip_json_write


def main(args):
    obj = gunzip_json_read(Path(args.input))
    assert obj is not None, "Could not read completions from " + \
        f"{args.input}"
    completions = obj["items"]
    model = OutcomeRewardModel(args.model, device=args.device)
    accurate = 0
    for c in tqdm(completions, desc="Processing completions"):
        print(c.keys())
        chunks = chunkify(list(enumerate(c["results"])), args.batch_size)
        for chunk in tqdm(chunks, desc="Processing chunks"):
            indices, results = zip(*chunk)
            contents = []
            for r in results:
                contents.append(
                    py_prompt(c["prompt"], c["starter_code"] + r["code"]))

            scores = model.score(contents)
            for i, (label, score) in zip(indices, scores):
                c["results"][i]["orm_label"] = label
                c["results"][i]["orm_score"] = score
                # get accuracy
                if bool(label) == c["results"][i]["passing"]:
                    accurate += 1

    gunzip_json_write(Path(args.output), obj)
    print(f"Accuracy: {accurate / len(completions)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the ORM")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use. By default, uses the first available GPU.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the input result file computed by the generator model.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the output file to write the results with ORM labels and scores.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for the ORM model.")
    args = parser.parse_args()
    main(args)
