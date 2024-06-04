from pathlib import Path
from tqdm import tqdm
from codeprm.model import OutcomeRewardModel
from codeprm.prompts import py_prompt
from codeprm.utils import strip_python_comments
from codeprm.utils import chunkify, gunzip_json_read, gunzip_json_write


def main(args):
    obj = gunzip_json_read(Path(args.input))
    assert obj is not None, "Could not read completions from " + \
        f"{args.input}"

    if ("gpt-" in obj["model"] or "chat" in obj["model"]) and not args.no_prefix_starter:
        print("Warning: Looks like the model is chat based but --no-prefix-starter is not set. " +
              "This might cause incorrect results. Please set --no-prefix-starter if the model is chat based.")

    completions = obj["items"]
    model = OutcomeRewardModel(args.model, device=args.device)
    for c in tqdm(completions, desc="Processing completions"):
        chunks = chunkify(list(enumerate(c["results"])), args.batch_size)
        for chunk in tqdm(chunks, desc="Processing chunks"):
            indices, results = zip(*chunk)
            contents = []
            for r in results:
                code = r["code"]
                if not args.no_prefix_starter:
                    code = c["starter_code"] + code
                if args.strip_comments:
                    code = strip_python_comments(code)

                contents.append(py_prompt(c["prompt"], code))

            scores = model.score(contents)
            for i, (label, score) in zip(indices, scores):
                c["results"][i]["orm_label"] = label
                c["results"][i]["orm_score"] = score

    gunzip_json_write(Path(args.output), obj)


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
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for the ORM model. Batching requires padding, so 1 is recommended.")
    parser.add_argument("--strip-comments", action="store_true",
                        help="Strip comments from the code before passing it to the ORM model.")
    parser.add_argument("--no-prefix-starter", action="store_true",
                        help="Do not prefix the starter code to the completion before passing it to the ORM model. This is needed for chat completion models.")
    args = parser.parse_args()
    main(args)
