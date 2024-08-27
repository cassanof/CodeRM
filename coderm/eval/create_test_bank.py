import json
from coderm.prompts import py_prompt
from tqdm import tqdm
from coderm.execution import smart_exec_tests_queuebatched
from coderm import execution
import datasets
import hashlib
import gzip


def main(args):
    splits = []
    if args.split == "both":
        splits = ["train", "test"]
    elif args.split == "train":
        splits = ["train"]
    else:
        splits = ["test"]
    

    input_tests = []
    codes = []

    for split in splits:
        dataset = datasets.load_dataset(args.dataset, split=split)
        assert isinstance(dataset, datasets.Dataset)
        dataset = dataset.to_list()

        for i, item in enumerate(dataset):
            input_output = json.loads(item["input_output"])
            if input_output.get("exec_string", None) is not None:
                input_tests.append(input_output["exec_string"])
            else:
                input_tests.append(input_output)

            p = py_prompt(item["question"], item["starter_code"])
            codes.append(p)
            if "public_input_output" in item:
                public_input_output = json.loads(item["public_input_output"])
                if public_input_output.get("exec_string", None) is not None:
                    input_tests.append(public_input_output["exec_string"])
                else:
                    input_tests.append(public_input_output)

                codes.append(p)

    ids_special = []
    id_to_testout = {}
    for i in range(len(input_tests)):
        id_to_testout[i] = None
        ids_special.append(codes[i] + f"\n__START__{i}__END__")

    # monkey-patch exec_test in code_exec_server
    def patched_exec_test(server, code, test, *args, **kwargs):
        # extract id
        t_id = int(code.split("__START__")[1].split("__END__")[0])
        # put testout in id_to_testout
        assert test != ""
        id_to_testout[t_id] = test
        return False, "null"

    execution.exec_test = patched_exec_test

    smart_exec_tests_queuebatched(ids_special, input_tests, workers=1)

    bank = []
    already_hash = set()
    for i, test in tqdm(id_to_testout.items(), total=len(id_to_testout)):
        assert test is not None
        hashed = hashlib.md5(test.encode()).hexdigest()
        assert hashed not in already_hash, "Hash collision; check"
        bank.append({"test": test, "hash": hashed})
        already_hash.add(hashed)

    bank = datasets.Dataset.from_list(bank)
    if args.push:
        bank.push_to_hub(args.output, private=False)
    else:
        bank.save_to_disk(args.output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "both"],
        required=True,
        help="Split of the dataset to evaluate/generate from"
    )
    args = parser.parse_args()
    main(args)

