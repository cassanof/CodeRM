import json
from tqdm import tqdm
from coderm.execution import smart_exec_tests_queuebatched
from coderm import execution
import datasets
import hashlib
import gzip


def main(args):
    dataset = datasets.load_dataset(args.dataset, split=args.split)
    dataset = dataset.to_list()

    input_tests = []
    if args.dataset_format == "lcb":
        for i, item in enumerate(dataset):
            input_tests.append(json.loads(item["input_output"]))
            if "public_input_output" in item:
                input_tests.append(json.loads(item["public_input_output"]))
    else:
        raise NotImplementedError

    ids_special = []
    id_to_testout = {}
    for i in range(len(input_tests)):
        id_to_testout[i] = None
        ids_special.append(f"_START_{i}_END_")

    # monkey-patch exec_test in code_exec_server
    def patched_exec_test(server, code, test, *args, **kwargs):
        # extract id
        t_id = int(code.split("_START_")[1].split("_END_")[0])
        print(t_id)
        # put testout in id_to_testout
        assert test != ""
        id_to_testout[t_id] = test
        return False, "null"

    execution.exec_test = patched_exec_test

    smart_exec_tests_queuebatched(ids_special, input_tests)

    bank = []
    for i, test in tqdm(id_to_testout.items(), total=len(id_to_testout)):
        assert test is not None
        hashed = hashlib.md5(test.encode()).hexdigest()
        bank.append({"test": test, "hash": hashed})

    bank = datasets.Dataset.from_list(bank)
    if args.push:
        bank.push_to_hub(args.output, private=False)
    else:
        bank.save_to_disk(args.output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="codegenning/livecodebench_lite_filtered")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset-format", type=str,
                        choices=["lcb"], default="lcb")
    args = parser.parse_args()
    main(args)
