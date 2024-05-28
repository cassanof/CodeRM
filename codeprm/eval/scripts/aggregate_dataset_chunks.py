import datasets
from pathlib import Path
from tqdm import tqdm


def main(args):
    dses = []
    for subdir in tqdm(list(Path(args.input).iterdir()), desc='Loading datasets'):
        if subdir.is_dir():
            dses.append(datasets.load_from_disk(subdir))

    print('++ Concatenating datasets')
    ds = datasets.concatenate_datasets(dses)
    ds.save_to_disk(args.output)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
