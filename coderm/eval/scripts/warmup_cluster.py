from coderm.execution import smart_exec_tests_queuebatched
    
def main(args):
    dummy_code = "print('hi')"
    codes = [dummy_code] * args.workers
    tests = [""] * args.workers
    smart_exec_tests_queuebatched(codes, tests, workers=args.workers)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, required=True)
    args = parser.parse_args()
    main(args)