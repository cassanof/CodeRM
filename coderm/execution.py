from typing import List, Optional, Tuple, Union, Any
import os
import threading
from tqdm import tqdm
import queue
import re
import random
import time
from math import pow 
from coderm.code_exec_server.code_exec_reqs import exec_test
import multiprocessing


SOL_DEPS = """import sys
import time
import itertools
from itertools import accumulate, product, permutations, combinations
import collections
from collections import Counter, OrderedDict, deque, defaultdict, ChainMap
from functools import lru_cache
import math
from math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2
import fractions
from typing import List, Tuple
import numpy as np
import random
import heapq
from heapq import *
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)
if hasattr(sys, "setrecursionlimit"):
    sys.setrecursionlimit(10**6)
"""

IGNORE_WARNINGS = """
import warnings
warnings.filterwarnings("ignore")
"""

START_TIME_FACTOR = 15
EXPONENT = (1)
CLOSE_THRESHOLD = 0.001
WAIT_TIME_PER_LOOP = 0.5
MAX_STALL_LOOPS = (60 * 15) / WAIT_TIME_PER_LOOP

def compare_io(actual, expected, debug=False) -> bool:
    if isinstance(expected, list):  # this can happen apparently
        expected = "\n".join(expected)

    if actual == expected:
        if debug:
            print("exact match")
        return True

    actual = actual.strip()
    expected = expected.strip()
    if actual == expected:
        if debug:
            print("exact match after strip")
        return True

    if actual.lower() == expected.lower():
        if debug:
            print("case-insensitive match")
        return True

    try:
        # split into lines
        actual_lines = actual.splitlines()
        expected_lines = expected.splitlines()
        # strip each line
        actual_lines = [line.strip() for line in actual_lines]
        expected_lines = [line.strip() for line in expected_lines]

        if len(actual_lines) != len(expected_lines):
            # attempt to remove empty lines
            actual_lines = [line for line in actual_lines if line]
            expected_lines = [line for line in expected_lines if line]

            if len(actual_lines) != len(expected_lines):
                if debug:
                    print("line count mismatch")
                return False  # nevermind

        # compare each line
        for aline, eline in zip(actual_lines, expected_lines):
            if aline == eline:
                if debug:
                    print(f"line match: {aline} == {eline}")
                continue
            if aline.lower() == eline.lower():
                if debug:
                    print(f"line match (case-insensitive): {aline} == {eline}")
                continue
            # try float comparison, with some tolerance
            a = float(aline)
            e = float(eline)
            diff = abs(a - e)
            if diff < 1e-4:
                if debug:
                    print(f"float match: {a} == {e} (diff: {diff})")
                continue
            if debug:
                print(f"mismatch: {a} != {e} (diff: {diff})")
            return False

        if debug:
            print("all lines match")
        return True
    except ValueError:
        pass

    if debug:
        print("fallback comparison")
    return False


FROM_IMPORT_ALL_RE = re.compile(r"from\s+\S+\s+import\s+\*")
SYS_EXIT_RE = re.compile(r"sys.exit\(.*\)")
EXIT_RE = re.compile(r"exit\(.*\)")
QUIT_RE = re.compile(r"quit\(.*\)")


def instrument_io_code(code: str, inputs: List[str]) -> Tuple[str, str]:
    imports = re.findall(FROM_IMPORT_ALL_RE, code)
    code = re.sub(FROM_IMPORT_ALL_RE, "", code)

    # transform exits into returns. this is a bit of a hack, but in general, the model should not
    # use exits anyways.
    code = re.sub(SYS_EXIT_RE, "return", code)
    code = re.sub(EXIT_RE, "return", code)
    code = re.sub(QUIT_RE, "return", code)

    code_indented = "\n".join([f"    {line}" for line in code.splitlines()])
    code_closed = "def __run_prog__():\n" + code_indented

    for imp in imports:
        code_closed = imp + "\n" + code_closed

    instru = SOL_DEPS + IGNORE_WARNINGS + code_closed
    tests_str = "__inputs__ = " + str(inputs) + "\n"
    tests_str += "for __inp__ in __inputs__:\n"  # NOTE: sys is imported in SOL_DEPS
    tests_str += "    import io\n"
    tests_str += "    sys.stdin = io.TextIOWrapper(io.BytesIO(__inp__.encode()), encoding='utf8')\n"
    tests_str += "    __run_prog__()\n"
    tests_str += "    print(\"___SENTINEL___\")\n"
    return instru, tests_str


def exec_io_test_instrumented(code, inps, outs, executor="http://127.0.0.1:8000", timeout=30, testbank=None) -> Tuple[bool, str]:
    instru_code, instru_tests = instrument_io_code(code, inps)
    passing, outputs = exec_test(
        executor,
        instru_code,
        instru_tests,
        timeout=timeout,
        timeout_on_client=False,
        testhash_repo=testbank,
    )
    if not passing:
        return False, f"errored with {outputs!r}\n"

    outputs = outputs.split("___SENTINEL___")[:-1]
    if len(outputs) != len(outs):
        return False, f"expected {len(outs)} number of outputs but got {len(outputs)}\n"

    feedback = ""
    for i, (out, actual) in enumerate(zip(outs, outputs)):
        if not compare_io(actual, out):
            feedback = f"[{i}] expected {out!r} but got {actual!r}\n"
            break

    return not bool(feedback), feedback


EQ_INSTRUMENTATION = """
def is_eq(a, b):
    if a == b:
        return True
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-4
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if not is_eq(x, y):
                return False
        return True
    return False
"""


def exec_named_test(code, inps, outs, entrypoints: Union[str, List[str]], executor="http://127.0.0.1:8000", timeout: int = 30, has_Solution: Optional[bool] = None, testbank: Optional[str] = None) -> Tuple[bool, str]:
    if isinstance(entrypoints, str):
        entrypoints = [entrypoints] * len(inps)
    assert len(inps) == len(outs) == len(entrypoints)

    if has_Solution is None:
        has_Solution = "class Solution:" in code

    if has_Solution:
        entrypoints = [f"Solution().{e}" for e in entrypoints]

    instru = SOL_DEPS + code
    tests = EQ_INSTRUMENTATION
    for inp, out, entrypoint in zip(inps, outs, entrypoints):
        args = ""
        for arg in inp:
            args += f"{arg!r}, "
        args = args.rstrip(", ")
        tests += f"assert is_eq({entrypoint}({args}), {out!r})\n"

    tests += "print('___SENTINEL___')\n"

    passing, outs = exec_test(
        executor,
        instru,
        tests,
        timeout=timeout,
        timeout_on_client=False,
        testhash_repo=testbank,
    )
    if passing:
        if "___SENTINEL___" not in outs:
            return False, "missing ___SENTINEL___ in output"
        return True, ""
    else:
        return False, f"errored with {outs!r}"


def exec_test_stringified(code, tests, executor="http://127.0.0.1:8000", timeout=30, testbank=None) -> Tuple[bool, str]:
    instru = SOL_DEPS + code
    tests = tests + "\n\n\nprint('___SENTINEL___')\n"
    passing, outs = exec_test(
        executor,
        instru,
        tests,
        timeout=timeout,
        timeout_on_client=False,
        testhash_repo=testbank,
    )
    if passing:
        if "___SENTINEL___" not in outs:
            return False, "missing ___SENTINEL___ in output"
        return True, ""
    else:
        return False, f"errored with {outs!r}"


def smart_exec_tests(code, tests, executor="http://127.0.0.1:8000", timeout=30, has_Solution: Optional[bool] = None, testbank: Optional[str] = None) -> Tuple[bool, str]:
    """
    Supports various test formats:
        - simple I/O tests that use stdin and stdout
        - named tests that use assert statements
        - stringified tests that use a custom format
    """

    if isinstance(tests, str):
        return exec_test_stringified(code, tests, executor=executor, timeout=timeout, testbank=testbank)
    else:
        inputs = tests["inputs"]
        outputs = tests["outputs"]
        if "fn_name" in tests:
            name: Union[str, List[str]] = tests["fn_name"]
            return exec_named_test(code, inputs, outputs, name, executor=executor, timeout=timeout, has_Solution=has_Solution, testbank=testbank)
        else:
            return exec_io_test_instrumented(code, inputs, outputs, executor=executor, timeout=timeout, testbank=testbank)


def smart_exec_tests_queuebatched_with_mp(
    codes: list[str],
    tests_per_code: list[Union[dict[str, Any], str]],
    executor: str = "http://127.0.0.1:8000",
    timeouts: Optional[List[int]] = None,
    has_Solution_per_code: Optional[list[Optional[bool]]] = None,
    num_workers: int = os.cpu_count(),
    total_num_concurrent: int = 1000,
    use_tqdm: bool = True,
    testbank: Optional[str] = None,
    return_none: bool = False
) -> List[Tuple[bool, str]]:
    if timeouts is None or len(timeouts) == 0:
        timeouts = [30] * len(codes)
    if has_Solution_per_code is None:
        has_Solution_per_code = [None] * len(codes)

    assert len(timeouts) == len(codes) == len(tests_per_code), \
        f"Length mismatch in inputs: timeouts({len(timeouts)}), codes({len(codes)}), tests_per_code({len(tests_per_code)})"

    num_threads = [total_num_concurrent // num_workers] * num_workers
    for i in range(total_num_concurrent % num_workers):
        num_threads[i] += 1
    
    print(len(num_threads), "AAAAA\n\nAAAA", num_threads)

    codes_per_process = [[] for _ in range(num_workers)]
    tests_per_process = [[] for _ in range(num_workers)]
    timeouts_per_process = [[] for _ in range(num_workers)]
    has_Solution_per_processes = [[] for _ in range(num_workers)]
    orig_idxs_per_process = [[] for _ in range(num_workers)]

    for i in range(len(codes)):
        codes_per_process[i % num_workers].append(codes[i])
        tests_per_process[i % num_workers].append(tests_per_code[i])
        timeouts_per_process[i % num_workers].append(timeouts[i])
        has_Solution_per_processes[i % num_workers].append(has_Solution_per_code[i])
        orig_idxs_per_process[i % num_workers].append(i)
    
    final_results = [None] * len(codes)
    with multiprocessing.Manager() as manager:
        with tqdm(total=len(codes)) as pbar:
            iter_tracker = manager.list([0] * num_workers)
            return_list = manager.list([None] * num_workers)
            curr_finished = 0
            processes: list[multiprocessing.Process] = []
            for i in range(num_workers):
                processes.append(
                    multiprocessing.Process(
                        target=smart_exec_tests_queuebatched,
                        args=(
                            codes_per_process[i],
                            tests_per_process[i],
                            executor,
                            timeouts_per_process[i],
                            has_Solution_per_processes[i],
                            num_threads[i],
                            False,
                            testbank,
                            i,
                            iter_tracker,
                            return_list,
                        )
                    )
                )
            # Start all processes
            for process in processes:
                process.start()

            num_iter_same = 0
            is_hanging = False
            # Wait for all processes to finish
            while any(process.is_alive() for process in processes):
                new_sum = sum(iter_tracker)
                if new_sum == curr_finished:
                    num_iter_same += 1
                else:
                    num_iter_same = 0
                if num_iter_same > MAX_STALL_LOOPS and (len(codes) - curr_finished) / len(codes) <= CLOSE_THRESHOLD:
                    is_hanging = True
                    print("uhhhhh... hanging? So jumping to joins.")
                    break
                pbar.update(new_sum - curr_finished)
                assert new_sum >= curr_finished
                curr_finished = new_sum
                time.sleep(0.5)  # Sleep briefly to avoid excessive CPU usage

            print("Probably finished")

        if is_hanging:
            print("Forcefully terminating processes.")
            for process in processes:
                process.terminate()

        # Ensure all processes have finished
        for process in processes:
            process.join()
        # Final update to curr_finished
        assert all(l is not None for l in return_list)
        final_sum = sum(len(l) for l in return_list)
        assert final_sum == len(codes)
        pbar.update(final_sum - curr_finished)

        for orig_idxs, proc_outputs in zip(orig_idxs_per_process, return_list):
            assert isinstance(proc_outputs, list)
            for orig_idx, proc_output in zip(orig_idxs, proc_outputs):
                if proc_output is None and not return_none:
                    final_results[orig_idx] = (False, "Process hanging error.")
                else:
                    final_results[orig_idx] = proc_output

    assert all(res is not None for res in final_results)
    return final_results
  

def smart_exec_tests_queuebatched(
        codes,
        tests_per_code,
        executor="http://127.0.0.1:8000",
        timeouts: Optional[List[int]] = None,
        has_Solution_per_code: Optional[list[Optional[bool]]] = None,
        workers=os.cpu_count(),
        use_tqdm=True,
        testbank=None,
        process_idx: Optional[int] = None,
        iter_tracker: Optional[list[int]] = None,
        return_list: Optional[list[Any]] = None,
) -> List[Tuple[bool, str]]:
    if timeouts is None:
        timeouts = []
    if len(timeouts) == 0:
        timeouts = [30] * len(codes)
    assert len(timeouts) == len(codes) == len(tests_per_code), \
        f"Length mismatch in inputs: timeouts({len(timeouts)}), codes({len(codes)}), tests_per_code({len(tests_per_code)})"

    assert (iter_tracker is None) == (process_idx is None) == (return_list is None)
    is_in_subprocess = iter_tracker is not None
    if is_in_subprocess:
        assert not use_tqdm

    if workers is None:
        print("WARNING: couldn't get number of workers, defaulting to 1")
        workers = 1
    if has_Solution_per_code is None:
        has_Solution_per_code = [None] * len(codes)

    results: List[Optional[Tuple[bool, str]]] = [None] * len(codes)
    return_list[process_idx] = results

    lock = threading.Lock()

    def worker(q: queue.Queue, pbar: Optional[tqdm], i: int):
        sleep_time = random.random() * pow(i, EXPONENT) * START_TIME_FACTOR
        time.sleep(sleep_time)
        while True:
            item = q.get()
            if item is None:
                break  # closed!

            i, code, tests, timeout, has_Solution = item
            results[i] = smart_exec_tests(
                code,
                tests,
                executor=executor,
                timeout=timeout,
                has_Solution=has_Solution,
                testbank=testbank
            )
            q.task_done()

            with lock:
                if is_in_subprocess:
                    iter_tracker[process_idx] += 1
                    return_list[process_idx] = results

                if pbar is not None:
                        pbar.update(1)

    q = queue.Queue()
    for i, (code, tests, timeout, has_Solution) in enumerate(zip(codes, tests_per_code, timeouts, has_Solution_per_code)):
        q.put((i, code, tests, timeout, has_Solution))

    if use_tqdm:
        pbar = tqdm(total=len(codes), desc="Executing tests")
    else:
        pbar = None

    threads = []
    for i in range(workers):
        t = threading.Thread(target=worker, args=(q, pbar, i))
        t.start()
        threads.append(t)

    # block until all tasks are done
    q.join()

    # close the threads
    for _ in range(workers):
        q.put(None)

    for t in threads:
        t.join()

    results_new = []
    for r in results:
        if r is None:
            results_new.append(
                (False, "Failed to execute program. Thread error."))
            assert False
        else:
            results_new.append(r)
    
    if is_in_subprocess:
        return_list[process_idx] = results_new

    return results_new


def parse_time_limit(limit: str, default=30, scaling_factor=2) -> int:
    if limit is None or not isinstance(limit, str):
        return default
    if "-" in limit:
        # get the second number after the dash
        limit = limit.split("-")[1].strip().split()[0]
        limit = float(limit)  # type: ignore
        return (int(limit) + 1) * scaling_factor
    split = limit.split()
    num = float(split[0])
    return (int(num) + 1) * scaling_factor  # add 1 second to be safe
