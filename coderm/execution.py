from typing import List, Optional, Tuple, Union
import os
import threading
from tqdm import tqdm
import queue
import re
from coderm.code_exec_server.code_exec_reqs import exec_test, exec_test_batched
from coderm.utils import cached


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


def exec_io_test_batched(code, inps, outs, executor="http://127.0.0.1:8000", timeout=30) -> Tuple[bool, str]:
    instrus = [SOL_DEPS + IGNORE_WARNINGS + code for _ in inps]
    res = exec_test_batched(executor, instrus, [
                            ""] * len(instrus), timeout=timeout, stdins=inps, timeout_on_client=False)
    feedback = ""
    for i, (out, (passing, outs)) in enumerate(zip(outs, res)):
        if not passing:
            feedback = f"[{i}] errored with {outs!r}\n"
            break
        elif not compare_io(outs, out):
            feedback = f"[{i}] expected {out!r} but got {outs!r}\n"
            break

    return not bool(feedback), feedback


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


def exec_io_test_instrumented(code, inps, outs, executor="http://127.0.0.1:8000", timeout=30) -> Tuple[bool, str]:
    instru_code, instru_tests = instrument_io_code(code, inps)
    passing, outputs = exec_test(
        executor, instru_code, instru_tests, timeout=timeout, timeout_on_client=False)
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


@cached
def instrument_exec_tests(inps, outs, entrypoints: Union[str, List[str]]):
    tests = EQ_INSTRUMENTATION
    for inp, out, entrypoint in zip(inps, outs, entrypoints):
        args = ""
        for arg in inp:
            args += f"{arg!r}, "
        args = args.rstrip(", ")
        tests += f"assert is_eq({entrypoint}({args}), {out!r})\n"

    tests += "print('___SENTINEL___')\n"
    return tests


def exec_named_test(code, inps, outs, entrypoints: Union[str, List[str]], executor="http://127.0.0.1:8000", timeout=30) -> Tuple[bool, str]:
    if isinstance(entrypoints, str):
        entrypoints = [entrypoints] * len(inps)
    assert len(inps) == len(outs) == len(entrypoints)

    if "class Solution:" in code:
        entrypoints = [f"Solution().{e}" for e in entrypoints]

    instru = SOL_DEPS + code
    tests = instrument_exec_tests(inps, outs, entrypoints)
    passing, outs = exec_test(executor, instru, tests,
                              timeout=timeout, timeout_on_client=False)
    if passing:
        if "___SENTINEL___" not in outs:
            return False, "missing ___SENTINEL___ in output"
        return True, ""
    else:
        return False, f"errored with {outs!r}"


def exec_test_stringified(code, tests, executor="http://127.0.0.1:8000", timeout=30) -> Tuple[bool, str]:
    instru = SOL_DEPS + code
    tests = tests + "\n\n\nprint('___SENTINEL___')\n"
    passing, outs = exec_test(executor, instru, tests,
                              timeout=timeout, timeout_on_client=False)
    if passing:
        if "___SENTINEL___" not in outs:
            return False, "missing ___SENTINEL___ in output"
        return True, ""
    else:
        return False, f"errored with {outs!r}"


def smart_exec_tests(code, tests, executor="http://127.0.0.1:8000", timeout=30, batched_io=False) -> Tuple[bool, str]:
    """
    Supports various test formats:
        - simple I/O tests that use stdin and stdout
        - named tests that use assert statements
        - stringified tests that use a custom format
    """

    if batched_io:
        exec_io_test_fn = exec_io_test_batched
    else:
        exec_io_test_fn = exec_io_test_instrumented

    if isinstance(tests, str):
        return exec_test_stringified(code, tests, executor=executor, timeout=timeout)
    else:
        inputs = tests["inputs"]
        outputs = tests["outputs"]
        if "fn_name" in tests:
            name: Union[str, List[str]] = tests["fn_name"]
            return exec_named_test(code, inputs, outputs, name, executor=executor, timeout=timeout)
        else:
            return exec_io_test_fn(code, inputs, outputs, executor=executor, timeout=timeout)


def smart_exec_tests_queuebatched(
        codes,
        tests_per_code,
        executor="http://127.0.0.1:8000",
        timeouts: List[float] = [],
        workers=os.cpu_count(),
        use_tqdm=True
) -> List[Tuple[bool, str]]:
    if workers is None:
        print("WARNING: couldn't get number of workers, defaulting to 1")
        workers = 1

    results: List[Optional[Tuple[bool, str]]] = [None] * len(codes)

    if len(timeouts) == 0:
        timeouts = [30] * len(codes)

    assert len(timeouts) == len(codes) == len(tests_per_code), \
        f"Length mismatch in inputs: timeouts({len(timeouts)}), codes({len(codes)}), tests_per_code({len(tests_per_code)})"

    lock = threading.Lock()

    def worker(q: queue.Queue, pbar: Optional[tqdm]):
        while True:
            item = q.get()
            if item is None:
                break  # closed!

            i, code, tests, timeout = item
            results[i] = smart_exec_tests(
                code, tests, executor=executor, timeout=timeout)
            q.task_done()

            if pbar is not None:
                with lock:
                    pbar.update(1)

    q = queue.Queue()
    for i, (code, tests, timeout) in enumerate(zip(codes, tests_per_code, timeouts)):
        q.put((i, code, tests, timeout))

    if use_tqdm:
        pbar = tqdm(total=len(codes), desc="Executing tests")
    else:
        pbar = None

    threads = []
    for _ in range(workers):
        t = threading.Thread(target=worker, args=(q, pbar))
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
        else:
            results_new.append(r)

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
