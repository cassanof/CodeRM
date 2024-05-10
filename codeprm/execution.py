from typing import Tuple
from codeprm.code_exec_server.code_exec_reqs import exec_test, exec_test_batched


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
"""


def compare_io(actual, expected, debug=False) -> bool:
    if actual == expected:
        return True
    actual = actual.strip()
    expected = expected.strip()
    if actual == expected:
        return True

    if actual.lower() == expected.lower():
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
                return False # nevermind

        # compare each line
        for aline, eline in zip(actual_lines, expected_lines):
            if aline == eline:
                continue
            if aline.lower() == eline.lower():
                continue
            # try float comparison, with some tolerance
            a = float(aline)
            e = float(eline)
            diff = abs(a - e)
            if diff < 1e-4:
                continue
            if debug:
                print(f"mismatch: {a} != {e} (diff: {diff})")
            return False

        return True
    except ValueError:
        pass

    return False

def exec_io_test(code, inps, outs, executor="http://127.0.0.1:8000", timeout=30) -> Tuple[bool, str]:
    instrus = [SOL_DEPS + code for _ in inps]
    res = exec_test_batched(executor, instrus, [""] * len(instrus), timeout=timeout, stdins=inps)
    feedback = ""
    good = True
    for inp, out, (passing, outs) in zip(inps, outs, res):
        if not passing:
            good = False
            feedback += f"[{inp!r}] errored with {outs!r}\n"
        elif not compare_io(outs, out):
            good = False
            feedback += f"[{inp!r}] expected {out!r} but got {outs!r}\n"

    return good, feedback


EQ_INSTRUMENTATION = """
def is_eq(a, b):
    if a == b:
        return True
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-4
    return False
"""

def exec_named_test(code, inps, outs, entrypoint, executor="http://127.0.0.1:8000", timeout=30) -> Tuple[bool, str]:
    instru = SOL_DEPS + code + EQ_INSTRUMENTATION
    tests = ""
    for inp, out in zip(inps, outs):
        args = ""
        for arg in inp:
            args += f"{arg!r}, "
        args = args.rstrip(", ")
        tests += f"assert is_eq({entrypoint}({args}), {out!r}), f\"\"\"expected {out!r} but got {{ {entrypoint}({args}) }}\"\"\"\n"

    passing, outs = exec_test(executor, instru, tests, timeout=timeout)
    return passing, outs
    
        
def smart_exec_tests(code, tests, executor="http://127.0.0.1:8000", timeout=30) -> Tuple[bool, str]:
    inputs = tests["inputs"]
    outputs = tests["outputs"]
    if "fn_name" in tests:
        name = tests["fn_name"]
        return exec_named_test(code, inputs, outputs, name, executor=executor, timeout=timeout)
    else:
        return exec_io_test(code, inputs, outputs, executor=executor, timeout=timeout)
