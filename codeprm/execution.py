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
    if actual.strip() == expected.strip():
        return True

    try:
        actual = actual.strip()
        expected = expected.strip()
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
        


if __name__ == "__main__":
    e = "0.5\n0.58333333\n0.70833333\n0.625\n0.578125\n0.525\n0.5\n0.83333333\n0.75\n0.56944444\n"
    a = "0.5\n0.5833333333\n0.7083333333\n0.6250000000\n0.5781250000\n0.5250000000\n0.5\n0.8333333333\n0.7500000000\n0.5694444444\n"
    print(compare_io(e, a, debug=True))
