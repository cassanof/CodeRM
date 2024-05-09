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


def instrument_input(inp):
    return f"""from io import StringIO
import sys
sys.stdin = StringIO({inp!r})
"""


def compare_io(actual, expected) -> bool:
    if actual == expected:
        return True
    if actual.strip() == expected.strip():
        return True

    try:
        # try float comparison
        actual = float(actual)
        expected = float(expected)
        return abs(actual - expected) < 1e-6
    except ValueError:
        pass

    return False

def exec_io_test(code, inps, outs, executor="http://127.0.0.1:8000", timeout=30) -> Tuple[bool, str]:
    instrus = [SOL_DEPS + instrument_input(inp) + code for inp in inps]
    res = exec_test_batched(executor, instrus, [""] * len(instrus), timeout=timeout)
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
        

