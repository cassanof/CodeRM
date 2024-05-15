def py_prompt(question: str, code=""):
    # escape any triple quotes in the question
    question = question.replace('"""', r'\"""')
    return f'''"""
{question}
"""
{code}'''


def py_prompt_3shot_taco(question: str, code=""):
    question = question.replace('"""', r'\"""')
    shots = '''# START NEW CODE
"""
You are given positive integers X and Y. If there exists a positive integer not greater than 10^{18} that is a multiple of X but not a multiple of Y, choose one such integer and print it. If it does not exist, print -1.

Constraints

* 1 ≤ X,Y ≤ 10^9
* X and Y are integers.

Input

Input is given from Standard Input in the following format:


X Y


Output

Print a positive integer not greater than 10^{18} that is a multiple of X but not a multiple of Y, or print -1 if it does not exist.

Examples

Input

8 6


Output

16


Input

3 3


Output

-1
"""
(A, B) = map(int, input().split())
if A % B == 0:
  print(-1)
  exit()
else:
  print(A)
# START NEW CODE
"""
The task is simply stated. Given an integer n (3 < n < 10^(9)), find the length of the smallest list of [*perfect squares*](https://en.wikipedia.org/wiki/Square_number) which add up to n. Come up with the best algorithm you can; you'll need it!

Examples:

sum_of_squares(17) = 2  17 = 16 + 1 (4 and 1 are perfect squares).
sum_of_squares(15) = 4  15 = 9 + 4 + 1 + 1. There is no way to represent 15 as the sum of three perfect squares.
sum_of_squares(16) = 1  16 itself is a perfect square.

Time constraints:

5 easy (sample) test cases: n < 20

5 harder test cases: 1000 < n < 15000

5 maximally hard test cases: 5 * 1e8 < n < 1e9

```if:java
300 random maximally hard test cases: 1e8 < n < 1e9
```
```if:c#
350 random maximally hard test cases: 1e8 < n < 1e9
```
```if:python
15 random maximally hard test cases: 1e8 < n < 1e9
```
```if:ruby
25  random maximally hard test cases: 1e8 < n < 1e9
```
```if:javascript
100 random maximally hard test cases: 1e8 < n < 1e9
```
```if:crystal
250 random maximally hard test cases: 1e8 < n < 1e9
```
```if:cpp
Random maximally hard test cases: 1e8 < n < 1e9
```
"""
def one_square(n):
  return round(n ** 0.5) ** 2 == n

def two_squares(n):
  while n % 2 == 0:
    n //= 2
  p = 3
  while p * p <= n:
    while n % (p * p) == 0:
      n //= p * p
    while n % p == 0:
      if p % 4 == 3:
        return False
      n //= p
    p += 2
  return n % 4 == 1

def three_squares(n):
  while n % 4 == 0:
    n //= 4
  return n % 8 != 7

def sum_of_squares(n):
  if one_square(n):
    return 1
  if two_squares(n):
    return 2
  if three_squares(n):
    return 3
  return 4
# START NEW CODE
"""
Little Petya likes to play very much. And most of all he likes to play the following game:

He is given a sequence of N integer numbers. At each step it is allowed to increase the value of any number by 1 or to decrease it by 1. The goal of the game is to make the sequence non-decreasing with the smallest number of steps. Petya is not good at math, so he asks for your help.

The sequence a is called non-decreasing if a1 ≤ a2 ≤ ... ≤ aN holds, where N is the length of the sequence.

Input

The first line of the input contains single integer N (1 ≤ N ≤ 5000) — the length of the initial sequence. The following N lines contain one integer each — elements of the sequence. These numbers do not exceed 109 by absolute value.

Output

Output one integer — minimum number of steps required to achieve the goal.

Examples

Input

5
3 2 -1 2 11


Output

4


Input

5
2 1 1 1 1


Output

1
"""
from bisect import insort

def min_steps_N(arr):
  pri_q = []
  ans = 0
  for n in arr:
    if pri_q:
      if pri_q[-1] > n:
        ans += pri_q[-1] - n
        pri_q.pop()
        insort(pri_q, n)
    insort(pri_q, n)
  return ans
N = input()
arr = list(map(int, input().split(' ')))
print(min_steps_N(arr))
# START NEW CODE
'''
    return shots + f'''"""
{question}
"""
{code}'''
