from typing import List, Tuple
import datasets
from coderm.utils import chunkify
import random
from tqdm import tqdm
from vllm import LLM, SamplingParams
from coderm.model import autodetect_dtype_str


FEW_SHOTS: List[Tuple[str, str]] = [
    ("""For some reason in many American cartoons anvils fall from time to time onto heroes' heads. Of course, safes, wardrobes, cruisers, planes fall sometimes too... But anvils do so most of all.

Anvils come in different sizes and shapes. Quite often they get the hero stuck deep in the ground. But have you ever thought who throws anvils from the sky? From what height? We are sure that such questions have never troubled you!

It turns out that throwing an anvil properly is not an easy task at all. Let's describe one of the most popular anvil throwing models.

Let the height p of the potential victim vary in the range [0;a] and the direction of the wind q vary in the range [ - b;b]. p and q could be any real (floating) numbers. Then we can assume that the anvil will fit the toon's head perfectly only if the following equation has at least one real root:

<image>

Determine the probability with which an aim can be successfully hit by an anvil.

You can assume that the p and q coefficients are chosen equiprobably and independently in their ranges.

Input

The first line contains integer t (1 ≤ t ≤ 10000) — amount of testcases.

Each of the following t lines contain two space-separated integers a and b (0 ≤ a, b ≤ 106).

Pretests contain all the tests with 0 < a < 10, 0 ≤ b < 10.

Output

Print t lines — the probability of a successful anvil hit for each testcase. The absolute or relative error of the answer should not exceed 10 - 6.

Examples

Input

2
4 2
1 2


Output

0.6250000000
0.5312500000""",
     """In many American cartoons, it’s a recurring gag to see anvils falling on the heroes’ heads. Occasionally, safes, wardrobes, and even larger objects like cruisers or planes might fall too, but anvils are the most common. These anvils come in various sizes and shapes and often leave the hero stuck in the ground. But have you ever wondered who is dropping these anvils from the sky? From what height? It’s likely that these questions have never crossed your mind!

Throwing an anvil accurately turns out to be quite challenging. Let’s delve into one of the most common models for this.

Consider the height p of the potential victim, which varies in the range [0; a], and the wind direction q, which varies in the range [-b; b]. Both p and q can be any real (floating-point) numbers. We assume that the anvil will hit the target’s head precisely if the following equation has at least one real root:

(image)

Your task is to determine the probability of hitting the target accurately with the anvil.

Assume p and q are chosen uniformly and independently within their respective ranges.

### Input

- The first line contains an integer t (1 ≤ t ≤ 10000) — the number of test cases.
- Each of the following t lines contains two space-separated integers a and b (0 ≤ a, b ≤ 10^6).

The pretests include all tests with 0 < a < 10 and 0 ≤ b < 10.

### Output

Print t lines — the probability of a successful anvil hit for each test case. The absolute or relative error of the answer should not exceed 10^-6.

### Examples

**Input**
2
4 2
1 2

**Output**
0.6250000000
0.5312500000"""),
    (r"""Write a program which reads $n$ items and sorts them. Each item has attributes $\\{value, weight, type, date, name\\}$ and they are represented by $\\{$ integer, integer, upper-case letter, integer, string $\\}$ respectively. Sort the items based on the following priorities.

1. first by value (ascending)
2. in case of a tie, by weight (ascending)
3. in case of a tie, by type (ascending in lexicographic order)
4. in case of a tie, by date (ascending)
5. in case of a tie, by name (ascending in lexicographic order)

Constraints

* $1 \leq n \leq 100,000$
* $0 \leq v_i \leq 1,000,000,000$
* $0 \leq w_i \leq 1,000,000,000$
* $t_i$ is a upper-case letter
* $0 \leq d_i \leq 2,000,000,000,000$
* $1 \leq $ size of $s_i \leq 20$
* $s_i \ne s_j$ if $(i \ne j)$

Input

The input is given in the following format.


$n$
$v_0 \; w_0 \; t_0 \; d_0 \; s_0$
$v_1 \; w_1 \; t_1 \; d_1 \; s_1$
:
$v_{n-1} \; w_{n-1} \; t_{n-1} \; d_{n-1} \; s_{n-1}$


In the first line, the number of items $n$. In the following $n$ lines, attributes of each item are given. $v_i \; w_i \; t_i \; d_i \; s_i$ represent value, weight, type, date and name of the $i$-th item respectively.

Output

Print attributes of each item in order. Print an item in a line and adjacency attributes should be separated by a single space.

Example

Input

5
105 24 C 1500000000000 white
100 23 C 1500000000000 blue
105 23 A 1480000000000 pink
110 25 B 1500000000000 black
110 20 A 1300000000000 gree


Output

100 23 C 1500000000000 blue
105 23 A 1480000000000 pink
105 24 C 1500000000000 white
110 20 A 1300000000000 gree
110 25 B 1500000000000 black""",
     r"""Create a program that reads `n` items and sorts them according to certain attributes. Each item has five attributes: value, weight, type, date, and name. These attributes are represented as follows: integer, integer, uppercase letter, integer, and string, respectively. The sorting should be performed based on the following criteria:

1. First, by value (in ascending order)
2. If values are equal, then by weight (in ascending order)
3. If both value and weight are equal, then by type (in ascending lexicographic order)
4. If value, weight, and type are all equal, then by date (in ascending order)
5. If all previous attributes are equal, then by name (in ascending lexicographic order)

Constraints:
- 1 ≤ n ≤ 100,000
- 0 ≤ value ≤ 1,000,000,000
- 0 ≤ weight ≤ 1,000,000,000
- type is an uppercase letter
- 0 ≤ date ≤ 2,000,000,000,000
- 1 ≤ length of name ≤ 20
- Each name is unique

Input:
The input format is as follows:
- The first line contains an integer `n`, representing the number of items.
- Each of the next `n` lines contains five space-separated values: value (integer), weight (integer), type (uppercase letter), date (integer), and name (string).

Output:
Print the sorted items, each on a new line with attributes separated by a single space.

Example:

Input:
5
105 24 C 1500000000000 white
100 23 C 1500000000000 blue
105 23 A 1480000000000 pink
110 25 B 1500000000000 black
110 20 A 1300000000000 gree

Output:
100 23 C 1500000000000 blue
105 23 A 1480000000000 pink
105 24 C 1500000000000 white
110 20 A 1300000000000 gree
110 25 B 1500000000000 black"""
     ),
    (
        r"""Gildong has a square board consisting of n rows and n columns of square cells, each consisting of a single digit (from 0 to 9). The cell at the j-th column of the i-th row can be represented as (i, j), and the length of the side of each cell is 1. Gildong likes big things, so for each digit d, he wants to find a triangle such that:

  * Each vertex of the triangle is in the center of a cell.
  * The digit of every vertex of the triangle is d.
  * At least one side of the triangle is parallel to one of the sides of the board. You may assume that a side of length 0 is parallel to both sides of the board.
  * The area of the triangle is maximized.



Of course, he can't just be happy with finding these triangles as is. Therefore, for each digit d, he's going to change the digit of exactly one cell of the board to d, then find such a triangle. He changes it back to its original digit after he is done with each digit. Find the maximum area of the triangle he can make for each digit.

Note that he can put multiple vertices of the triangle on the same cell, and the triangle can be a [degenerate triangle](https://cutt.ly/NhbjZ2l); i.e. the area of the triangle can be 0. Also, note that he is allowed to change the digit of a cell from d to d.

Input

Each test contains one or more test cases. The first line contains the number of test cases t (1 ≤ t ≤ 1000).

The first line of each test case contains one integer n (1 ≤ n ≤ 2000) — the number of rows and columns of the board.

The next n lines of each test case each contain a string of n digits without spaces. The j-th digit of the i-th line is the digit of the cell at (i, j). Each digit is one of the characters from 0 to 9.

It is guaranteed that the sum of n^2 in all test cases doesn't exceed 4 ⋅ 10^6.

Output

For each test case, print one line with 10 integers. The i-th integer is the maximum area of triangle Gildong can make when d = i-1, multiplied by 2.

Example

Input


5
3
000
122
001
2
57
75
4
0123
4012
3401
2340
1
9
8
42987101
98289412
38949562
87599023
92834718
83917348
19823743
38947912


Output


4 4 1 0 0 0 0 0 0 0
0 0 0 0 0 1 0 1 0 0
9 6 9 9 6 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
18 49 49 49 49 15 0 30 42 42

Note

In the first case, for d=0, no matter which cell he chooses to use, the triangle with vertices at (1, 1), (1, 3), and (3, 1) is the biggest triangle with area of \cfrac{2 ⋅ 2}{2} = 2. Since we should print it multiplied by 2, the answer for d=0 is 4.

For d=1, Gildong can change the digit of the cell at (1, 3) into 1, making a triangle with vertices on all three 1's that has an area of 2.

For d=2, Gildong can change the digit of one of the following six cells into 2 to make a triangle with an area of \cfrac{1}{2}: (1, 1), (1, 2), (1, 3), (3, 1), (3, 2), and (3, 3).

For the remaining digits (from 3 to 9), the cell Gildong chooses to change will be the only cell that contains that digit. Therefore the triangle will always be a degenerate triangle with an area of 0.

In the third case, for d=4, note that the triangle will be bigger than the answer if Gildong changes the digit of the cell at (1, 4) and use it along with the cells at (2, 1) and (4, 3), but this is invalid because it violates the condition that at least one side of the triangle must be parallel to one of the sides of the board.""",
        r"""Gildong has a square board with n rows and n columns, where each cell contains a single digit from 0 to 9. The cell at the j-th column of the i-th row is denoted as (i, j), and each cell is a square with a side length of 1. Gildong enjoys maximizing things, so for each digit d, he wants to find the largest possible triangle that meets the following criteria:

- Each vertex of the triangle is located at the center of a cell.
- The digit at each vertex of the triangle is d.
- At least one side of the triangle is parallel to one of the board's sides (a side of length 0 is considered parallel to both sides).
- The area of the triangle is maximized.

To achieve this, for each digit d, he will change one cell on the board to d, find the largest possible triangle, and then revert the cell back to its original digit. The goal is to determine the maximum possible area of the triangle for each digit.

Notes:
- Vertices of the triangle can overlap, resulting in a degenerate triangle with an area of 0.
- Changing a cell from digit d to d is allowed.

=== Input ===
The input consists of multiple test cases.

1. The first line contains an integer t (1 ≤ t ≤ 1000), representing the number of test cases.
2. For each test case:
    - The first line contains an integer n (1 ≤ n ≤ 2000), the number of rows and columns of the board.
    - The next n lines each contain a string of n digits, representing the board.

It is guaranteed that the sum of n^2 across all test cases does not exceed 4 × 10^6.

=== Output ===
For each test case, print a line with 10 integers. The i-th integer is the maximum area of the triangle Gildong can make when d = i-1, multiplied by 2.

=== Example ===

**Input:**
5
3
000
122
001
2
57
75
4
0123
4012
3401
2340
1
9
8
42987101
98289412
38949562
87599023
92834718
83917348
19823743
38947912

**Output:**
4 4 1 0 0 0 0 0 0 0
0 0 0 0 0 1 0 1 0 0
9 6 9 9 6 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
18 49 49 49 49 15 0 30 42 42

=== Note ===
In the first case, for d=0, the largest triangle has vertices at (1, 1), (1, 3), and (3, 1) with an area of 2. Since the output should be multiplied by 2, the result for d=0 is 4.

For d=1, changing the digit of the cell at (1, 3) to 1 forms a triangle with an area of 2.

For d=2, changing any of the six cells to 2 results in a triangle with an area of 0.5.

For digits from 3 to 9, changing one cell results in a degenerate triangle with an area of 0.

In the third case, for d=4, the triangle is invalid if one side is not parallel to the board sides, despite being larger."""
    ),
    (
        r"""Squirrel Liss lived in a forest peacefully, but unexpected trouble happens. Stones fall from a mountain. Initially Squirrel Liss occupies an interval [0, 1]. Next, n stones will fall and Liss will escape from the stones. The stones are numbered from 1 to n in order.

The stones always fall to the center of Liss's interval. When Liss occupies the interval [k - d, k + d] and a stone falls to k, she will escape to the left or to the right. If she escapes to the left, her new interval will be [k - d, k]. If she escapes to the right, her new interval will be [k, k + d].

You are given a string s of length n. If the i-th character of s is "l" or "r", when the i-th stone falls Liss will escape to the left or to the right, respectively. Find the sequence of stones' numbers from left to right after all the n stones falls.

Input

The input consists of only one line. The only line contains the string s (1 ≤ |s| ≤ 106). Each character in s will be either "l" or "r".

Output

Output n lines — on the i-th line you should print the i-th stone's number from the left.

Examples

Input

llrlr


Output

3
5
4
2
1


Input

rrlll


Output

1
2
5
4
3


Input

lrlrr


Output

2
4
5
3
1

Note

In the first example, the positions of stones 1, 2, 3, 4, 5 will be <image>, respectively. So you should print the sequence: 3, 5, 4, 2, 1.""",
        r"""Squirrel Liss was enjoying a tranquil life in the forest, but suddenly, stones began falling from a mountain. Initially, Liss occupies the interval [0, 1]. As each of the n stones falls, Liss must decide to escape to the left or the right of the falling stone. The stones are numbered from 1 to n in the order they fall.

When a stone falls at the center of Liss's interval [k - d, k + d], she can:
- Escape to the left, changing her interval to [k - d, k]
- Escape to the right, changing her interval to [k, k + d]

You are provided with a string s of length n, where each character indicates Liss's escape direction:
- 'l' means Liss escapes to the left.
- 'r' means Liss escapes to the right.

Your task is to determine the order of the stones from left to right after all the stones have fallen.

### Input Section:
- A single line containing the string s (1 ≤ |s| ≤ 10^6). Each character in s will be either 'l' or 'r'.

### Output Section:
- Output n lines. On the i-th line, print the i-th stone's number from the left.

### Sample Input and Output:

**Example 1:**
Input: llrlr

Output:
3
5
4
2
1

**Example 2:**
Input: rrlll

Output:
1
2
5
4
3

**Example 3:**
Input: lrlrr

Output:
2
4
5
3
1

### Explanation:
In the first example, the stones fall and settle in the order 3, 5, 4, 2, 1. This sequence is printed as the output."""
    ),
    (
        r"""Don't you tell me what you think that I can be

If you say that Arkady is a bit old-fashioned playing checkers, you won't be right. There is also a modern computer game Arkady and his friends are keen on. We won't discuss its rules, the only feature important to this problem is that each player has to pick a distinct hero in the beginning of the game.

There are 2 teams each having n players and 2n heroes to distribute between the teams. The teams take turns picking heroes: at first, the first team chooses a hero in its team, after that the second team chooses a hero and so on. Note that after a hero is chosen it becomes unavailable to both teams.

The friends estimate the power of the i-th of the heroes as p_i. Each team wants to maximize the total power of its heroes. However, there is one exception: there are m pairs of heroes that are especially strong against each other, so when any team chooses a hero from such a pair, the other team must choose the other one on its turn. Each hero is in at most one such pair.

This is an interactive problem. You are to write a program that will optimally choose the heroes for one team, while the jury's program will play for the other team. Note that the jury's program may behave inefficiently, in this case you have to take the opportunity and still maximize the total power of your team. Formally, if you ever have chance to reach the total power of q or greater regardless of jury's program choices, you must get q or greater to pass a test.

Input

The first line contains two integers n and m (1 ≤ n ≤ 10^3, 0 ≤ m ≤ n) — the number of players in one team and the number of special pairs of heroes.

The second line contains 2n integers p_1, p_2, …, p_{2n} (1 ≤ p_i ≤ 10^3) — the powers of the heroes.

Each of the next m lines contains two integer a and b (1 ≤ a, b ≤ 2n, a ≠ b) — a pair of heroes that are especially strong against each other. It is guaranteed that each hero appears at most once in this list.

The next line contains a single integer t (1 ≤ t ≤ 2) — the team you are to play for. If t = 1, the first turn is yours, otherwise you have the second turn.

Hacks

In order to hack, use the format described above with one additional line. In this line output 2n distinct integers from 1 to 2n — the priority order for the jury's team. The jury's team will on each turn select the first possible hero from this list. Here possible means that it is not yet taken and does not contradict the rules about special pair of heroes.

Interaction

When it is your turn, print a single integer x (1 ≤ x ≤ 2n) — the index of the hero chosen by you. Note that you can't choose a hero previously chosen by either you of the other player, and you must follow the rules about special pairs of heroes.

When it is the other team's turn, read a line containing a single integer x (1 ≤ x ≤ 2n) — the index of the hero chosen by the other team. It is guaranteed that this index is not chosen before and that the other team also follows the rules about special pairs of heroes.

After the last turn you should terminate without printing anything.

After printing your choice do not forget to output end of line and flush the output. Otherwise you will get Idleness limit exceeded. To do this, use:

  * fflush(stdout) or cout.flush() in C++;
  * System.out.flush() in Java;
  * flush(output) in Pascal;
  * stdout.flush() in Python;
  * see documentation for other languages.



Jury's answer -1 instead of a valid choice means that you made an invalid turn. Exit immediately after receiving -1 and you will see Wrong answer verdict. Otherwise you can get an arbitrary verdict because your solution will continue to read from a closed stream.

Examples

Input


3 1
1 2 3 4 5 6
2 6
1

2

4

1


Output






6

5

3


Input


3 1
1 2 3 4 5 6
1 5
2
6

1

3


Output







5

4

2

Note

In the first example the first turn is yours. In example, you choose 6, the other team is forced to reply with 2. You choose 5, the other team chooses 4. Finally, you choose 3 and the other team choose 1.

In the second example you have the second turn. The other team chooses 6, you choose 5, forcing the other team to choose 1. Now you choose 4, the other team chooses 3 and you choose 2.""",
        r"""Arkady enjoys a modern computer game where each player selects a unique hero at the beginning. There are two teams, each with n players, and there are 2n heroes available for selection. The teams alternate turns picking heroes: the first team picks first, followed by the second team, and so on. Once a hero is selected, it becomes unavailable to both teams.

Each hero has a power value p_i, and both teams aim to maximize their total power. However, there are m pairs of heroes that are particularly strong against each other. If a team selects one hero from such a pair, the other team must pick the remaining hero in that pair on their turn. Each hero is part of at most one such pair.

This is an interactive problem where you need to write a program that selects heroes optimally for one team, while the other team is controlled by the jury's program. Note that the jury's program may not always make the best choices, so you must take advantage of this to maximize your team's power. If it is possible to achieve a total power of q or more regardless of the jury's choices, you must achieve at least q to pass the test.

++ Input ++

- The first line contains two integers n and m (1 ≤ n ≤ 1000, 0 ≤ m ≤ n) — the number of players in one team and the number of special pairs of heroes.
- The second line contains 2n integers p_1, p_2, ..., p_{2n} (1 ≤ p_i ≤ 1000) — the powers of the heroes.
- Each of the next m lines contains two integers a and b (1 ≤ a, b ≤ 2n, a ≠ b) — a pair of heroes that are especially strong against each other. Each hero appears at most once in this list.
- The next line contains a single integer t (1 ≤ t ≤ 2) — the team you are to play for. If t = 1, you have the first turn; if t = 2, you have the second turn.

++ Hacks ++

To hack, use the format described above with an additional line containing 2n distinct integers from 1 to 2n. This line represents the priority order for the jury's team. The jury's team will select the first available hero from this list during their turn.

++ Interaction ++

When it's your turn, print a single integer x (1 ≤ x ≤ 2n) — the index of the hero chosen by you. You cannot choose a previously chosen hero, and you must follow the special pair rules.

When it's the other team's turn, read a line containing a single integer x (1 ≤ x ≤ 2n) — the index of the hero chosen by the other team. The other team also follows the special pair rules.

Terminate your program after the last turn without printing anything further. After making your choice, flush the output to avoid the "Idleness limit exceeded" error.

- `fflush(stdout)` or `cout.flush()` in C++
- `System.out.flush()` in Java
- `flush(output)` in Pascal
- `stdout.flush()` in Python
- Refer to documentation for other languages

If the jury's answer is -1, it means you made an invalid move. Exit immediately after receiving -1 to see the "Wrong answer" verdict.

++ Examples ++

**Input:**
3 1
1 2 3 4 5 6
2 6
1

**Interaction:**
Your turn: 6
Jury's turn: 2
Your turn: 5
Jury's turn: 4
Your turn: 3

**Output:**
6
5
3

**Input:**
3 1
1 2 3 4 5 6
1 5
2

**Interaction:**
Jury's turn: 6
Your turn: 5
Jury's turn: 1
Your turn: 4
Jury's turn: 3
Your turn: 2

**Output:**
5
4
2

**Note:**
In the first example, you choose hero 6 first, the other team is forced to pick hero 2. You then choose hero 5, the other team picks hero 4. Finally, you pick hero 3 and the other team picks hero 1.

In the second example, the other team picks hero 6 first, you pick hero 5, forcing them to pick hero 1. You then pick hero 4, the other team picks hero 3, and you pick hero 2."""
    ),
    (r"""A number m of the form 10x + y is divisible by 7 if and only if x − 2y is divisible by 7. In other words, subtract twice the last digit
from the number formed by the remaining digits. Continue to do this until a number known to be divisible or not by 7 is obtained;
you can stop when this number has *at most* 2 digits because you are supposed to know if a number of at most 2 digits is divisible by 7 or not.

The original number is divisible by 7 if and only if the last number obtained using this procedure is divisible by 7.

Examples:

1 - `m = 371 -> 37 − (2×1) -> 37 − 2 = 35` ; thus, since 35 is divisible by 7, 371 is divisible by 7.

The number of steps to get the result is `1`.

2 - `m = 1603 -> 160 - (2 x 3) -> 154 -> 15 - 8 = 7` and 7 is divisible by 7.

3 - `m = 372 -> 37 − (2×2) -> 37 − 4 = 33` ; thus, since 33 is not divisible by 7, 372 is not divisible by 7.

The number of steps to get the result is `1`.

4 - `m = 477557101->47755708->4775554->477547->47740->4774->469->28` and 28 is divisible by 7, so is 477557101.

The number of steps is 7.

# Task:
Your task is to return to the function ```seven(m)``` (m integer >= 0) an array (or a pair, depending on the language) of numbers,
the first being the *last* number `m` with at most 2 digits obtained by your function (this last `m` will be divisible or not by 7), the second one being the number of steps to get the result.

## Forth Note:
Return on the stack `number-of-steps, last-number-m-with-at-most-2-digits `

## Examples:
```
seven(371) should return [35, 1]
seven(1603) should return [7, 2]
seven(477557101) should return [28, 7]
```""",
        r"""A number m of the form 10x + y is divisible by 7 if and only if x − 2y is divisible by 7. This means you subtract twice the last digit from the number formed by the remaining digits. Continue doing this until you get a number that is known to be divisible or not by 7. You can stop when this number has at most 2 digits because it’s easy to determine if a number with at most 2 digits is divisible by 7.

The original number is divisible by 7 if and only if the final number obtained through this procedure is divisible by 7.

### Examples:

1. `m = 371 -> 37 − (2×1) -> 37 − 2 = 35` ; since 35 is divisible by 7, 371 is divisible by 7.
   The number of steps to get the result is `1`.

2. `m = 1603 -> 160 - (2 x 3) -> 154 -> 15 - 8 = 7` and 7 is divisible by 7.
   The number of steps to get the result is `2`.

3. `m = 372 -> 37 − (2×2) -> 37 − 4 = 33` ; since 33 is not divisible by 7, 372 is not divisible by 7.
   The number of steps to get the result is `1`.

4. `m = 477557101 -> 47755708 -> 4775554 -> 477547 -> 47740 -> 4774 -> 469 -> 28` and 28 is divisible by 7, so 477557101 is divisible by 7.
   The number of steps to get the result is `7`.

### Task:
Your task is to write a function `seven(m)` that takes an integer m (m ≥ 0) and returns an array (or a pair, depending on the language) of two numbers. The first number in the array should be the last number m with at most 2 digits obtained by the function (this last m will be either divisible by 7 or not), and the second number should be the number of steps taken to reach this result.

### Note for Forth:
Return the result on the stack as `number-of-steps, last-number-m-with-at-most-2-digits`.

### Examples:
- `seven(371)` should return `[35, 1]`
- `seven(1603)` should return `[7, 2]`
- `seven(477557101)` should return `[28, 7]`"""
     ),
    (
        r"""> If you've finished this kata, you can try the [more difficult version](https://www.codewars.com/kata/5b256145a454c8a6990000b5).


## Taking a walk
A promenade is a way of uniquely representing a fraction by a succession of “left or right” choices.

For example, the promenade `"LRLL"` represents the fraction `4/7`.

Each successive choice (`L` or `R`) changes the value of the promenade by combining the values of the
promenade before the most recent left choice with the value before the most recent right choice. If the value before the most recent left choice was *l/m* and the value before the most recent right choice
was r/s then the new value will be *(l+r) / (m+s)*.

If there has never been a left choice we use *l=1* and *m=0*;
if there has never been a right choice we use *r=0* and *s=1*.


So let's take a walk.

* `""` An empty promenade has never had a left choice nor a right choice. Therefore we use *(l=1 and m=0)* and *(r=0 and s=1)*.
So the value of `""` is *(1+0) / (0+1) = 1/1*.
* `"L"`. Before the most recent left choice we have `""`, which equals *1/1*. There still has never been a right choice, so *(r=0 and s=1)*. So the value of `"L"` is *(1+0)/(1+1) = 1/2*
* `"LR"` = 2/3 as we use the values of `""` (before the left choice) and `"L"` (before the right choice)
* `"LRL"` = 3/5 as we use the values of `"LR"` and `"L"`
* `"LRLL"` = 4/7 as we use the values of `"LRL"` and `"L"`


Fractions are allowed to have a larger than b.


## Your task

Implement the `promenade` function, which takes an promenade as input (represented as a string), and returns
the corresponding fraction (represented as a tuple, containing the numerator and the denominator).

```Python
promenade("") == (1,1)
promenade("LR") == (2,3)
promenade("LRLL") == (4,7)
```
```Java
Return the Fraction as an int-Array:
promenade("") == [1,1]
promenade("LR") == [2,3]
promenade("LRLL") == [4,7]
```


*adapted from the 2016 British Informatics Olympiad*""",
        r"""A promenade is a unique way of representing a fraction through a series of "left or right" choices. For example, the promenade "LRLL" represents the fraction 4/7. Each successive choice (L or R) alters the value of the promenade by combining the values from before the most recent left and right choices. If the value before the most recent left choice was l/m and before the most recent right choice was r/s, the new value will be (l+r)/(m+s).

If there has never been a left choice, we use l=1 and m=0.
If there has never been a right choice, we use r=0 and s=1.

### Let's take a walk:

- `""`: An empty promenade has no left or right choices. Therefore, we use (l=1, m=0) and (r=0, s=1). The value of `""` is (1+0)/(0+1) = 1/1.
- `"L"`: Before the most recent left choice, we have `""`, which equals 1/1. There has never been a right choice, so (r=0, s=1). The value of `"L"` is (1+0)/(1+1) = 1/2.
- `"LR"`: This promenade is 2/3 as we use the values of `""` (before the left choice) and `"L"` (before the right choice).
- `"LRL"`: This promenade is 3/5 as we use the values of `"LR"` and `"L"`.
- `"LRLL"`: This promenade is 4/7 as we use the values of `"LRL"` and `"L"`.

### Your task

Implement the `promenade` function, which takes a promenade as input (represented as a string), and returns the corresponding fraction (represented as a tuple, containing the numerator and the denominator).

### Examples

```Python
promenade("") == (1,1)
promenade("LR") == (2,3)
promenade("LRLL") == (4,7)
```

```Java
Return the Fraction as an int-Array:
promenade("") == [1,1]
promenade("LR") == [2,3]
promenade("LRLL") == [4,7]
```"""
    ),
]


def get_prompt(question: str) -> str:
    random.shuffle(FEW_SHOTS)

    buf = "Your task is to rephrase the given prompt in another way that is significantly different from the original. However, it should still be clear and it is important to retain the original content, as the details are highly technical.\n"

    for i, o in FEW_SHOTS:
        buf += f"# Original:\n"
        buf += i + "\n"
        buf += f"# Rephrased:\n"
        buf += o + "\n"
        buf += "\n"

    buf += f"# Original:\n"
    buf += question + "\n"
    buf += f"# Rephrased:\n"
    return buf


def main(args):
    model = LLM(
        args.model,
        dtype=autodetect_dtype_str(),
        tensor_parallel_size=args.num_gpus,
    )
    dataset = datasets.load_dataset(args.dataset, split="train")

    if args.sample is not None:
        dataset = dataset.select(range(args.sample))

    indexed_prompts = []
    rephrases = [[] for _ in range(len(dataset))]
    for i, ex in enumerate(dataset):
        prompts = []
        for _ in range(len(ex["solutions"])):
            prompts.append((i, get_prompt(ex["question"])))

        indexed_prompts.extend(prompts)

    chunks = chunkify(indexed_prompts, args.batch_size)
    for chunk in tqdm(chunks):
        inputs = [x[1] for x in chunk]
        outputs = model.generate(
            inputs,
            sampling_params=SamplingParams(
                max_tokens=2048,
                temperature=0.45,
                top_p=0.9,
                repetition_penalty=1.1,
                stop=["# Original"],
            )
        )
        outputs = [o.outputs[0].text for o in outputs]
        for i, (idx, _) in enumerate(chunk):
            rephrases[idx].append(outputs[i])

    dataset = dataset.add_column("rephrased", rephrases)
    dataset.push_to_hub(args.push, private=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="bigcode/starcoder2-15b")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--push", type=str, required=True)
    args = parser.parse_args()
    random.seed(42)
    main(args)
