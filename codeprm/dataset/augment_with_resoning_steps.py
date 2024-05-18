import random
import datasets
from vllm import LLM, SamplingParams
from codeprm.model import autodetect_dtype
from codeprm.prompts import py_prompt
from codeprm.utils import chunkify
from tqdm import tqdm
import astdump


I_0 = '''"""
There are $n$ candy boxes in front of Tania. The boxes are arranged in a row from left to right, numbered from $1$ to $n$. The $i$-th box contains $r_i$ candies, candies have the color $c_i$ (the color can take one of three values ​​— red, green, or blue). All candies inside a single box have the same color (and it is equal to $c_i$).

Initially, Tanya is next to the box number $s$. Tanya can move to the neighbor box (that is, with a number that differs by one) or eat candies in the current box. Tanya eats candies instantly, but the movement takes one second.

If Tanya eats candies from the box, then the box itself remains in place, but there is no more candies in it. In other words, Tanya always eats all the candies from the box and candies in the boxes are not refilled.

It is known that Tanya cannot eat candies of the same color one after another (that is, the colors of candies in two consecutive boxes from which she eats candies are always different). In addition, Tanya's appetite is constantly growing, so in each next box from which she eats candies, there should be strictly more candies than in the previous one.

Note that for the first box from which Tanya will eat candies, there are no restrictions on the color and number of candies.

Tanya wants to eat at least $k$ candies. What is the minimum number of seconds she will need? Remember that she eats candies instantly, and time is spent only on movements.


-----Input-----

The first line contains three integers $n$, $s$ and $k$ ($1 \le n \le 50$, $1 \le s \le n$, $1 \le k \le 2000$) — number of the boxes, initial position of Tanya and lower bound on number of candies to eat. The following line contains $n$ integers $r_i$ ($1 \le r_i \le 50$) — numbers of candies in the boxes. The third line contains sequence of $n$ letters 'R', 'G' and 'B', meaning the colors of candies in the correspondent boxes ('R' for red, 'G' for green, 'B' for blue). Recall that each box contains candies of only one color. The third line contains no spaces.


-----Output-----

Print minimal number of seconds to eat at least $k$ candies. If solution doesn't exist, print "-1".


-----Examples-----
Input
5 3 10
1 2 3 4 5
RGBRR

Output
4

Input
2 1 15
5 6
RG

Output
-1



-----Note-----

The sequence of actions of Tanya for the first example:

  move from the box $3$ to the box $2$;  eat candies from the box $2$;  move from the box $2$ to the box $3$;  eat candy from the box $3$;  move from the box $3$ to the box $4$;  move from the box $4$ to the box $5$;  eat candies from the box $5$. 

Since Tanya eats candy instantly, the required time is four seconds.
"""
inf = 10000
(n, s, k) = map(int, input().split())
a = list(map(int, input().split()))
b = list(input())
for i in range(n):
	if b[i] == 'R':
		b[i] = 0
	elif b[i] == 'G':
		b[i] = 1
	else:
		b[i] = 2
boxes = [[a[i], b[i], i] for i in range(n)]
boxes.sort()
l = boxes[-1][0] * n + 1
s -= 1
dp = [[[inf, s, -1] for j in range(l)] for i in range(3)]
if l < k:
	print(-1)
	return
dp[0][0][0] = 0
dp[1][0][0] = 0
dp[2][0][0] = 0
for i in range(n):
	pos = boxes[i][2]
	clr = boxes[i][1]
	cnt = boxes[i][0]
	for j in range(l - cnt):
		for c in range(3):
			if c == clr:
				continue
			if dp[clr][j + cnt][0] > dp[c][j][0] + abs(dp[c][j][1] - pos) and cnt > dp[c][j][2]:
				dp[clr][j + cnt][0] = dp[c][j][0] + abs(dp[c][j][1] - pos)
				dp[clr][j + cnt][1] = pos
				dp[clr][j + cnt][2] = cnt
ans = min(dp[0][k][0], min(dp[1][k][0], dp[2][k][0]))
for i in range(k, l):
	ans = min(min(ans, dp[0][i][0]), min(dp[1][i][0], dp[2][i][0]))
if ans < inf:
	print(ans)
else:
	print(-1)'''
O_0 = '''inf = 10000

# Reading input values for number of boxes, starting position, and minimum candies to eat
(n, s, k) = map(int, input().split())

# Reading the number of candies in each box
a = list(map(int, input().split()))

# Reading the color of candies in each box
b = list(input())

# Converting candy colors to numerical values for easier processing
for i in range(n):
	if b[i] == 'R':
		b[i] = 0
	elif b[i] == 'G':
		b[i] = 1
	else:
		b[i] = 2

# Creating a list of boxes with candy count, color, and position
boxes = [[a[i], b[i], i] for i in range(n)]

# Sorting boxes based on the number of candies (first element of each sublist)
boxes.sort()

# Defining the upper bound on the number of candies Tanya can eat
l = boxes[-1][0] * n + 1

# Adjusting starting position to be zero-based
s -= 1

# Initializing the DP table with a large number (inf)
# dp[color][candies_eaten][0] = minimum time
# dp[color][candies_eaten][1] = position of the last box eaten from
# dp[color][candies_eaten][2] = number of candies in the last box eaten from
dp = [[[inf, s, -1] for j in range(l)] for i in range(3)]

# If the total possible candies are less than the required candies, print -1
if l < k:
	print(-1)
	return

# Initializing DP for the base case: 0 candies eaten, no time spent
dp[0][0][0] = 0
dp[1][0][0] = 0
dp[2][0][0] = 0

# Iterating through all the boxes
for i in range(n):
	pos = boxes[i][2]
	clr = boxes[i][1]
	cnt = boxes[i][0]

	# Iterating through all possible amounts of candies eaten
	for j in range(l - cnt):
		# Iterating through all candy colors
		for c in range(3):
			# Skipping if the color is the same as the current box color
			if c == clr:
				continue

			# Updating the DP table if the conditions are met
			if dp[clr][j + cnt][0] > dp[c][j][0] + abs(dp[c][j][1] - pos) and cnt > dp[c][j][2]:
				dp[clr][j + cnt][0] = dp[c][j][0] + abs(dp[c][j][1] - pos)
				dp[clr][j + cnt][1] = pos
				dp[clr][j + cnt][2] = cnt

# Finding the minimum time required to eat at least k candies across all colors
ans = min(dp[0][k][0], min(dp[1][k][0], dp[2][k][0]))
for i in range(k, l):
	ans = min(min(ans, dp[0][i][0]), min(dp[1][i][0], dp[2][i][0]))

# Printing the result if a valid answer exists, otherwise print -1
if ans < inf:
	print(ans)
else:
	print(-1)'''
I_1 = '''"""
If you visit Aizu Akabeko shrine, you will find a unique paper fortune on which a number with more than one digit is written.

Each digit ranges from 1 to 9 (zero is avoided because it is considered a bad omen in this shrine). Using this string of numeric values, you can predict how many years it will take before your dream comes true. Cut up the string into more than one segment and compare their values. The difference between the largest and smallest value will give you the number of years before your wish will be fulfilled. Therefore, the result varies depending on the way you cut up the string. For example, if you are given a string 11121314 and divide it into segments, say, as 1,11,21,3,14, then the difference between the largest and smallest is 21 - 1 = 20. Another division 11,12,13,14 produces 3 (i.e. 14 - 11) years. Any random division produces a game of luck. However, you can search the minimum number of years using a program.

Given a string of numerical characters, write a program to search the minimum years before your wish will be fulfilled.



Input

The input is given in the following format.


n


An integer n is given. Its number of digits is from 2 to 100,000, and each digit ranges from 1 to 9.

Output

Output the minimum number of years before your wish will be fulfilled.

Examples

Input

11121314


Output

3


Input

123125129


Output

6


Input

119138


Output

5
"""
def sub(maxs, mins):
	for i in range(len(maxs)):
		if maxs[i] != mins[i]:
			if i == len(maxs) - 1:
				return int(maxs[i]) - int(mins[i])
			if i == len(maxs) - 2:
				return int(maxs[i:i + 2]) - int(mins[i:i + 2])
			return 10
	return 0

def checkEqual(S):
	ans = 8
	for k in range(1, len(S)):
		if len(S) % k != 0:
			continue
		mins = maxs = S[0:k]
		for s in range(0, len(S), k):
			maxs = max(maxs, S[s:s + k])
			mins = min(mins, S[s:s + k])
		ans = min(ans, sub(maxs, mins))
	return ans

def check12(S):
	maxv = 0
	minv = 10
	p = 0
	while p < len(S):
		v = int(S[p])
		if S[p] == '1' and p + 1 < len(S):
			v = 10 + int(S[p + 1])
			p += 1
		maxv = max(maxv, v)
		minv = min(minv, v)
		p += 1
	return maxv - minv
S = input()
print(min(checkEqual(S), check12(S)))'''
O_1 = '''# Function to calculate the difference between the digits of two strings
def sub(maxs, mins):
    # Loop through each digit in maxs and mins
    for i in range(len(maxs)):
        # Check if the current digit in maxs and mins are different
        if maxs[i] != mins[i]:
            # Return the difference of the digits or the last two digits if necessary
            if i == len(maxs) - 1:
                return int(maxs[i]) - int(mins[i])
            if i == len(maxs) - 2:
                return int(maxs[i:i + 2]) - int(mins[i:i + 2])
            return 10
    # If all digits are the same, return 0
    return 0

# Function to check the equality of segments in the string
def checkEqual(S):
    ans = 8
    # Loop through possible segment lengths
    for k in range(1, len(S)):
        if len(S) % k != 0:
            continue
        # Initialize min and max segments
        mins = maxs = S[0:k]
        # Loop through the string in segments of length k
        for s in range(0, len(S), k):
            maxs = max(maxs, S[s:s + k])
            mins = min(mins, S[s:s + k])
        # Update the answer with the minimum difference found
        ans = min(ans, sub(maxs, mins))
    return ans

# Function to check the difference between the maximum and minimum values in the string
def check12(S):
    maxv = 0
    minv = 10
    p = 0
    # Loop through each character in the string
    while p < len(S):
        v = int(S[p])
        # Treat '1x' as a two-digit number if applicable
        if S[p] == '1' and p + 1 < len(S):
            v = 10 + int(S[p + 1])
            p += 1
        # Update the max and min values
        maxv = max(maxv, v)
        minv = min(minv, v)
        p += 1
    return maxv - minv

# Read the input string
S = input()
# Calculate and print the minimum number of years before the wish is fulfilled
print(min(checkEqual(S), check12(S)))'''
I_2 = '''"""
You have a simple undirected graph consisting of $n$ vertices and $m$ edges. The graph doesn't contain self-loops, there is at most one edge between a pair of vertices. The given graph can be disconnected.

Let's make a definition.

Let $v_1$ and $v_2$ be two some nonempty subsets of vertices that do not intersect. Let $f(v_{1}, v_{2})$ be true if and only if all the conditions are satisfied:  There are no edges with both endpoints in vertex set $v_1$.  There are no edges with both endpoints in vertex set $v_2$.  For every two vertices $x$ and $y$ such that $x$ is in $v_1$ and $y$ is in $v_2$, there is an edge between $x$ and $y$. 

Create three vertex sets ($v_{1}$, $v_{2}$, $v_{3}$) which satisfy the conditions below;  All vertex sets should not be empty.  Each vertex should be assigned to only one vertex set.  $f(v_{1}, v_{2})$, $f(v_{2}, v_{3})$, $f(v_{3}, v_{1})$ are all true. 

Is it possible to create such three vertex sets? If it's possible, print matching vertex set for each vertex.


-----Input-----

The first line contains two integers $n$ and $m$ ($3 \le n \le 10^{5}$, $0 \le m \le \text{min}(3 \cdot 10^{5}, \frac{n(n-1)}{2})$) — the number of vertices and edges in the graph.

The $i$-th of the next $m$ lines contains two integers $a_{i}$ and $b_{i}$ ($1 \le a_{i} \lt b_{i} \le n$) — it means there is an edge between $a_{i}$ and $b_{i}$. The graph doesn't contain self-loops, there is at most one edge between a pair of vertices. The given graph can be disconnected.


-----Output-----

If the answer exists, print $n$ integers. $i$-th integer means the vertex set number (from $1$ to $3$) of $i$-th vertex. Otherwise, print $-1$.

If there are multiple answers, print any.


-----Examples-----
Input
6 11
1 2
1 3
1 4
1 5
1 6
2 4
2 5
2 6
3 4
3 5
3 6

Output
1 2 2 3 3 3 
Input
4 6
1 2
1 3
1 4
2 3
2 4
3 4

Output
-1



-----Note-----

In the first example, if $v_{1} = \{ 1 \}$, $v_{2} = \{ 2, 3 \}$, and $v_{3} = \{ 4, 5, 6 \}$ then vertex sets will satisfy all conditions. But you can assign vertices to vertex sets in a different way; Other answers like "2 3 3 1 1 1" will be accepted as well. [Image] 

In the second example, it's impossible to make such vertex sets.
"""
(n, m) = map(int, input().split())
d = {}
a = [['0'] for i in range(n + 1)]
for i in range(m):
	(u, v) = map(int, input().split())
	a[u].append(str(v) + '*')
	a[v].append(str(u) + '*')
count = 1
for i in range(1, n + 1):
	if len(a[i]) == 1:
		print('-1')
		return
	a[i].sort()
for i in range(1, n + 1):
	a[i] = ''.join(a[i])
for i in range(1, n + 1):
	if a[i] not in d:
		d[a[i]] = count
		count += 1
if len(d) != 3:
	print('-1')
	return
for i in range(1, n + 1):
	print(d[a[i]], end=' ')'''
O_2 = '''# Read the number of vertices (n) and edges (m)
(n, m) = map(int, input().split())

# Initialize a dictionary to store unique edge patterns
d = {}

# Initialize a list to store adjacency information for each vertex
a = [['0'] for i in range(n + 1)]

# Read each edge and store adjacency information
for i in range(m):
	(u, v) = map(int, input().split())
	a[u].append(str(v) + '*')
	a[v].append(str(u) + '*')

# Initialize a counter to assign unique identifiers to edge patterns
count = 1

# Check if any vertex has no edges and sort adjacency information
for i in range(1, n + 1):
	if len(a[i]) == 1:
		print('-1')  # If a vertex has no edges, the conditions can't be satisfied
		return
	a[i].sort()  # Sort adjacency list for consistent pattern comparison

# Concatenate adjacency information into strings for pattern recognition
for i in range(1, n + 1):
	a[i] = ''.join(a[i])

# Assign unique identifiers to each unique edge pattern
for i in range(1, n + 1):
	if a[i] not in d:
		d[a[i]] = count
		count += 1

# Check if there are exactly 3 unique edge patterns
if len(d) != 3:
	print('-1')  # If not exactly 3 patterns, conditions can't be satisfied
	return

# Output the unique identifier for each vertex based on its edge pattern
for i in range(1, n + 1):
	print(d[a[i]], end=' ')'''
I_3 = '''"""
Appleman has a tree with n vertices. Some of the vertices (at least one) are colored black and other vertices are colored white.

Consider a set consisting of k (0 ≤ k < n) edges of Appleman's tree. If Appleman deletes these edges from the tree, then it will split into (k + 1) parts. Note, that each part will be a tree with colored vertices.

Now Appleman wonders, what is the number of sets splitting the tree in such a way that each resulting part will have exactly one black vertex? Find this number modulo 1000000007 (109 + 7).

Input

The first line contains an integer n (2 ≤ n ≤ 105) — the number of tree vertices. 

The second line contains the description of the tree: n - 1 integers p0, p1, ..., pn - 2 (0 ≤ pi ≤ i). Where pi means that there is an edge connecting vertex (i + 1) of the tree and vertex pi. Consider tree vertices are numbered from 0 to n - 1.

The third line contains the description of the colors of the vertices: n integers x0, x1, ..., xn - 1 (xi is either 0 or 1). If xi is equal to 1, vertex i is colored black. Otherwise, vertex i is colored white.

Output

Output a single integer — the number of ways to split the tree modulo 1000000007 (109 + 7).

Examples

Input

3
0 0
0 1 1


Output

2


Input

6
0 1 1 0 4
1 1 0 0 1 0


Output

1


Input

10
0 1 2 1 4 4 4 0 8
0 0 0 1 0 1 1 0 0 1


Output

27
"""
n = int(input())
edges = [int(x) for x in input().split()]
color = [int(x) for x in input().split()]
graph = [[] for _ in range(n)]
for (a, b) in enumerate(edges):
	graph[a + 1].append(b)
	graph[b].append(a + 1)
dp = [[0] * 2 for _ in range(n)]
visited = [0] * n
stack = [0]
while stack:
	v = stack[-1]
	visited[v] = -1
	cn = 0
	for u in graph[v]:
		if visited[u] != 0:
			continue
		else:
			cn += 1
			stack.append(u)
	if not cn:
		dp[v][0] = 1
		dp[v][1] = 0
		for u in graph[v]:
			if visited[u] == -1:
				continue
			dp[v][1] *= dp[u][0]
			dp[v][1] += dp[v][0] * dp[u][1]
			dp[v][0] *= dp[u][0]
			dp[v][1] %= 1000000007
			dp[v][0] %= 1000000007
		if color[v] == 1:
			dp[v][1] = dp[v][0]
		else:
			dp[v][0] += dp[v][1]
			dp[v][0] %= 1000000007
		visited[v] = 1
		stack.pop()
ans = dp[0][1]
print(ans)'''
O_3 = '''# Read the number of vertices in the tree
n = int(input())

# Read the edges connecting the vertices
edges = [int(x) for x in input().split()]

# Read the colors of the vertices
color = [int(x) for x in input().split()]

# Initialize an adjacency list to represent the tree
graph = [[] for _ in range(n)]

# Populate the adjacency list with edges
for (a, b) in enumerate(edges):
	graph[a + 1].append(b)
	graph[b].append(a + 1)

# Initialize dp array to store the number of ways to split the tree
dp = [[0] * 2 for _ in range(n)]

# Initialize visited array to track visited nodes
visited = [0] * n

# Initialize stack for DFS traversal
stack = [0]

# Perform DFS traversal to calculate dp values
while stack:
	v = stack[-1]
	visited[v] = -1
	cn = 0

	# Traverse the adjacent nodes
	for u in graph[v]:
		if visited[u] != 0:
			continue
		else:
			cn += 1
			stack.append(u)

	# If no more nodes to traverse, process the current node
	if not cn:
		dp[v][0] = 1
		dp[v][1] = 0

		# Calculate dp values based on child nodes
		for u in graph[v]:
			if visited[u] == -1:
				continue
			dp[v][1] *= dp[u][0]
			dp[v][1] += dp[v][0] * dp[u][1]
			dp[v][0] *= dp[u][0]
			dp[v][1] %= 1000000007
			dp[v][0] %= 1000000007

		# Update dp values based on the color of the current node
		if color[v] == 1:
			dp[v][1] = dp[v][0]
		else:
			dp[v][0] += dp[v][1]
			dp[v][0] %= 1000000007

		# Mark the current node as visited and pop from stack
		visited[v] = 1
		stack.pop()

# The final answer is the number of ways to split the tree with root having exactly one black vertex
ans = dp[0][1]
print(ans)'''
I_4 = '''"""
You are playing a game of Jongmah. You don't need to know the rules to solve this problem. You have n tiles in your hand. Each tile has an integer between 1 and m written on it.

To win the game, you will need to form some number of triples. Each triple consists of three tiles, such that the numbers written on the tiles are either all the same or consecutive. For example, 7, 7, 7 is a valid triple, and so is 12, 13, 14, but 2,2,3 or 2,4,6 are not. You can only use the tiles in your hand to form triples. Each tile can be used in at most one triple.

To determine how close you are to the win, you want to know the maximum number of triples you can form from the tiles in your hand.

Input

The first line contains two integers integer n and m (1 ≤ n, m ≤ 10^6) — the number of tiles in your hand and the number of tiles types.

The second line contains integers a_1, a_2, …, a_n (1 ≤ a_i ≤ m), where a_i denotes the number written on the i-th tile.

Output

Print one integer: the maximum number of triples you can form.

Examples

Input

10 6
2 3 3 3 4 4 4 5 5 6


Output

3


Input

12 6
1 5 3 3 3 4 3 5 3 2 3 3


Output

3


Input

13 5
1 1 5 1 2 3 3 2 4 2 3 4 5


Output

4

Note

In the first example, we have tiles 2, 3, 3, 3, 4, 4, 4, 5, 5, 6. We can form three triples in the following way: 2, 3, 4; 3, 4, 5; 4, 5, 6. Since there are only 10 tiles, there is no way we could form 4 triples, so the answer is 3.

In the second example, we have tiles 1, 2, 3 (7 times), 4, 5 (2 times). We can form 3 triples as follows: 1, 2, 3; 3, 3, 3; 3, 4, 5. One can show that forming 4 triples is not possible.
"""
from collections import Counter
(n, m) = map(int, input().split())
B = list(map(int, input().split()))
cnt = Counter(B)
A = sorted(cnt.keys())
n = len(A)
dp = [[0] * 3 for _ in range(3)]
for (i, a) in enumerate(A):
	dp2 = [[0] * 3 for _ in range(3)]
	for x in range(1 if i >= 2 and a - 2 != A[i - 2] else 3):
		for y in range(1 if i >= 1 and a - 1 != A[i - 1] else 3):
			for z in range(3):
				if x + y + z <= cnt[a]:
					dp2[y][z] = max(dp2[y][z], dp[x][y] + z + (cnt[a] - x - y - z) // 3)
	dp = dp2
print(dp[0][0])'''
O_4 = '''from collections import Counter

# Read the number of tiles and the number of tile types
(n, m) = map(int, input().split())

# Read the numbers written on the tiles
B = list(map(int, input().split()))

# Count the occurrences of each tile number
cnt = Counter(B)

# Sort the tile numbers
A = sorted(cnt.keys())

# Get the length of the sorted tile numbers list
n = len(A)

# Initialize a 3x3 DP table to store the maximum number of triples that can be formed
dp = [[0] * 3 for _ in range(3)]

# Iterate through the sorted tile numbers
for (i, a) in enumerate(A):
    # Initialize a new 3x3 DP table for the current tile number
    dp2 = [[0] * 3 for _ in range(3)]

    # Iterate through the possible combinations of tiles for the current tile number
    for x in range(1 if i >= 2 and a - 2 != A[i - 2] else 3):
        for y in range(1 if i >= 1 and a - 1 != A[i - 1] else 3):
            for z in range(3):
                # If the number of tiles for the current combination is less than or equal to the count of the current tile number
                if x + y + z <= cnt[a]:
                    # Update the maximum number of triples that can be formed for the current combination
                    dp2[y][z] = max(dp2[y][z], dp[x][y] + z + (cnt[a] - x - y - z) // 3)

    # Update the main DP table with the results for the current tile number
    dp = dp2

# Print the maximum number of triples that can be formed
print(dp[0][0])'''
I_5 = '''"""
Taro is planning a long trip by train during the summer vacation. However, in order for Taro, who is a high school student, to travel as far as possible during the summer vacation, which has only one month, he cannot make a good plan unless he finds the cheapest and the fastest way. Let's create a program to help Taro's plan so that he can enjoy a wonderful trip.


<image>



Create a program that outputs the minimum amount or the shortest time in response to inquiries by inputting track information and the number of stations.



Input

A sequence of multiple datasets is given as input. The end of the input is indicated by two lines of zeros. Each dataset is given in the following format:


n m
a1 b1 cost1 time1
a2 b2 cost2 time2
::
an bn costn timen
k
p1 q1 r1
p2 q2 r2
::
pk qk rk


The first line gives the number of track information n (1 ≤ n ≤ 3000) and the number of stations m (1 ≤ m ≤ 100).

The following n lines give information on the i-th line. As information on each line, the numbers ai, bi (1 ≤ ai, bi ≤ m) of the two stations connecting the lines, the toll costi (1 ≤ costi ≤ 1000), and the travel time timei (1 ≤ timei ≤ 1000) are given. I will. However, each station shall be numbered in order from 1 to m. If ai and bi are connected by railroad tracks, both ai to bi and bi to ai can be moved at the same rate and time.

The following line is given the number of queries k (1 ≤ k ≤ 200). The next k line is given the i-th query. For each query, the departure station pi, the arrival station qi, and the type of value to output ri (0 or 1) are given. Inquiries must have a route.

The number of datasets does not exceed 50.

Output

Outputs the minimum amount or minimum time on one line for each data set. When ri is 0, the minimum amount is output, and when ri is 1, the minimum time is output.

Example

Input

6 5
1 2 200 10
1 4 400 15
1 3 250 25
2 4 100 10
4 5 150 20
3 5 300 20
2
1 5 0
1 5 1
0 0


Output

450
35
"""
def warshall_floyd(v_count: int, matrix: list) -> list:
	for i in range(v_count):
		for (j, c2) in enumerate((row[i] for row in matrix)):
			for (k, (c1, c3)) in enumerate(zip(matrix[j], matrix[i])):
				if c1 > c2 + c3:
					matrix[j][k] = c2 + c3
	return matrix
while True:
	(e_count, v_count) = map(int, input().split())
	if not e_count:
		break
	inf = float('inf')
	(edges_cost, edges_time) = ([[inf] * v_count for _ in [0] * v_count], [[inf] * v_count for _ in [0] * v_count])
	for _ in [0] * e_count:
		(a, b, cost, time) = map(int, input().split())
		(a, b) = (a - 1, b - 1)
		edges_cost[a][b] = cost
		edges_cost[b][a] = cost
		edges_time[a][b] = time
		edges_time[b][a] = time
	warshall_floyd(v_count, edges_cost)
	warshall_floyd(v_count, edges_time)
	for _ in [0] * int(input()):
		(p, q, r) = map(int, input().split())
		print((edges_time if r else edges_cost)[p - 1][q - 1])'''
O_5 = '''
# Function to perform the Warshall-Floyd algorithm for finding shortest paths in a weighted graph
def warshall_floyd(v_count: int, matrix: list) -> list:
    # Iterate through each vertex as an intermediate vertex
    for i in range(v_count):
        # Iterate through each pair of vertices (j, k)
        for (j, c2) in enumerate((row[i] for row in matrix)):
            for (k, (c1, c3)) in enumerate(zip(matrix[j], matrix[i])):
                # Update the distance if a shorter path is found
                if c1 > c2 + c3:
                    matrix[j][k] = c2 + c3
    # Return the updated distance matrix
    return matrix

while True:
    # Read the number of edges and vertices
    (e_count, v_count) = map(int, input().split())
    # Break the loop if the input indicates the end
    if not e_count:
        break
    # Initialize infinite cost and time matrices
    inf = float('inf')
    (edges_cost, edges_time) = ([[inf] * v_count for _ in [0] * v_count], [[inf] * v_count for _ in [0] * v_count])
    # Read edge information
    for _ in [0] * e_count:
        (a, b, cost, time) = map(int, input().split())
        # Convert to zero-based index
        (a, b) = (a - 1, b - 1)
        # Set the cost and time for the bidirectional edges
        edges_cost[a][b] = cost
        edges_cost[b][a] = cost
        edges_time[a][b] = time
        edges_time[b][a] = time
    # Apply Warshall-Floyd algorithm to find all pairs shortest path for cost
    warshall_floyd(v_count, edges_cost)
    # Apply Warshall-Floyd algorithm to find all pairs shortest path for time
    warshall_floyd(v_count, edges_time)
    # Process each query
    for _ in [0] * int(input()):
        (p, q, r) = map(int, input().split())
        # Output the minimum cost or time based on the query type
        print((edges_time if r else edges_cost)[p - 1][q - 1])'''
I_6 = '''"""
Creatnx has $n$ mirrors, numbered from $1$ to $n$. Every day, Creatnx asks exactly one mirror "Am I beautiful?". The $i$-th mirror will tell Creatnx that he is beautiful with probability $\frac{p_i}{100}$ for all $1 \le i \le n$.

Creatnx asks the mirrors one by one, starting from the $1$-st mirror. Every day, if he asks $i$-th mirror, there are two possibilities:  The $i$-th mirror tells Creatnx that he is beautiful. In this case, if $i = n$ Creatnx will stop and become happy, otherwise he will continue asking the $i+1$-th mirror next day;  In the other case, Creatnx will feel upset. The next day, Creatnx will start asking from the $1$-st mirror again. 

You need to calculate the expected number of days until Creatnx becomes happy.

This number should be found by modulo $998244353$. Formally, let $M = 998244353$. It can be shown that the answer can be expressed as an irreducible fraction $\frac{p}{q}$, where $p$ and $q$ are integers and $q \not \equiv 0 \pmod{M}$. Output the integer equal to $p \cdot q^{-1} \bmod M$. In other words, output such an integer $x$ that $0 \le x < M$ and $x \cdot q \equiv p \pmod{M}$.


-----Input-----

The first line contains one integer $n$ ($1\le n\le 2\cdot 10^5$) — the number of mirrors.

The second line contains $n$ integers $p_1, p_2, \ldots, p_n$ ($1 \leq p_i \leq 100$).


-----Output-----

Print the answer modulo $998244353$ in a single line.


-----Examples-----
Input
1
50

Output
2

Input
3
10 20 50

Output
112



-----Note-----

In the first test, there is only one mirror and it tells, that Creatnx is beautiful with probability $\frac{1}{2}$. So, the expected number of days until Creatnx becomes happy is $2$.
"""
n = int(input())
a = list(map(int, input().split()))
mod = 998244353
rng = 1100
inv = [0, 1]
for i in range(2, rng):
	inv.append(pow(i, mod - 2, mod))
acc = [0] * n
acc[-1] = 100 * inv[a[-1]]
for i in range(n - 1)[::-1]:
	acc[i] = acc[i + 1] * 100 * inv[a[i]] % mod
print(sum(acc) % mod)'''
O_6 = '''# Read the number of mirrors
n = int(input())
# Read the probabilities for each mirror
a = list(map(int, input().split()))
# Define the modulo value
mod = 998244353
# Define a range for calculating inverses
rng = 1100
# Initialize the list of modular inverses
inv = [0, 1]
# Compute the modular inverses for numbers from 2 to rng-1
for i in range(2, rng):
    inv.append(pow(i, mod - 2, mod))
# Initialize a list to accumulate probabilities
acc = [0] * n
# Set the probability for the last mirror
acc[-1] = 100 * inv[a[-1]]
# Calculate the cumulative probabilities for each mirror in reverse order
for i in range(n - 1)[::-1]:
    acc[i] = acc[i + 1] * 100 * inv[a[i]] % mod
# Output the sum of the accumulated probabilities modulo mod
print(sum(acc) % mod)'''

FEW_SHOTS = [
    (I_0, O_0),
    (I_1, O_1),
    (I_2, O_2),
    (I_3, O_3),
    (I_4, O_4),
    (I_5, O_5),
    (I_6, O_6),
]


def check_astmatch(inp, out):
    try:
        outdump = astdump.indented(out, printres=False)
        inpdump = astdump.indented(inp, printres=False)
    except Exception as e:
        print(f"Failed to dump AST: {e}")
        return False

    if outdump is None or inpdump is None:
        print("Failed to dump AST. None returned.")
        return False

    outdump = outdump.split("\n")
    inpdump = inpdump.split("\n")

    for j, (i, o) in enumerate(zip(inpdump, outdump)):
        if i != o:
            print(f"Failed at {j}: {i} != {o}")
            return False

    return True


def add_reasoning_steps_prompt(tokenizer, code: str) -> str:
    system = "You are an exceptional code reasoner. Your job is to take uncommented algorithmic Python code, and to insert reasoning steps as single-line comments before each important step in the algorithm."
    turn_req = "Can you please add reasoning steps to the following code? IMPORTANT: DO NOT CHANGE ANY CODE, ANY CHANGES TO THE CODE ARE STRICTLY PROHOBITED. Your job is to add a line-comment before each important steps detailing the step in words."
    turn_resp = "Sure, I can add reasoning steps to the code. Here's the modified code with comments:"
    random.shuffle(FEW_SHOTS)
    turns = []
    for user, ai in FEW_SHOTS:
        turns.extend(
            [
                {
                    "role": "user",
                    "content": f"{turn_req}\n```py\n{user}\n```",
                },
                {
                    "role": "assistant",
                    "content": f"{turn_resp}\n```py\n{ai}\n```",
                },
            ]
        )
    prompt = [
        {
            "role": "system",
            "content": system,
        },
        *turns,
        {
            "role": "user",
            "content": f"{turn_req}\n```py\n{code}\n```",
        }
    ]
    formatted = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True)
    post = formatted + "Sure, I can add reasoning steps to the code. Here's the modified code with comments:\n```py"
    return post


def main(args):
    model = LLM(
        args.model,
        dtype=autodetect_dtype(),
        tensor_parallel_size=args.num_gpus,
        enable_prefix_caching=not args.no_prefix_caching,
    )
    tokenizer = model.get_tokenizer()
    dataset = datasets.load_dataset(args.dataset, split="train")

    if args.sample:
        dataset = dataset.select(range(args.sample))

    indexed_prompts = []
    for i, ex in tqdm(enumerate(dataset), desc="Processing dataset"):
        question = ex["question"]
        for sol in ex["solutions"]:
            indexed_prompts.append(
                (
                    i,
                    sol,
                    add_reasoning_steps_prompt(
                        tokenizer, py_prompt(question, sol))
                )
            )

    with_steps = [[] for _ in range(len(dataset))]
    chunks = chunkify(indexed_prompts, args.batch_size)

    # some stats
    num_does_match = 0
    num_processed = 0

    for chunk in tqdm(chunks, desc=f"Generating reasoning steps"):
        print(f"Current match rate: {num_does_match / (num_processed + 1e-6)}")
        inputs = [inp for _, _, inp in chunk]
        outputs = model.generate(
            inputs,
            sampling_params=SamplingParams(
                max_tokens=4096,
                temperature=0.0,
                stop="```",
            ),
        )
        outputs = [o.outputs[0].text for o in outputs]
        for (i, inp_sol, _), out in zip(chunk, outputs):
            num_processed += 1
            if check_astmatch(inp_sol, out):
                with_steps[i].append(out)
                num_does_match += 1

    dataset = dataset.add_column("reasoning_steps", with_steps)
    dataset.push_to_hub(args.push, private=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="Qwen/CodeQwen1.5-7B-Chat")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--dataset", type=str,
                        default="cassanof/taco_cleaned_exec_filtered_max75_v3")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--push", type=str, required=True)
    parser.add_argument(
        "--no_prefix_caching",
        dest="no_prefix_caching",
        action="store_true",
        help="Disable prefix caching",
    )
    args = parser.parse_args()
    main(args)
