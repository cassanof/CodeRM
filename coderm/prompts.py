from typing import Dict, List, Union


Message = Dict[str, str]
Conversation = List[Message]
Prompt = Union[str, Conversation]


def py_prompt(question: str, code=""):
    # escape any triple quotes in the question
    question = question.replace('"""', r'\"""')
    return f'''"""
{question}
"""
{code}'''.strip()


def py_prompt_evolve(before: str, after=""):
    return f'''{before}
# ==== EVOLVED CODE ====
{after}'''.strip()


# few-shots taken from: https://github.com/LiveCodeBench/LiveCodeBench/tree/45db82290ef929811e21d4cdb67c8db43010c1e0/lcb_runner/prompts/few_shot_examples/generation

LCB_IO_FEWSHOT = [
    {
        "question": "You have $n$ gifts and you want to give all of them to children. Of course, you don't want to offend anyone, so all gifts should be equal between each other. The $i$-th gift consists of $a_i$ candies and $b_i$ oranges.\n\nDuring one move, you can choose some gift $1 \\le i \\le n$ and do one of the following operations:\n\n  eat exactly one candy from this gift (decrease $a_i$ by one);  eat exactly one orange from this gift (decrease $b_i$ by one);  eat exactly one candy and exactly one orange from this gift (decrease both $a_i$ and $b_i$ by one). \n\nOf course, you can not eat a candy or orange if it's not present in the gift (so neither $a_i$ nor $b_i$ can become less than zero).\n\nAs said above, all gifts should be equal. This means that after some sequence of moves the following two conditions should be satisfied: $a_1 = a_2 = \\dots = a_n$ and $b_1 = b_2 = \\dots = b_n$ (and $a_i$ equals $b_i$ is not necessary).\n\nYour task is to find the minimum number of moves required to equalize all the given gifts.\n\nYou have to answer $t$ independent test cases.\n\n\n-----Input-----\n\nThe first line of the input contains one integer $t$ ($1 \\le t \\le 1000$) \u2014 the number of test cases. Then $t$ test cases follow.\n\nThe first line of the test case contains one integer $n$ ($1 \\le n \\le 50$) \u2014 the number of gifts. The second line of the test case contains $n$ integers $a_1, a_2, \\dots, a_n$ ($1 \\le a_i \\le 10^9$), where $a_i$ is the number of candies in the $i$-th gift. The third line of the test case contains $n$ integers $b_1, b_2, \\dots, b_n$ ($1 \\le b_i \\le 10^9$), where $b_i$ is the number of oranges in the $i$-th gift.\n\n\n-----Output-----\n\nFor each test case, print one integer: the minimum number of moves required to equalize all the given gifts.\n\n\n-----Example-----\nInput\n5\n3\n3 5 6\n3 2 3\n5\n1 2 3 4 5\n5 4 3 2 1\n3\n1 1 1\n2 2 2\n6\n1 1000000000 1000000000 1000000000 1000000000 1000000000\n1 1 1 1 1 1\n3\n10 12 8\n7 5 4\n\nOutput\n6\n16\n0\n4999999995\n7\n\n\n\n-----Note-----\n\nIn the first test case of the example, we can perform the following sequence of moves:\n\n  choose the first gift and eat one orange from it, so $a = [3, 5, 6]$ and $b = [2, 2, 3]$;  choose the second gift and eat one candy from it, so $a = [3, 4, 6]$ and $b = [2, 2, 3]$;  choose the second gift and eat one candy from it, so $a = [3, 3, 6]$ and $b = [2, 2, 3]$;  choose the third gift and eat one candy and one orange from it, so $a = [3, 3, 5]$ and $b = [2, 2, 2]$;  choose the third gift and eat one candy from it, so $a = [3, 3, 4]$ and $b = [2, 2, 2]$;  choose the third gift and eat one candy from it, so $a = [3, 3, 3]$ and $b = [2, 2, 2]$.",
        "code": "def minimum_moves(t, test_cases):\n    for _ in range(t):\n        n = test_cases[_][0]\n        candies = test_cases[_][1]\n        oranges = test_cases[_][2]\n        min_candies = min(candies)\n        min_oranges = min(oranges)\n        ans = 0\n        for i in range(n):\n            ans += max(candies[i] - min_candies, oranges[i] - min_oranges)\n        print(ans)\n\n\ndef main():\n    t = int(input())\n    test_cases = []\n    for _ in range(t):\n        n = int(input())\n        candies = list(map(int, input().split()))\n        oranges = list(map(int, input().split()))\n        test_cases.append((n, candies, oranges))\n    minimum_moves(t, test_cases)\n\n\nmain()\n",
        "cot": """# - Identify Minimums: For each test case, the first step is to identify the minimum number of candies (min_candies) and oranges (min_oranges) among all gifts. These values act as the target to which all gifts must be equalized by reducing the counts in the others.
# - Calculate Difference for Each Gift: For each gift, the number of operations needed to reduce the count of candies and oranges to min_candies and min_oranges, respectively, is determined. If both candies and oranges need to be reduced, it checks which count is higher (candies or oranges) and matches the reduction to the larger discrepancy.
# - Maximize Simultaneous Reduction: The solution cleverly uses the max function to determine the number of moves for each gift. It calculates the maximum of the difference between the current number of candies and min_candies, and the difference between the current number of oranges and min_oranges. This effectively simulates reducing both candies and oranges simultaneously as much as possible, which minimizes the total moves.
# - Aggregate Moves Across Gifts: The number of moves calculated for each gift (from the previous step) is summed up for each test case to determine the total moves required to equalize all gifts.
# - Output Result: For each test case, the total number of moves is printed, representing the minimal operations required to equalize the candies and oranges for all gifts in that test case. This output reflects the efficient strategy of simultaneous reduction wherever possible to minimize the number of operations."""
    },
    {
        "question": "Let's call a string a phone number if it has length 11 and fits the pattern \"8xxxxxxxxxx\", where each \"x\" is replaced by a digit.\n\nFor example, \"80123456789\" and \"80000000000\" are phone numbers, while \"8012345678\" and \"79000000000\" are not.\n\nYou have n cards with digits, and you want to use them to make as many phone numbers as possible. Each card must be used in at most one phone number, and you don't have to use all cards. The phone numbers do not necessarily have to be distinct.\n\nInput\n\nThe first line contains an integer n \u2014 the number of cards with digits that you have (1 \u2264 n \u2264 100).\n\nThe second line contains a string of n digits (characters \"0\", \"1\", ..., \"9\") s_1, s_2, \u2026, s_n. The string will not contain any other characters, such as leading or trailing spaces.\n\nOutput\n\nIf at least one phone number can be made from these cards, output the maximum number of phone numbers that can be made. Otherwise, output 0.\n\nExamples\n\nInput\n\n11\n00000000008\n\n\nOutput\n\n1\n\n\nInput\n\n22\n0011223344556677889988\n\n\nOutput\n\n2\n\n\nInput\n\n11\n31415926535\n\n\nOutput\n\n0\n\nNote\n\nIn the first example, one phone number, \"8000000000\", can be made from these cards.\n\nIn the second example, you can make two phone numbers from the cards, for example, \"80123456789\" and \"80123456789\".\n\nIn the third example you can't make any phone number from the given cards.",
        "code": "def count_phone_numbers(num_cards, card_digits):\n    count_eights = card_digits.count(\"8\")\n    max_phone_numbers = num_cards // 11\n    max_possible = min(count_eights, max_phone_numbers)\n    return max_possible\n\ndef main():\n    num_cards = int(input())\n    card_digits = input().strip()\n    max_possible = count_phone_numbers(num_cards, card_digits)\n    print(max_possible)\n\nmain()",
        "cot": """# - Count Eights: The first step is to count the number of cards with the digit 8, as this digit is crucial for forming phone numbers. This count represents the maximum number of phone numbers that can be made from the given cards.
# - Determine Maximum Possible Numbers: The maximum number of phone numbers that can be formed is calculated by dividing the total number of cards by 11, as each phone number requires 11 digits. This value represents the upper limit on the number of phone numbers that can be formed.
# - Find Minimum of Counts: The solution then determines the maximum possible number of phone numbers by taking the minimum of the count of cards with the digit 8 and the maximum possible numbers calculated in the previous step. This ensures that the number of phone numbers formed is limited by the availability of cards with the digit 8.
# - Output Result: The maximum possible number of phone numbers that can be formed is printed as the final output. This output reflects the efficient strategy of maximizing the use of cards with the digit 8 to form phone numbers."""
    }
]

LCB_FN_FEWSHOT = [
    {
        "question": "You are given a 0-indexed array of positive integers nums. Find the number of triplets (i, j, k) that meet the following conditions:\n\n0 <= i < j < k < nums.length\nnums[i], nums[j], and nums[k] are pairwise distinct.\n\t\nIn other words, nums[i] != nums[j], nums[i] != nums[k], and nums[j] != nums[k].\n\n\n\nReturn the number of triplets that meet the conditions.\n \nExample 1:\n\nInput: nums = [4,4,2,4,3]\nOutput: 3\nExplanation: The following triplets meet the conditions:\n- (0, 2, 4) because 4 != 2 != 3\n- (1, 2, 4) because 4 != 2 != 3\n- (2, 3, 4) because 2 != 4 != 3\nSince there are 3 triplets, we return 3.\nNote that (2, 0, 4) is not a valid triplet because 2 > 0.\n\nExample 2:\n\nInput: nums = [1,1,1,1,1]\nOutput: 0\nExplanation: No triplets meet the conditions so we return 0.\n\n \nConstraints:\n\n3 <= nums.length <= 100\n1 <= nums[i] <= 1000\n\n",
        "code": "class Solution:\n    def unequalTriplets(self, a: List[int]) -> int:\n        ans = 0\n        n = len(a)\n        for i in range(n):\n            for j in range(i + 1, n):\n                for k in range(j + 1, n):\n                    ans += len({a[i], a[j], a[k]}) == 3\n        return ans",
        "cot": """# - Initialize Count: The solution initializes a variable ans to 0 to store the count of valid triplets that meet the conditions.
# - Iterate Over Triplets: The solution uses three nested loops to iterate over all possible triplets (i, j, k) where 0 <= i < j < k < nums.length. This ensures that the indices are distinct and in increasing order.
# - Check Pairwise Distinctness: For each triplet, the solution checks if the values nums[i], nums[j], and nums[k] are pairwise distinct by converting them to a set and checking if the length of the set is equal to 3. If the values are pairwise distinct, the count ans is incremented.
# - Return Count: The final count of valid triplets is returned as the output. This approach exhaustively checks all possible triplets to count the number of triplets that meet the conditions."""
    },
    {
        "question": "You are given two strings s and t consisting of only lowercase English letters.\nReturn the minimum number of characters that need to be appended to the end of s so that t becomes a subsequence of s.\nA subsequence is a string that can be derived from another string by deleting some or no characters without changing the order of the remaining characters.\n \nExample 1:\n\nInput: s = \"coaching\", t = \"coding\"\nOutput: 4\nExplanation: Append the characters \"ding\" to the end of s so that s = \"coachingding\".\nNow, t is a subsequence of s (\"coachingding\").\nIt can be shown that appending any 3 characters to the end of s will never make t a subsequence.\n\nExample 2:\n\nInput: s = \"abcde\", t = \"a\"\nOutput: 0\nExplanation: t is already a subsequence of s (\"abcde\").\n\nExample 3:\n\nInput: s = \"z\", t = \"abcde\"\nOutput: 5\nExplanation: Append the characters \"abcde\" to the end of s so that s = \"zabcde\".\nNow, t is a subsequence of s (\"zabcde\").\nIt can be shown that appending any 4 characters to the end of s will never make t a subsequence.\n\n \nConstraints:\n\n1 <= s.length, t.length <= 10^5\ns and t consist only of lowercase English letters.\n\n",
        "code": "class Solution:\n    def appendCharacters(self, s: str, t: str) -> int:\n        i = 0\n        for char in s:\n            if i < len(t) and char == t[i]:\n                i += 1\n        return len(t) - i",
        "cot": """# - Initialize Index: The solution initializes an index i to 0 to keep track of the position in the string t that needs to be matched.
# - Iterate Over Characters: The solution iterates over each character in the string s.
# - Match Characters: For each character in s, the solution checks if the character matches the character at index i in the string t. If there is a match, the index i is incremented to move to the next character in t.
# - Calculate Remaining Characters: The solution calculates the number of characters remaining in t that need to be appended to s to make t a subsequence of s by subtracting the final index i from the length of t.
# - Return Result: The number of characters that need to be appended to s is returned as the output. This approach efficiently finds the minimum number of characters needed to make t a subsequence of s by matching characters in both strings."""
    }
]


def py_prompt_2shot_lcb(question: str, code="", cot=False) -> str:
    question = question.replace('"""', r'\"""')

    if code == "" or code is None:
        shots_arr = LCB_IO_FEWSHOT
    else:
        shots_arr = LCB_FN_FEWSHOT

    shots = ""
    for shot in shots_arr:
        if cot:
            cot_str = shot["cot"] + "\n\n"
        else:
            cot_str = ""
        shots += f'''# START NEW CODE
"""
{shot["question"]}
"""
{cot_str}{shot["code"]}
'''
    return shots + f'''# START NEW CODE
"""
{question}
"""
{code}'''.strip()


def py_prompt_2shot_lcb_chat(question: str, code="") -> Conversation:
    system = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program inside markdown codeblocks"
    og_q = question.replace('"""', r'\"""')
    if code == "" or code is None:
        shots_arr = LCB_IO_FEWSHOT
    else:
        shots_arr = LCB_FN_FEWSHOT

    shots = [{
        "role": "system",
        "content": system
    }]
    for shot in shots_arr:
        question = shot["question"]
        response = shot["code"]

        shots.append({
            "role": "user",
            "content": f"Please write a Python program that meets the following requirements:\n\n{question}"
        })
        shots.append({
            "role": "system",
            "content": f"```python\n{response}\n```"
        })

    shots.append({
        "role": "user",
        "content": f"Please write a Python program that meets the following requirements:\n\n{og_q}"
    })

    if not (code == "" or code is None):
        shots[-1]["content"] += "\n\nYour solution should utilize the following starter code:\n\n```python\n" + code + "\n```"

    return shots
