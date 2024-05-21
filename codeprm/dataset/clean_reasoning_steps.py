import re
import datasets
from typing import List


def get_reasoning_steps(code: str) -> List[str]:
    lines = code.split('\n')
    comments = []
    current_comment = []

    for line in lines:
        stripped_line = line.lstrip()
        if stripped_line.startswith('#'):
            comment_content = stripped_line[1:].strip()
            current_comment.append(comment_content)
        else:
            if current_comment:
                comments.append(' '.join(current_comment))
                current_comment = []

    if current_comment:
        comments.append(' '.join(current_comment))

    return comments


def transform_trailing_to_leading_comments(code: str) -> str:
    def is_inside_string(s, pos):
        """Check if position is inside a string"""
        in_string = False
        escaped = False
        string_char = ''

        for i, char in enumerate(s):
            if char in ('"', "'"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif in_string and string_char == char and not escaped:
                    in_string = False
            if in_string and char == '\\':
                escaped = not escaped
            else:
                escaped = False
            if i == pos:
                return in_string

        return False

    lines = code.split('\n')
    transformed_lines = []

    for line in lines:
        if '#' in line:
            split_index = None
            for match in re.finditer('#', line):
                if not is_inside_string(line, match.start()):
                    split_index = match.start()
                    break

            if split_index is not None:
                code_part = line[:split_index].rstrip()
                comment_part = line[split_index + 1:].strip()
                leading_whitespace = line[:len(line) - len(line.lstrip())]
                if code_part:
                    transformed_lines.append(
                        f"{leading_whitespace}# {comment_part}")
                    transformed_lines.append(code_part)
                else:
                    transformed_lines.append(
                        f"{leading_whitespace}# {comment_part}")
            else:
                transformed_lines.append(line)
        else:
            transformed_lines.append(line)

    return '\n'.join(transformed_lines)


def main(args):
    ds = datasets.load_dataset(args.dataset, split="train")
    ds = ds.map(lambda x: {
        "reasoning_steps": [transform_trailing_to_leading_comments(r) for r in x["reasoning_steps"] if r is not None],
    })
    ds = ds.map(lambda x: {
        "reasoning_steps": [r for r in x["reasoning_steps"] if 0 < len(get_reasoning_steps(r)) <= 50],
    })
    ds = ds.filter(lambda x: len(x["reasoning_steps"]) > 0)
    ds.save_to_disk(args.output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="cassanof/taco_cleaned_exec_filtered_v4_reasoning10")
    parser.add_argument("--output", type=str,
                        default="./taco_cleaned_exec_filtered_v4_reasoning10_cleaned")
    args = parser.parse_args()
    main(args)
