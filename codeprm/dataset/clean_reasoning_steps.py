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


def remove_comments(code):
    """Remove comments from the code."""
    return "\n".join(line for line in code.split('\n') if not line.strip().startswith('#'))


def count_characters(code):
    """Count the number of characters without comments."""
    code_no_comments = remove_comments(code)
    return len(code_no_comments.replace('\n', '').replace(' ', ''))


def get_steps_to_char_ratio(code):
    """Calculate the steps to number of characters ratio for the given code."""
    reasoning_steps = len(get_reasoning_steps(code))
    characters_count = count_characters(code)
    return reasoning_steps / characters_count if characters_count > 0 else 0


def attempt_convert_tabs_to_spaces(code):
    """Attempt to convert tabs to spaces in the code. Uses ast.parse to make sure the code is valid."""
    import ast

    converted_code = []
    for line in code.split('\n'):
        converted_code.append(line.expandtabs(4))

    converted = '\n'.join(converted_code)
    try:
        ast.parse(converted)
        return converted
    except SyntaxError:
        print("Syntax error after converting tabs to spaces.")
        return code


def main(args):
    ds = datasets.load_dataset(args.dataset, split="train")
    ds = ds.map(lambda x: {
        "reasoning_steps": [transform_trailing_to_leading_comments(r) for r in x["reasoning_steps"] if r is not None],
    })
    # steps to char ratio needs to be `0.01 <= ratio <= 0.8`
    # this is based on visual inspection of the data (see taco_reasoning_analysis.ipynb)
    ds = ds.map(lambda x: {
        "reasoning_steps": [r for r in x["reasoning_steps"] if 0.01 <= get_steps_to_char_ratio(r) <= 0.8],
    })
    # convert tabs to spaces
    ds = ds.map(lambda x: {
        "reasoning_steps": [attempt_convert_tabs_to_spaces(r) for r in x["reasoning_steps"]],
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
