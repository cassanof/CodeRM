from transformers import AutoTokenizer

DIRECT_CHAT_TEMPLATE = """\
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
        {{ raise_exception('System messages are not allowed in this template.') }}
    {%- else %}
{{message['content']}}
    {%- endif %}
{%- endfor %}
"""

TEST_CONVO = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
]


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(getattr(args, "in"))
    tokenizer.chat_template = DIRECT_CHAT_TEMPLATE

    # try it out
    convo = tokenizer.apply_chat_template(TEST_CONVO, tokenize=False)
    print(convo)

    if args.push:
        tokenizer.push_to_hub(args.out, private=True)
    else:
        tokenizer.save_pretrained(args.out)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    main(args)
