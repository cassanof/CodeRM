def py_prompt(question: str, code=""):
    # escape any triple quotes in the question
    question = question.replace('"""', r'\"""')
    return f'''"""
{question}
"""
{code}'''


def py_prompt_3shot(question: str, code=""):
    question = question.replace('"""', r'\"""')
    return f'''"""
{question}
"""
{code}'''
