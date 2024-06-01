import libcst as cst

# Mutation operators
# AOR: Arithmetic Operator Replacement: a + b -> a - b
# LCR: Logical Connector Replacement: a and b -> a or b
# ROR: Relational Operator Replacement: a > b -> a < b
# UOI: Unary Operator Insertion: b -> not b, i -> i + 1
# SBR: Statement Block Replacement: stmt -> 0


class MutationStepGatherer(cst.CSTVisitor):
    """
    Finds the number of nodes that can be mutated by an operator.
    """
    def __init__(self):
        self.aor = 0
        self.lcr = 0
        self.ror = 0
        self.uoi = 0
        self.sbr = 0

    def visit_BinaryOperation(self, node: cst.BinaryOperation):



def mutate(code: str) -> str:
    module = cst.parse_module(code)
    pass

def test():
    MY_CODE = """
a = 1
b = 2
for i in range(10):
    e = a + b
    print(e)

def my_func(a, b):
    return a + b
"""
    mutated_code = mutate(MY_CODE)
    print(mutated_code)


if __name__ == "__main__":
    test()
