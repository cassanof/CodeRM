from typing import List
import random
import libcst as cst

# Mutation operators
# AOR: Arithmetic Operator Replacement: a + b -> a - b
# LCR: Logical Connector Replacement: a and b -> a or b
# ROR: Relational Operator Replacement: a > b -> a < b
# SBR: Statement Block Replacement: stmt -> 0
# COR: Conditional Operator Replacement: a if b else c -> a if not b else c

AOR_INSTANCES = [
    cst.Add,
    cst.Subtract,
    cst.Multiply,
    cst.FloorDivide,
    cst.Divide,
    cst.Modulo,
    cst.Power,
]

LCR_INSTANCES = [cst.And, cst.Or]

ROR_INSTANCES = [cst.GreaterThan, cst.LessThan,
                 cst.GreaterThanEqual, cst.LessThanEqual]

SBR_INSTANCES = [cst.SimpleStatementLine]

MUTATION_OPERATORS = [
    "AOR",
    "LCR",
    "ROR",
    "SBR",
    "COR",
]


class PossibleMutations:
    def __init__(self):
        self.aor = 0
        self.lcr = 0
        self.ror = 0
        self.sbr = 0
        self.cor = 0

    def possible(self) -> List[str]:
        p = []
        if self.aor > 0:
            p.append("AOR")
        if self.lcr > 0:
            p.append("LCR")
        if self.ror > 0:
            p.append("ROR")
        if self.sbr > 0:
            p.append("SBR")
        if self.cor > 0:
            p.append("COR")
        return p

    def pick_random_of_op(self, op: str) -> int:
        if op == "AOR":
            u = self.aor
        elif op == "LCR":
            u = self.lcr
        elif op == "ROR":
            u = self.ror
        elif op == "SBR":
            u = self.sbr
        elif op == "COR":
            u = self.cor
        else:
            raise ValueError(f"Unknown operator: {op}")

        return random.randint(0, u-1)


class MutationStepGatherer(cst.CSTVisitor):
    """
    Finds the number of nodes that can be mutated by an operator.
    """

    def __init__(self):
        self.p = PossibleMutations()

    def get_possible_mutations(self) -> PossibleMutations:
        return self.p

    def leave_BinaryOperation(self, original_node: cst.BinaryOperation) -> None:
        if isinstance(original_node.operator, tuple(AOR_INSTANCES)):
            self.p.aor += 1

    def leave_BooleanOperation(self, original_node: cst.BooleanOperation) -> None:
        if isinstance(original_node.operator, tuple(LCR_INSTANCES)):
            self.p.lcr += 1

    def leave_Comparison(self, original_node: cst.Comparison) -> None:
        for op in original_node.comparisons:
            if isinstance(op.operator, tuple(ROR_INSTANCES)):
                self.p.ror += 1

    def leave_SimpleStatementLine(self, original_node: cst.SimpleStatementLine) -> None:
        self.p.sbr += 1

    def leave_If(self, original_node: cst.If) -> None:
        self.p.cor += 1

    def leave_IfExp(self, original_node: cst.IfExp) -> None:
        self.p.cor += 1

    def leave_CompIf(self, original_node: cst.CompIf) -> None:
        self.p.cor += 1


class Mutator(cst.CSTTransformer):
    def __init__(
            self,
            possible_mutations: PossibleMutations,
    ):
        self.p = possible_mutations
        self.t = PossibleMutations()
        assert len(self.p.possible()) > 0, "No mutations possible"
        self.mut = random.choice(self.p.possible())
        self.mut_idx = self.p.pick_random_of_op(self.mut)

    def leave_BinaryOperation(self, original_node: cst.BinaryOperation, updated_node: cst.BinaryOperation) -> cst.BinaryOperation:
        if isinstance(original_node.operator, tuple(AOR_INSTANCES)):
            if self.mut == "AOR" and self.mut_idx == self.t.aor:
                op = random.choice(
                    [op for op in AOR_INSTANCES if op != original_node.operator.__class__])
                updated_node = updated_node.with_changes(operator=op())
            self.t.aor += 1
        return updated_node

    def leave_BooleanOperation(self, original_node: cst.BooleanOperation, updated_node: cst.BooleanOperation) -> cst.BooleanOperation:
        if isinstance(original_node.operator, tuple(LCR_INSTANCES)):
            if self.mut == "LCR" and self.mut_idx == self.t.lcr:
                op = random.choice(
                    [op for op in LCR_INSTANCES if op != original_node.operator.__class__])
                updated_node = updated_node.with_changes(operator=op())
            self.t.lcr += 1
        return updated_node

    def leave_Comparison(self, original_node: cst.Comparison, updated_node: cst.Comparison) -> cst.Comparison:
        for i, ct in enumerate(original_node.comparisons):
            if isinstance(ct.operator, tuple(ROR_INSTANCES)):
                if self.mut == "ROR" and self.mut_idx == self.t.ror:
                    new_op = random.choice(
                        [op for op in ROR_INSTANCES if op != ct.operator.__class__])
                    new_comparisons = [
                        cst.ComparisonTarget(
                            comparator=ct.comparator,
                            operator=new_op(),
                        )
                        if j == i else ct2 for j, ct2 in enumerate(original_node.comparisons)]
                    updated_node = updated_node.with_changes(
                        comparisons=new_comparisons)
                self.t.ror += 1
        return updated_node

    def leave_SimpleStatementLine(self, original_node: cst.SimpleStatementLine, updated_node: cst.SimpleStatementLine) -> cst.SimpleStatementLine:
        stmts = original_node.body
        if self.mut == "SBR" and self.mut_idx == self.t.sbr and len(stmts) > 0:
            keep_out = random.randint(0, len(stmts)-1)
            new_stmts = [stmt for i, stmt in enumerate(stmts) if i != keep_out]
            updated_node = updated_node.with_changes(body=new_stmts)
        self.t.sbr += 1
        return updated_node

    def leave_If(self, original_node: cst.If, updated_node: cst.If) -> cst.If:
        if self.mut == "COR" and self.mut_idx == self.t.cor:
            updated_node = updated_node.with_changes(
                test=cst.UnaryOperation(
                    operator=cst.Not(),
                    expression=original_node.test,
                )
            )
        self.t.cor += 1
        return updated_node

    def leave_IfExp(self, original_node: cst.IfExp, updated_node: cst.IfExp) -> cst.IfExp:
        if self.mut == "COR" and self.mut_idx == self.t.cor:
            updated_node = updated_node.with_changes(
                test=cst.UnaryOperation(
                    operator=cst.Not(),
                    expression=original_node.test,
                )
            )
        self.t.cor += 1
        return updated_node

    def leave_CompIf(self, original_node: cst.CompIf, updated_node: cst.CompIf) -> cst.CompIf:
        if self.mut == "COR" and self.mut_idx == self.t.cor:
            updated_node = updated_node.with_changes(
                test=cst.UnaryOperation(
                    operator=cst.Not(),
                    expression=original_node.test,
                )
            )
        self.t.cor += 1
        return updated_node


def mutate(code: str) -> str:
    module = cst.parse_module(code)
    # visit the module to gather the number of nodes that can be mutated
    gatherer = MutationStepGatherer()
    module.visit(gatherer)
    p = gatherer.get_possible_mutations()
    # mutate the module
    mutator = Mutator(p)
    transformed_module = module.visit(mutator)
    return transformed_module.code


def test():
    MY_CODE = """
# add two numbers
def add(a, b):
    return a + b

# is a and b both true
def is_both_true(a, b):
    return a and b

# is a greater than b
def is_greater(a, b):
    return a > b

# is a not true
def is_not_true(a):
    return not a

# increment i by 1
def increment(i):
    print(i + 3.2)
    return i + 1

# return true
def true_statement():
    return True

# a bunch of statements
def bunch_of_statements():
    print("Hello")
    print("World")
    print("!")
"""
    mutated_code = mutate(MY_CODE)
    import difflib
    for line in difflib.unified_diff(MY_CODE.splitlines(), mutated_code.splitlines()):
        print(line)


def test_fuzz():
    from tqdm import tqdm
    import ast
    import datasets
    ds = datasets.load_dataset(
        "nuprl/stack-dedup-python-testgen-starcoder-filter-v2", split="train")
    ipf = 20
    for ex in tqdm(ds, total=len(ds)):  # type: ignore
        code = ex["content"]  # type: ignore
        for i in range(ipf):
            try:
                mut = mutate(code)
                ast.parse(mut)
            except Exception as e:
                print(e)
                print(code)


if __name__ == "__main__":
    test()
    test_fuzz()
