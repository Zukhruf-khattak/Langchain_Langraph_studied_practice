import ast
import math
import operator
from typing import Any, Dict

# Allowed binary and unary operators for safe AST evaluation
SAFE_OPERATORS: Dict[Any, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allowed math functions and constants
SAFE_NAMES: Dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "abs": abs,
    "round": round,
    "pow": pow,
}


def _eval_node(node: ast.AST) -> Any:
    """Recursively evaluate an AST node safely."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    elif isinstance(node, ast.Constant):  # Numbers or strings
        return node.value
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_node(node.operand)
        return SAFE_OPERATORS[op_type](operand)
    elif isinstance(node, ast.Name):
        if node.id in SAFE_NAMES:
            return SAFE_NAMES[node.id]
        raise ValueError(f"Unknown variable or constant: {node.id}")
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in SAFE_NAMES:
            func = SAFE_NAMES[node.func.id]
            args = [_eval_node(arg) for arg in node.args]
            return func(*args)
        raise ValueError(f"Function call not permitted: {ast.dump(node)}")
    else:
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.

    Args:
        expression: A mathematical string expression to calculate (e.g. '2 + 2', 'sqrt(144) * 3', '2**10', 'sin(pi/2)').

    Returns:
        The calculated result as a string, or an error message if evaluation fails.
    """
    try:
        # Clean expression
        cleaned = expression.strip()
        parsed = ast.parse(cleaned, mode="eval")
        result = _eval_node(parsed)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression '{expression}': {str(e)}"
