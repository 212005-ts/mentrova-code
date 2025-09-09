# app/tools/calculator.py
import math

def safe_eval(expression: str) -> str:
    """Safely evaluate simple math expressions."""
    try:
        # Only allow digits, operators, parentheses, decimal
        allowed = "0123456789+-*/(). "
        if not all(c in allowed for c in expression):
            return " Invalid expression"
        result = eval(expression, {"__builtins__": {}}, math.__dict__)
        return str(result)
    except Exception as e:
        return f" Error: {str(e)}"
