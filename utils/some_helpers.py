def caret_to_exponent(expr_str):
    expr_str = expr_str.replace("^", "**")
    return sp.sympify(expr_str)