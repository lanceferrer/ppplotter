def caret_to_exponent(expr_str):
    # first check if the string even contains a caret
    if "^" not in expr_str:
        return expr_str
    else:
        return expr_str.replace("^", "**")