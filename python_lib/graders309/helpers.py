
from __future__ import division
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import math
import numpy as np
import random

from graders309.exceptions import *


def np_randomizer():
    """
    Open edX provides a seed to the random module in order to assure that
    data in randomized problems is the same when the problem is presented
    to a student and when the answer is graded. Since such seed is not provided
    for numpy random generators, these generators can't be used directly.
    To fix this this function initializes the random state of numpy using
    a the random module to select a seed.
    """
    random_seed = random.randrange(100)
    np.random.seed(random_seed)


def plot2img(fig, dpi=200):
    """
    Converts a matplotlib figure objext into a base64-encoded image file
    in memory. The result can be embedded in an html code to display the image.

    :fig:
        A matplotlib figure object.
    :dpi:
        dpi of the image file.

    Returns:
        A string with base64-encoded image file.
    """

    tmp = BytesIO()
    fig.savefig(tmp, format="png", dpi=dpi)
    enc = base64.b64encode(tmp.getvalue()).decode("utf-8")
    tmp.close()
    return enc



def scrub_string(s, glob_dir={}, loc_dir={}, clear=False):
    """
    Scrubs a string, evaluates it, and returns the
    evaluated expression. It also replaces "^" with "**" in the string.

    :s:
        A string.
    :glob_dir:
        A directory of global objects permitted in the string
    :local_dir:
        A directory of local objects names permitted in the string
    :clear: Boolean.
        If False all objects from the math module as well as the abs
        and pow functions are inserted in glob_dir.

    Returns:
        The evaluated string.
    """

    if not clear:
        allowed_builtins = {"abs":abs, "pow":pow, }
        math_dir = {s:getattr(math, s) for s in dir(math) if "__" not in s}
        glob_dir["__builtins__"] = allowed_builtins
        glob_dir.update(math_dir)

    # disable double underscores since it can be used to get object attributes
    if "__" in s:
        raise FormatError("Invalid input")

    try:
        s.strip()
        s.replace(" ", "")
        s = s.replace("^", "**")
        return eval(s ,glob_dir, loc_dir)
    except Exception:
        raise FormatError("Invalid input")



def srt_2_array(s, dtype=float, glob_dir={}, loc_dir={}, clear=False):
    """
    Converts a string into a numpy array.
    Useful for checking correctness of student solutions.

    :s:
        A string.
    :dtype:
        dtype of the resulting numpy array.
    :glob_dir:
    :loc_dir:
    :clear:
        The same as in the scrub_string functions

    Returns:
        A numpy array
    """
    try:
        s = scrub_string(s, glob_dir, loc_dir, clear)
        a = np.array(s).astype(dtype)
    except Exception:
        raise FormatError("Invalid input")
    return a

#fixing typo in the function name...above
str_2_array = srt_2_array



def matrix2str(A, float_prec=None):
    """
    Converts a sympy or numpy matrix into a string.
    Useful for converting a numerical solution of a
    problem string that can be inserted as expected
    answer to a problem.

    :A:
        A matrix.
    :float_prec:
        The number of decimal digits in matrix entries.
        If A is a sympy matrix whose entries are fractions,
        the default None value will result in a string where
        matrix entries will be give by the fractions.

    Returns:
        A string representation of the matrix.
    """

    if len(A.shape) == 1:
        A = A.reshape(1, -1)

    m, n = A.shape
    A_str = []
    for i in range(m):
        if float_prec is not None:
            row_str = ["{:.{}f}".format(float(x), float_prec) for x in A[i, :]]
        else:
            row_str  = [str(x) for x in A[i, :]]
        row_str = "[" + ", ".join(row_str) + "]"
        A_str.append(row_str)
    if m == 1:
        return A_str[0]
    else:
        return "[" + ", ".join(A_str) + "]"



def minus_space(num):
    """
    Converts numbers to string for in a form convenient for
    LaTeX vector/matrix formatting.

    :n:
        A number.

    Returns:
        If n is  negative number a string representatiion of n.
        If n is non-negative appends "\phantom{-}" in front of the string
        representation of n.

    """
    if num < 0:
        return str(num)
    else:
        return r"\phantom{{-}}{}".format(num)



def vector_2_latex(v, add_min = True):
    """
    Converts a vector into a LateX-formatted string
    Useful for embedding vectors from Python script
    into text of a problem.

    :v:
        A vector, i.e. a list of numbers
    :add_min:
       Boolean. If True spacing is added in from of all non-negative
       entries to justify them with negative entries.
    Returns:
        A string with a LaTeX string representing the vector.
    """

    if add_min:
        f = minus_space
    else:
        f = str

    s_middle = " ".join([f(num) + r" \\" for num in v])
    s_begin = r"\begin{bmatrix} "
    s_end = r" \end{bmatrix}"
    return s_begin + s_middle + s_end



def matrix_2_latex(A, add_min = True):
    """
    Converts a numpy matrix into a LateX-formatted string
    Useful for embedding matrices from Python script
    into text of a problem.

    :A:
        A numpy matrix.
    :add_min:
        Boolean. If True spacing is added in from of all non-negative
        entries to justify them with negative entries.
    Returns:
        A string with a LaTeX string representing the vector.
    """

    if add_min:
        f = minus_space
    else:
        f = str

    s = ""

    if len(A.shape) == 1:
        A = A.reshape((-1, 1))
    r, c = A.shape
    for i in range(r):
        for j in range(c-1):
            s += f(A[i, j]) + " &amp; "
        s += f(A[i, c-1]) + r" \\" + "\n"
    s_begin = r"\begin{bmatrix} " + "\n"
    s_end = r"\end{bmatrix}"
    return s_begin + s + s_end



def format_coeff(a, i, first=False, var_name="x"):
    """
    Formats variable expressions in linear equations.
    For example, format_coeff(5, 1, first=False, var_name="x")
    will return the string "+ 5x_1"

    :a:
        Coeffient of the expression.
    :i:
        Index of the variable.
    :first:
        True if this is the expression which begins the equation,
        False otherwise.
    :var_name:
        Name of variables in the equation.
    Returns:
        A string with the formatted expression.
    """

    if a == 0:
        return ""
    elif a == 1:
        if first:
            return "{}_{{{}}}".format(var_name, i)
        else:
            return "+ {}_{{{}}}".format(var_name, i)
    elif a == -1:
        if first:
            return "-{}_{{{}}}".format(var_name, i)
        else:
            return "- {}_{{{}}}".format(var_name, i)
    elif a > 0:
        if first:
            return "{}{}_{{{}}}".format(a, var_name, i)
        else:
            return " + {}{}_{{{}}}".format(a, var_name, i)
    else:
        if first:
            return "-{}{}_{{{}}}".format(abs(a), var_name, i)
        else:
            return " - {}{}_{{{}}}".format(abs(a), var_name, i)



def format_lin_eqs(A, var_name="x"):
    """
    For given matrix A returns a LaTeX string with a system of linear
    equations that has A as the augmented matrix.

    :A:
        The augmented matrix of the system
    :var_name:
        The name of variables kin the system.

    Returns:
        A string with the LaTeX code of the system of equations.
    """

    r, c = A.shape
    s = ""
    for i in range(r):
        first = True
        for j in range(c-1):
            s += format_coeff(A[i, j], j+1, first=first, var_name=var_name)
            if A[i, j] != 0:
                first = False
        s += r" = {} \\".format(A[i, -1]) + "\n"

    s = r"\begin{cases}" +"\n" + s + r"\end{cases}"
    return s


def format_poly(coeffs, var_name = "x"):
    """
    Formats a string representing a polynomial based
    on listy of its coefficients

    :coeffs:
       List of numbers giving coefficients of the polynomial
       ordering in the increasing degree order.
    :var_name:
        String. Name of the polynomial variable.

    Returns:
        A string with the formatted polynomial.
    """

    def format_mono(c, i, var_name):
        """
        Format monomials
        """

        if c == 0:
            return ""
        if i == 0:
            return str(c)
        xi = "{}^{}".format(var_name, i) if i>1 else var_name
        if c == 1:
            return "+" + xi
        if c == -1:
            return "-" + xi
        if c > 0:
            return "+" + str(c) + "*" + xi
        if c < 0:
            return str(c) + "*" + xi

    poly = "".join([format_mono(c, i, var_name) for (i, c) in enumerate(coeffs)])
    if poly == "":
        return "0"
    if poly[0] == "+":
        poly = poly[1:]
    poly = poly[0] + poly[1:].replace("+", " + ").replace("-", " - ")
    return poly
