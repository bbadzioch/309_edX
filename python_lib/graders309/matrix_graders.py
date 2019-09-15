from __future__ import division
from math import *
import sympy
import numpy as np
import re

from graders309.exceptions import *
from graders309.helpers import *
from graders309.lin_alg import *



def array_grader(atol, rtol, dtype=float):

    def check_answer(expect, ans):

        s_expect = expect.lower().strip()
        s_ans = expect.lower().strip()
        if s_expect == "none" or s_ans == "none":
            return s_expect == s_ans

        A_ans = srt_2_array(ans, dtype=dtype)
        A_exp = srt_2_array(expect, dtype=dtype)

        if A_ans.shape != A_exp.shape:
            return False
        if not np.allclose(A_ans, A_exp, atol= atol, rtol=rtol):
            return False
        else:
            return True

    return check_answer




def linear_eqs_grader(A, b, symb = "x", atol=0.1, rtol=0):

    A = sympy.Matrix(A)
    b = sympy.Matrix(b)
    n = A.shape[1]
    Ab = A.col_insert(n, b)
    no_solutions = (n in Ab.rref()[1])
    rank = len(A.rref()[1])
    nullity = n - rank


    # columns of the test matrix give test values of free variables
    # for checking the solution
    if nullity > 0:
        test_matrix = sympy.eye(nullity).col_insert(0, sympy.Matrix([0]*nullity))
    else:
        test_matrix = sympy.Matrix([0])

    # prepare a dictionary of names of variables appearing in the system of equations
    # and their associated sympy symbols; this is used for parsing answer strings
    variables = {}
    for i in range(n):
        variables["{0}_{1}".format(symb, i+1)] = sympy.Symbol("{0}_{1}".format(symb, i+1))

    # convert the coefficient matrix and the vector of constants into
    # numpy array - it is more convenient to test the solution in this form
    A_arr = np.array(A).astype(float)
    b_arr = np.array(b).astype(float).ravel()


    def check_answer(expect, ans):

        # test for no solutions
        ans_str = [s.strip().lower() for s in ans]
        if no_solutions or "none" in ans_str:
            if no_solutions == ("none" in ans_str):
                return True
            else:
                return False
    

        # number of entries in the answer must match the number of variables.
        if len(ans) != n:
            raise FormatError("The number of submitted values does not match the number of variables")

        # convert answer entries into sympy expressions
        # and collect free variables appearing in the answer
        free_vars = set()
        eval_ans = []

        for i in range(n):
            y = scrub_string(ans[i], glob_dir=variables)
            try :
                fy = y.free_symbols
            except AttributeError:
                fy = set()
            free_vars.update(fy)
            eval_ans.append(y)

            # free variables must evaluate to themselves, without the code below
            # solutions such as x_3 = -x_3 would be accepted
            x_var = variables["x_{}".format(i+1)]
            # check if x_i appears in the expression entered as the value of x_i
            if x_var in fy:
                # if there are any other variables in such expression reject the solution
                if len(fy) > 1:
                    return False
                # otherwise check is the expression evaluates to x_i withing some small tolerance
                else: 
                    test_vals = np.linspace(-1, 1, 10)
                    var_tol = 0.0000000001
                    for t in test_vals:
                        if abs(t - y.subs(x_var, t)) > var_tol:
                            return False


        free_vars = list(free_vars)

        # the number of free variables in the answer must match the nullity
        # of the coefficient matrix
        if len(free_vars) != nullity:
            return False

        # plug in test values into free variables in the answer
        # check is the resulting vector is a solution of the system
        for i in range(nullity+1):
            test_vals = test_matrix[:, i]
            test_sub = list(zip(free_vars, test_vals))
            ans_subs = []
            for y in eval_ans:
                try:
                    ys = y.subs(test_sub)
                except AttributeError:
                    ys = y
                ans_subs.append(ys)
            ans_subs_arr = np.array(ans_subs).astype(float)
            ans_prod = np.dot(A_arr, ans_subs_arr)
            if not np.allclose(ans_prod, b_arr, atol= atol, rtol=rtol):
                return False

        return True

    return check_answer



class Inequality_Grader():
    """
    This class returns a callable object which tests is two strings
    represent equivalent inequalities.
    """

    def double_eq(self, s):
        """
        Replaces "=" with "==" if needed.
        """
        t =  re.sub(r"(?<![<>=!])=(?!=)", "==", s)
        return t


    def format_sol(self, s):

        s = s.replace(" ", "")
        if s.lower() == "None".lower():
            return "None"
        elif s.lower() == "Any".lower():
            return "Any"
        else:
            return self.double_eq(s)

    def find_oper(self, s):
        """
        Searches for an operator (one of: '==', '!=', '<=', '>=', '<', '>')
        in a string representing an inequality. Returns the operator or None
        if not found.
        """
        m = re.search(r"[<>=!]=|[<>]", s)
        if m is None:
            return m
        else:
            return m.group(0)

    def split(self, s):
        """
        Splits an inequality into substrings representing the left hand side,
        the operator and the right hand side. It the operator is '<' or '<='
        replaces it with '>' or '>=', respectively, and intechanges lhs with rhs.
        """
        op = self.find_oper(s)

        if op is  None:
            return None

        n = s.find(op)
        m = len(op)
        lhs = s[:n]
        rhs = s[n+m:]
        if "<" in op:
            op = op.replace("<", ">")
            lhs, rhs = rhs, lhs
        return (lhs, op, rhs)

    def __call__(self, expect, ans, symb="k"):
        """
        Returns either True or False depending if the strings expect and ans
        represent equivalent or non-equivalent inequalities.
        """
        try:
            k = sympy.Symbol(symb)
            glob_dir = {symb:k}
            s_exp = self.format_sol(expect)
            s_ans = self.format_sol(ans)
            if s_exp in ["Any", "None"]:
                return s_exp == s_ans
            if symb not in s_ans:
                return False
            lhs_exp, op_exp, rhs_exp = self.split(s_exp)
            lhs_ans, op_ans, rhs_ans = self.split(s_ans)
            if op_exp != op_ans:
                return False
            comb_exp = scrub_string("(" + lhs_exp + ") - (" + rhs_exp + ")", glob_dir=glob_dir)
            comb_ans = scrub_string("(" + lhs_ans + ") - (" + rhs_ans + ")", glob_dir=glob_dir)
            q = (sympy.cancel(comb_ans/comb_exp))
            try:
                fq = float(q)
                if fq <= 0 and ">" in op_exp:
                    return False
                else:
                    return True
            except (TypeError, ValueError):
                return False
        except Exception:
            raise FormatError("Incorrect format of the answer: {}".format(ans))


def span_grader(A):

    """
    Returns a grader function which checks if an answer specifies
    a set of vectors whose span is the column space of the matrix A.
    """

    def check_answer(expect, ans):

        ans = ans.strip().replace(" ", "")
        # transform the answer into a matrix format if needed
        if not ans.startswith("[["):
            ans = "[" + ans + "]"

        B  =  srt_2_array(ans, dtype=float).T
        return check_span(A, B)

    return check_answer


def basis_grader(A, orthogonal=False):

    """
    Returns a grader function which checks if an answer specifies
    a set of vectors which give a basis of the column space of the matrix A.

    :A:
        A numpy array representing the matrix.
    :orthogonal:
        Boolean. If true checks if the given basis is orthogonal.
    """

    def check_answer(expect, ans):

        ans = ans.strip().replace(" ", "")
        # transform the answer into a matrix format if needed
        if not ans.startswith("[["):
            ans = "[" + ans + "]"

        B  =  srt_2_array(ans, dtype=float).T

        if not check_basis(A, B):
            return False

        if orthogonal:
            if not check_orthogonal_set(B):
                return False

        return True

    return check_answer
