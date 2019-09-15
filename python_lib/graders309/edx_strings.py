# an html table describing syntax for entering equality/inequality
# conditions on a parameter
def ineq_table(param="k"):

    s= r"""
    <table style="width:100%; border: 1px solid #aeb4bf;">
      <tr style="background-color:#dcdfe5;">
        <th style="padding-left: 15px; padding-right: 15px;">What you can enter:</th>
        <th style="padding-left: 15px; padding-right: 15px;">What it means:</th>
        <th style="padding-left: 15px; padding-right: 15px;">Example</th>
      </tr>
      <tr>
        <td style="padding-left: 15px; padding-right: 15px;">\(\color{red}{\tt __k__  =  a}\ \) </td>
        <td style="padding-left: 15px; padding-right: 15px;">The condition is satisfied only if \(\tt __k__ \) is equal to \( \tt a\).</td>
        <td style="padding-left: 15px; padding-right: 15px;">\({\tt __k__ = 4}\)</td>
      </tr>
      <tr>
        <td style="padding-left: 15px; padding-right: 15px;">\(\color{red}{\tt __k__ \hskip 0.5mm  !\!\!=  a}\ \) </td>
        <td style="padding-left: 15px; padding-right: 15px;">The condition is satisfied for any value of \(\tt __k__\) different than \(\tt a\).</td>
        <td style="padding-left: 15px; padding-right: 15px;">\({\tt __k__\hskip 0.5mm !\!\!= 4}\)</td>
      </tr>
      <tr>
        <td style="padding-left: 15px; padding-right: 15px;">\(\color{red}{\tt __k__ >  a}\ \) </td>
        <td style="padding-left: 15px; padding-right: 15px;">The condition is satisfied for any value of \(\tt __k__\) greater than \(\tt a\).</td>
        <td style="padding-left: 15px; padding-right: 15px;">\({\tt __k__ > 4}\)</td>
      </tr>
      <tr>
        <td style="padding-left: 15px; padding-right: 15px;">\(\color{red}{\tt __k__ >=  a}\ \) </td>
        <td style="padding-left: 15px; padding-right: 15px;">The condition is satisfied for any value of \(\tt __k__\) greater or equal to  \(\tt a\).</td>
        <td style="padding-left: 15px; padding-right: 15px;">\({\tt __k__ >= 4}\)</td>
      </tr>
      <tr>
        <td style="padding-left: 15px; padding-right: 15px;">\(\color{red}{\tt __k__ &lt;  a}\ \) </td>
        <td style="padding-left: 15px; padding-right: 15px;">The condition is satisfied for any value of \(\tt __k__\) smaller than \(\tt a\).</td>
        <td style="padding-left: 15px; padding-right: 15px;">\({\tt __k__ &lt; 4}\)</td>
      </tr>
      <tr>
        <td style="padding-left: 15px; padding-right: 15px;">\(\color{red}{\tt __k__ &lt;=  a}\ \) </td>
        <td style="padding-left: 15px; padding-right: 15px;">The condition is satisfied for any value of \(\tt __k__\) smaller or equal to \(\tt a\).</td>
        <td style="padding-left: 15px; padding-right: 15px;">\({\tt __k__ &lt;= 4}\)</td>
      </tr>
       <tr>
        <td style="padding-left: 15px; padding-right: 15px;">\(\color{red}{\tt Any }\ \) </td>
        <td style="padding-left: 15px; padding-right: 15px;">The condition is satisfied when \(\tt __k__\) is an arbitrary real number.</td>
        <td style="padding-left: 15px; padding-right: 15px;">\({\tt Any}\)</td>
      </tr>
       <tr>
        <td style="padding-left: 15px; padding-right: 15px;">\(\color{red}{\tt None }\ \) </td>
        <td style="padding-left: 15px; padding-right: 15px;">There is no value of \(\tt __k__\) that would satisfy the condition.</td>
        <td style="padding-left: 15px; padding-right: 15px;">\({\tt None}\)</td>
      </tr>
    </table>
    """
    return s.replace("__k__", param)



# instructions for entering solutions of systems of linear equations
# the variable include_no_sol controls if instructions for indicating
# that a system has no solutions should be displayed
def sys_eqs_instructions(include_no_sol=True):

    if include_no_sol:
        no_sol_s = r"""<li>If the system has no solutions enter <code style="font-size:18px">None</code> as the value of
        one of the variables. In such case you can leave values of the remaining variables blank."""
    else:
        no_sol_s = ""

    s = r"""
    <div style="background-color:rgb(245,245,245); padding:15px 15px 15px 15px; border: 1px solid #aeb4bf;; border-radius: 5px;">
      <b>How to enter the solution:</b>

    <ul style="padding-left:20px; padding-top:20px">
      <li>If the value of a variable is a number, just enter the number.</li>
       <li>If the value of a variable is a formula involving free variables, enter the formula. For example,
      if you obtain that \( x_1 = 1 - 2x_2 + 3x_3\) where \(x_2\) and \(x_3\) are free, then as the value of
         \(x_1\) you should enter
      <code style="font-size:18px">1 - 2*x_2 + 3*x_3</code>. Use the underscore
       <code style="font-size:18px">_</code> to indicate subscripts of variables, and <code style="font-size:18px">*</code> to indicate multiplication.
      </li>
      <li> If a variable is a free variable, enter the variable name as its value. For example, if
        \(x_2 \) is a free variable, then you should enter <code style="font-size:18px">x_2</code> as the value of
        \(x_2 \).
      </li>
      {}
    </ul>
    </div>
    """.format(no_sol_s)

    return s

# instructions for entering solutions of a matrix equation
# the variable include_no_sol controls if instructions for indicating
# that an equation has no solutions should be displayed
def matrix_eqs_instructions(include_no_sol=True):

    if include_no_sol:
        no_sol_s = r"""<li>If the equation has no solutions enter <code style="font-size:18px">None</code> as your answer."""
    else:
        no_sol_s = ""

    s = r"""
   <div style="background-color:rgb(245,245,245); padding:15px 15px 15px 15px; border: 1px solid #aeb4bf;; border-radius: 5px;">
      <b>How to enter solutions of a matrix equation \(A{{\bf x}} = {{\bf b}}\):</b>

      <br/><br/>
      Enter the solution vector \({{\bf x}} = [x_1, x_2, ..., x_n] \) as follows.
      <ul style="padding-left:20px; padding-top:20px">
      <li> If a coodinate \(x_i\) is a number,
      just enter the number.</li>
       <li>If the value of a coordinate \(x_i\) is a formula involving free variables, enter the formula. For example,
      if you obtain that \( x_1 = 1 - 2x_2 + 3x_3\) where \(x_2\) and \(x_3\) are free, then as the value of
         \(x_1\) you should enter
      <code style="font-size:18px">1 - 2*x_2 + 3*x_3</code>. Use the underscore
       <code style="font-size:18px">_</code> to indicate subscripts of variables, and <code style="font-size:18px">*</code> to indicate multiplication.
      </li>
      <li> If a coordinate \(x_i\) is a free variable, enter the variable name as its value. For example, if
        \(x_2 \) is a free variable, then you should enter <code style="font-size:18px">x_2</code> as the value of
        \(x_2 \).
      </li>
      {}
    </ul>
    <br/>
      <b>Example.</b> If the solution of an equation \(A{{\bf x}} = {{\bf b}}\) is given by a vector \({{\bf x}} = [x_1, x_2, x_3]\) where
      \(x_1 = 1\), \(x_2 = 2-5x_3\) and \(x_3\) is a free variable, when as your answer you should enter
      <code style="font-size:18px">[1, 2 - 5*x_3, x_3]</code>.

    </div>
    """.format(no_sol_s)

    return s



# instructions for entering polynomials
def polynomial_entry_instr():

    s = r"""
    <div style="background-color:rgb(245,245,245); padding:15px 15px 15px 15px; border: 1px solid #aeb4bf;; border-radius: 5px;">
    <b>How to enter polynomials.</b>
    <br/>
    Enter the formula for a polynomial using
    <code style="font-size:18px">*</code> to denote multiplication and <code style="font-size:18px">^</code> to denote the exponent.
    For example, if the answer is the polynomial
    \[ P(x) = 1 - 2x + \frac{3}{7}x^2 + 4.5x^3 \]
    then you should enter
    <code style="font-size:18px">1 - 2*x + (3/7)*x^2 + 4.5*x^3</code>.
    <br/>
    <br/>
    Do not forget to use <code style="font-size:18px">*</code> for multiplication!
    </div>
    """

    return s


#instructions for entering matrices
def matrix_entry_instructions():

    s = r"""
    <div style="background-color:rgb(245,245,245); padding:15px 15px 15px 15px; border: 1px solid #aeb4bf;; border-radius: 5px;">
        <b>How to enter matrices.</b>

          <br/>
          <br/>
          Martices should be entered row by row, enclosing each row in square brackets. There must be
          additional square brackets at the beginning and at the end of the whole matrix. For
          example, if you want to enter the matrix
                   \[
          \begin{bmatrix}
          2 &amp; -\frac{3}{2} &amp; \phantom{-}4 \\
          0 &amp; \phantom{-}\frac{1}{2} &amp; \phantom{-}2 \\
          \end{bmatrix}
          \]
          then you should do it as follows:

          <p style="text-align:center;">
            <code style="font-size:18px">[[2, -3/2, 4], [0, 1/2, 2]]</code>
          </p>
          Do not forget about commas between matrix entries and between rows.
        </div>
    """

    return s

#instructions for entering vectors
def vector_entry_instructions():

    s = r"""
    <div style="background-color:rgb(245,245,245); padding:15px 15px 15px 15px; border: 1px solid #aeb4bf;; border-radius: 5px;">
        <b>How to enter vectors.</b>

          <br/>
          <br/>
          Vectors should as a squence of numbers, separating numbers by commas, and enclosing the whole sequence in square brackets. For
          example, if you want to enter the vector
                   \[
          \begin{bmatrix}
          \phantom{-}2 \\
          \phantom{-}\frac{1}{3} \\
          -7
          \end{bmatrix}
          \]
          then you should do it as follows:

          <p style="text-align:center;">
            <code style="font-size:18px">[2, 1/3, -7]</code>
          </p>
        </div>
    """

    return s


#instructions for entering sets of vectors
def vector_set_entry_instructions():

    s = r"""
    <div style="background-color:rgb(245,245,245); padding:15px 15px 15px 15px; border: 1px solid #aeb4bf;; border-radius: 5px;">
        <b>How to enter a set of vectors.</b>

          <br/>
          <br/>
          In order to enter a set of vectors (e.g. a spanning set or a basis) enclose entries of each
          vector in square brackets and separate vectors by commas.
          For example, if you want to enter the set of vectors
          \[
          \left\{
          \begin{bmatrix}
          \phantom{-}5  \\
          -\frac{1}{3}  \\
          -1  \\
          \end{bmatrix}
          , \hskip 1mm
         \begin{bmatrix}
         -\frac{3}{2}\\
         \phantom{-}0 \\
         \phantom{-}2 \\
         \end{bmatrix}
         , \hskip 1mm
         \begin{bmatrix}
         -1 \\
         \phantom{-}\frac{1}{2} \\
         -3 \\
         \end{bmatrix}
          \right\}
          \]
          then you should do it as follows:

          <p style="text-align:center;">
            <code style="font-size:18px">[5,-1/3, -1], [-3/2, 0, 2], [-1, 1/2, -3]</code>
          </p>
        </div>
    """

    return s



#instructions for entering matrices
def matrix_entry_instructions():

    s = r"""
    <div style="background-color:rgb(245,245,245); padding:15px 15px 15px 15px; border: 1px solid #aeb4bf;; border-radius: 5px;">
        <b>How to enter matrices.</b>

          <br/>
          <br/>
          Martices should be entered row by row, enclosing each row in square brackets. There must be
          additional square brackets at the beginning and at the end of the whole matrix. For
          example, if you want to enter the matrix
                   \[
          \begin{bmatrix}
          2 &amp; -\frac{3}{2} &amp; \phantom{-}4 \\
          0 &amp; \phantom{-}\frac{1}{2} &amp; \phantom{-}2 \\
          \end{bmatrix}
          \]
          then you should do it as follows:

          <p style="text-align:center;">
            <code style="font-size:18px">[[2, -3/2, 4], [0, 1/2, 2]]</code>
          </p>
          Do not forget about commas between matrix entries and between rows.
        </div>
    """

    return s


#instructions for entering square roots
#useful p
def sq_root_instructions(precision = 3):

    s = r"""
    <div style="background-color:rgb(245,245,245); padding:15px 15px 15px 15px; border: 1px solid #aeb4bf;; border-radius: 5px;">
        <b>Note.</b>
          <br/>
          If your answer involves square roots of numbers, you can enter them using the \(\tt{{sqrt}}\) function.
          For example, if you want to enter \(\sqrt{{17}}\ \) you can do it as follows:
          <p style="text-align:center;">
            <code style="font-size:18px">sqrt(17)</code>
          </p>
          Alternatively, you can compute the value of the square root and enter it with precision of at least {}
          decimal digits.
        </div>
    """.format(precision)

    return s
