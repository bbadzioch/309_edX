# ListGrader

A `ListGrader` is used to grade a list of student inputs wherein each input is entered in a separate answer box. (In contrast, `SingleListGrader` can be used to grade a list of items entered all at once into a single answer box.) ListGraders work by farming out individual items to subgraders, and then collecting the results and working out the optimal farming scheme for the student.


## Basic usage

In this example, each input is checked against the corresponding answer, using `StringGrader` as the subgrader.

```python
grader = ListGrader(
    answers=['cat', 'dog'],
    subgraders=StringGrader()
)
```

Each element of answers is set as an answer that is passed as the answers key into the subgrader. This should be set up as two input boxes that the student types in. Note that the `answers` key is provided as a python list of individual answers for a `ListGrader`.

In the above example, the item grader just sees single strings as the answer. You can do more complicated things though, like the following.

```python
answer1 = (
    {'expect': 'zebra', 'grade_decimal': 1},
    {'expect': 'horse', 'grade_decimal': 0.45},
    {'expect': 'unicorn', 'grade_decimal': 0, 'msg': 'Unicorn? Really?'}
)
answer2 = (
    {'expect': 'cat', 'grade_decimal': 1},
    {'expect': 'feline', 'grade_decimal': 0.5}
)
grader = ListGrader(
    answers=[answer1, answer2],
    subgraders=StringGrader()
)
```
In this example, the grader will try assigning the first input to answer1 and the second to answer2, and computing the total score. Then it will repeat, with the inputs switched. The student will receive the highest grade. So, note that while `cat` and `unicorn` will get the unicorn message (and 1/2 points), `zebra` and `unicorn` will not (and also get 1/2 points).


## Ordered Input

By default, the ListGrader doesn't care what order the inputs are given in, so "cat" and "dog" is equivalent to "dog" and "cat". If you want the inputs to be ordered, simply set ordered to True.

```python
grader = ListGrader(
    answers=['cat', 'dog'],
    subgraders=StringGrader(),
    ordered=True
)
```

Now, "cat" and "dog" will receive full credit, but "dog" and "cat" will receive none.


## Multiple Graders

If you have inhomogeneous inputs, you can grade them using different graders. Simply give a list of subgraders, and the data will be passed into the graders in that order. Note that the length of answers must be the same as the number of subgraders in this case. Further note that you must set ordered to True when using a list of subgraders.

```python
grader = ListGrader(
    answers=['cat', 'x^2+1'],
    subgraders=[StringGrader(), FormulaGrader(variables=["x"])],
    ordered=True
)
```


## SingleListGraders in ListGrader

Some questions will require nested list graders. Simple versions can make use of a `SingleListGrader` subgrader, as in the following example.

Consider two input boxes, where the first should be a comma-separated list of even numbers beneath 5, and the second should be a comma-separated list of odd numbers beneath 5. The order of the boxes is important, but within each box, the order becomes unimportant. Here's how you can encode this type of problem.

```python
grader = ListGrader(
    answers=[
        ['2', '4'],
        ['1', '3']
    ],
    subgraders=SingleListGrader(
        subgrader=NumericalGrader()
    ),
    ordered=True
)
```

The nested `SingleListGrader` will be used to grade the first input box against an unordered answer of 2 and 4, and then the second input box against an unordered answer of 1 and 3.


## Grouped Inputs

If you find yourself wanting to nest ListGraders, then you will need to specify how the inputs should be grouped together to be passed to the subgraders. A simple example would be to ask for the name and number of each animal in a picture. Each name/number group needs to be graded together. Here is an example of such a question.

```python
grader = ListGrader(
    answers=[
        ['cat', '1'],
        ['dog', '2'],
        ['tiger', '3']
    ],
    subgraders=ListGrader(
        subgraders=[StringGrader(), NumericalGrader()],
        ordered=True
    ),
    grouping=[1, 1, 2, 2, 3, 3]
)
```

In this case, the second level of grader is receiving multiple inputs, and so itself needs to be a ListGrader. The grouping key specifies which group each input belongs to. In this case, answers 1 and 2 will be combined into a list and fed to the subgrader as group 1, as will 3 and 4 as group 2, and 5 and 6 as group 3. The third level of grader (StringGrader and NumericalGrader) will then receive a list of two inputs, and each of the items in the answers. Because this is an unordered list, the `ListGrader` will find the optimal ordering of (animal, number) pairs.

The grouping keys must be integers starting at 1 and increasing. If you have N groups, then all numbers from 1 to N must be present in the grouping, but they need not be in monotonic order. So for example, [1, 2, 1, 2] is a valid grouping. For unordered groups, the groupings must each have the same number of elements.

Here is another example. In this case, we have ordered entry, so we can specify a list of subgraders. We have three items in the first grouping and one item in the second, so we use a `ListGrader` for the first grouping, and a `StringGrader` for the second. Note that the first entry in answers is a list that is passed directly into the `ListGrader`, while the second entry is just a string. This second-level `ListGrader` is unordered.

```python
grader = ListGrader(
    answers=[
        ['bat', 'ghost', 'pumpkin'],
        'Halloween'
    ],
    subgraders=[
        ListGrader(
            subgraders=StringGrader()
        ),
        StringGrader()
    ],
    ordered=True,
    grouping=[1, 1, 1, 2]
)
```

Our last pair of examples are for a math class, where we have a matrix that has two eigenvalues, and each eigenvalue has a corresponding eigenvector. We start by grouping the eigenvalue and eigenvector boxes together, and then grade the groups in an unordered fashion. The eigenvectors are normalized, but have a sign ambiguity. A tuple contains both possible answers, and the grader will accept either of them.

```python
grader = ListGrader(
    answers=[
        ['1', (['1', '0'], ['-1', '0'])],
        ['-1', (['0', '1'], ['0', '-1'])],
    ],
    subgraders=ListGrader(
        subgraders=[
            NumericalGrader(),
            SingleListGrader(
                subgrader=NumericalGrader(),
                ordered=True
            )
        ],
        ordered=True
    ),
    grouping=[1, 1, 2, 2]
)
```

This example has four input boxes, with the first and third being graded by a `NumericalGrader`, and the second and fourth being graded by a `SingleListGrader`.

It is possible to specify a grouping on a nested `ListGrader`. The outer `ListGrader` must also have a grouping specified if doing so. Here is the same grader as above, where instead of taking the eigenvectors in a single input box list, there are four boxes to input each of the four vector components.

```python
grader = ListGrader(
    answers=[
        ['1', (['1', '0'], ['-1', '0'])],
        ['-1', (['0', '1'], ['0', '-1'])],
    ],
    subgraders=ListGrader(
        subgraders=[
            NumericalGrader(),
            ListGrader(
                subgraders=NumericalGrader(),
                ordered=True
            )
        ],
        ordered=True,
        grouping=[1, 2, 2]
    ),
    grouping=[1, 1, 1, 2, 2, 2]
)
```


## Option Listing

Here is the full list of options specific to a `ListGrader`.
```python
grader = ListGrader(
    answers=list,
    subgraders=AbstractGrader or list of AbstractGraders,
    ordered=bool, (default False)
    grouping=list
)
```
