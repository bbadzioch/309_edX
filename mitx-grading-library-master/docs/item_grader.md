# ItemGrader

When an individual input needs to be graded, it is graded by an item grader. All item graders work by specifying answers and their corresponding points/messages, as well as an optional message for wrong answers. In these examples, we use `StringGrader` as an example of how to use a generic ItemGrader. You cannot use a generic `ItemGrader` by itself.

```python
grader = StringGrader(
    answers='cat',
    wrong_msg='Try again!'
)
```

The `answers` key can be used to specify correct answers, specific feedback messages, and to assign partial credit. It accepts a few formats:

- A single dictionary can be used to specify an answer, feedback, correctness, and partial credit. The dictionary keys are:

    - `'expect'` (required): compared against student answer. Most ItemGraders use strings to specify the `'expect'` value.
    - `'grade_decimal'` (optional, a number between `0` and `1`): The credit associated with this answer; default value is `1`.
    - `'ok'` (optional, `True`, `False`, or `'partial'`): The answer's correctness; determines icon used by edX. The default value is inferred from `grade_decimal`.
    - `'msg'` (optional, string): A feedback message associated with this answer.

```python
grader = StringGrader(
    answers={'expect': 'zebra', 'ok': True, 'grade_decimal': 1, 'msg': 'Yay!'},
    wrong_msg='Try again!'
)
```

- A single `'expect'` value: can be used to specify the correct answer. For example,

```python
grader = StringGrader(
    answers='cat',
    # Equivalent to:
    # answers={'expect': 'cat', 'msg': '', 'grade_decimal': 1, 'ok': True}
    wrong_msg='Try again!'
)
```

  Again, most ItemGraders use strings to store `'expect'` values.

- A tuple of dictionaries or strings:

```python
grader = StringGrader(
    answers=(
        # the correct answer
        'wolf',
        # an alternative correct answer
        'canis lupus',
        # a partially correct answer
        {'expect': 'dog', 'grade_decimal': 0.5, 'msg': 'No, not dog!'},
        # a wrong answer with specific feedback
        {'expect': 'unicorn', 'grade_decimal': 0, 'msg': 'No, not unicorn!'}
    ),
    wrong_msg='Try again!'
)
```

Internally, the `ItemGrader` converts the answers entry into a tuple of dictionaries. When grading, it asks the specific grading class to grade the response against each possible answer, and selects the best outcome for the student.

The `wrong_msg` is only displayed if the score is zero and there are no other messages.

If no `answers` key is provided, the grader reads from the `expect` or `answer` parameter of the `customresponse` tag (see below). However, note that when using a `ListGrader` or a `SingleListGrader`, the `answers` key is required.


## Option Listing

Here is the full list of options specific to an `ItemGrader`.
```python
grader = ItemGrader(
    answers=answer or tuple of answers,
    wrong_msg=str (default '')
)
```
