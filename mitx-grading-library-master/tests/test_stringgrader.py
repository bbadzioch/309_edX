"""
Tests for StringGrader
"""
from mitxgraders import StringGrader

def test_strip():
    """Tests that strip is working correctly"""
    grader = StringGrader(answers="cat")
    assert grader(None, "cat")['ok']
    assert grader(None, "   cat   ")['ok']
    assert not grader(None, "c at")['ok']
    assert not grader(None, "  dog  ")['ok']

    grader = StringGrader(answers="         cat")
    assert grader(None, "cat")['ok']
    assert grader(None, "   cat   ")['ok']
    assert not grader(None, "c at")['ok']
    assert not grader(None, "  dog  ")['ok']

    grader = StringGrader(answers="cat", strip=False)
    assert grader(None, "cat")['ok']
    assert not grader(None, " cat")['ok']
    assert not grader(None, " cat ")['ok']
    assert not grader(None, "cat ")['ok']

    grader = StringGrader(answers=" cat", strip=False)
    assert not grader(None, "cat")['ok']
    assert grader(None, " cat")['ok']
    assert not grader(None, " cat ")['ok']
    assert not grader(None, "cat ")['ok']

    grader = StringGrader(answers=" cat", strip=True)
    assert grader(None, "cat")['ok']
    assert grader(None, " cat")['ok']
    assert grader(None, " cat ")['ok']
    assert grader(None, "cat ")['ok']

def test_case():
    """Tests that case is working correctly"""
    grader = StringGrader(answers="cat", case_sensitive=True)
    assert grader(None, "cat")['ok']
    assert not grader(None, "Cat")['ok']
    assert not grader(None, "CAT")['ok']

    grader = StringGrader(answers="Cat", case_sensitive=True)
    assert not grader(None, "cat")['ok']
    assert grader(None, "Cat")['ok']
    assert not grader(None, "CAT")['ok']

    grader = StringGrader(answers="CAT", case_sensitive=True)
    assert not grader(None, "cat")['ok']
    assert not grader(None, "Cat")['ok']
    assert grader(None, "CAT")['ok']

def test_any():
    """Tests that accept_any is working correctly"""
    grader = StringGrader(accept_any=True)
    assert grader(None, "cat")['ok']
    assert grader(None, "dog")['ok']
    assert grader(None, "")['ok']
    assert grader(None, " ")['ok']

def test_nonempty():
    """Tests that accept_nonempty is working correctly"""
    grader = StringGrader(accept_nonempty=True)
    assert grader(None, "cat")['ok']
    assert grader(None, "dog")['ok']
    assert not grader(None, "")['ok']
    assert not grader(None, " ")['ok']


# Documentation tests

def test_docs():
    """Test that the documentation examples work as intended"""
    # Opening example
    grader = StringGrader(
        answers='cat'
    )
    assert grader(None, 'cat')['ok']
    assert not grader(None, 'Cat')['ok']
    assert not grader(None, 'CAT')['ok']

    # Case sensitive
    grader = StringGrader(
        answers='Cat',
        case_sensitive=False
    )
    assert grader(None, 'cat')['ok']
    assert grader(None, 'Cat')['ok']
    assert grader(None, 'CAT')['ok']

    # Strip
    grader = StringGrader(
        answers=' cat',
        strip=False
    )
    assert not grader(None, 'cat')['ok']
    assert grader(None, ' cat')['ok']
    assert not grader(None, ' cat ')['ok']
    assert not grader(None, 'cat ')['ok']
