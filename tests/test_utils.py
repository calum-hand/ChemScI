import pytest

from chemsci.utils import (
    determine_default_or_callable
)

from chemsci.exceptions import (
    UserSelectionError
)

# ----------------------------------------------------------------------------------------------------------------------


def func():
    return True


class Call:
    def __call__(self):
        return True


SWITCH_DICT = {'func': func, 'callable': Call()}

# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("arg",
                         [
                             'func',
                             'callable',
                             func,
                             Call()
                         ])
def test_determine_default_or_callable_pass(arg):
    """Confirm tests pass for below:
    * key with function value
    * key with callable obj value (initialised in dict)
    * custom func
    * custom callable (initialised prior to passing)

    Parameters
    ----------
    arg : str or callable object

    """
    obj = determine_default_or_callable(arg, SWITCH_DICT)
    out = obj()
    assert out is True, 'Calling object should return True.'


@pytest.mark.parametrize("arg",
                         [
                             'Not_a_valid_key_or_callable',
                         ])
def test_determine_default_or_callable_fail(arg):
    """Should raise error if invalid dict key / non callable object is passed.

    Parameters
    ----------
    arg : str or callable object
    """
    with pytest.raises(UserSelectionError):
        determine_default_or_callable(arg, SWITCH_DICT)


# ----------------------------------------------------------------------------------------------------------------------
