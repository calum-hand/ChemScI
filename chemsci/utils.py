from chemsci.exceptions import UserSelectionError

# ----------------------------------------------------------------------------------------------------------------------


def determine_default_or_callable(arg, default_dict):
    """Logic used to allow user to select a predefined argument from a dictionary or pass their own
    callable object (i.e. function or class) instead.
    If passed `arg` is not a key in `default_dict` then it will check if `arg` is a callable and return `arg` instead.

    Notes
    -----
    If a function is passed as `arg` then it should be passed without calling function.
    If a callable object (class) is passed however then the object should be initialised.

    Raises
    ------
    UserSelectionError : If passed `arg` not in `default_dict` and not a callable object.

    Parameters
    ----------
    arg : str or callable
        A `str` present in `default_dict` or a callable object / function.

    default_dict : dict
        Dictionary of default options to select from.

    Returns
    -------
    obj : callable (function or object)
        Callable python object either present in `default_dict` or passed as `arg`.
    """
    if arg in default_dict:
        obj = default_dict[arg]
    elif callable(arg):
        obj = arg
    else:
        raise UserSelectionError(F'Passed argument {arg} must be callable or entry in {default_dict.keys()}')
    return obj

