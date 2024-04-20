""" General utility functions. """

from collections import defaultdict
from collections.abc import MutableMapping
import random
import string


def default_to_regular_dict(d):
    """Helper function to recursively convert defaultdict into a regular Python dict.

    Parameters
    ----------
    d: defaultdict or dict
        Original dictionary.

    Returns
    -------
    d: dict
        Python dictionary.

    """
    if isinstance(d, defaultdict):
        d = {k: default_to_regular_dict(v) for k, v in d.items()}
    return d


def flatten_dict(d, parent_key="", sep="_"):
    """Helper function to recursively flatten nested dictionary.

    Parameters
    ----------
    d: defaultdict or dict
        Original dictionary.
    parent_key: str, optional
        Parent key. Deafult is ''.
    sep: str, optional
        Seperator to seperate keys. Default is '_'.

    Returns
    -------
    d: dict
        Flattened dictionary.

    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_random_string(k=8):
    """Get random string of specified length.

    Parameters
    ----------
    k: int, optional
        Length of random string.

    Returns
    -------
    str
        Randomly generated string.

    """
    return "".join(random.choices(string.ascii_lowercase, k=k))
