"""
utils
"""


def in_ipynb():
    """Are we in a jupyter notebook?"""
    try:
        return 'ZMQ' in get_ipython().__class__.__name__
    except NameError:
        return False
