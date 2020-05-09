import numpy as np
import pandas as pd
from IPython.display import display


def in_ipynb():
    """Are we in a jupyter notebook?"""
    try:
        return 'ZMQ' in get_ipython().__class__.__name__
    except NameError:
        return False


def show_2Darray(ndarray, mask=None):
    """
    Display the puzzle.

    Parameters
    ----------
    ndarray : ndarray
        Numpy.ndarray for display
    mask : ndarray, optional
        Numpy.ndarray for mask
    """
    if mask is not None:
        ndarray[mask == False] = "■"
    if in_ipynb() is True:
        styles = [
            dict(selector="th", props=[("font-size", "90%"),
                                        ("text-align", "center"),
                                        ("color", "#ffffff"),
                                        ("background", "#777777"),
                                        ("border", "solid 1px white"),
                                        ("width", "30px"),
                                        ("height", "30px")]),
            dict(selector="td", props=[("font-size", "105%"),
                                        ("text-align", "center"),
                                        ("color", "#161616"),
                                        ("background", "#dddddd"),
                                        ("border", "solid 1px white"),
                                        ("width", "30px"),
                                        ("height", "30px")]),
            dict(selector="caption", props=[("caption-side", "bottom")])
        ]
        df = pd.DataFrame(ndarray)
        df = (df.style.set_table_styles(styles))
        display(df)
    else:
        ndarray = np.where(ndarray == "", "  ", ndarray)
        print(ndarray)