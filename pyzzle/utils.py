import copy
import json

import numpy as np
import pandas as pd
from IPython.display import display


def in_ipynb():
    """Are we in a jupyter notebook?"""
    try:
        return 'ZMQ' in get_ipython().__class__.__name__
    except NameError:
        return False


def show_2Darray(cell, mask=None):
    """
    Display the puzzle.

    Parameters
    ----------
    cell : ndarray
        Numpy.ndarray for display
    mask : ndarray, optional
        Numpy.ndarray for mask
    """
    array = copy.deepcopy(cell)
    if mask is not None:
        array[mask == False] = "■"
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
        df = pd.DataFrame(array)
        df = (df.style.set_table_styles(styles))
        display(df)
    else:
        array = np.where(array == "", "  ", array)
        print(array)

def decode_json(fpath):
    """
    Parameters
    ----------
    fpath : str
        File path to json.

    Returns
    -------
    cell : ndarray
    mask : ndarray
    word_list : list
    """
    with open(fpath, "rb") as f:
        data = json.load(f)
    dict_name = data["dict_name"]
    width = data["width"]
    height = data["height"]
    ori_i_j_words = data["list"]
    mask = np.array(data["mask"])
    name = data["name"]
    nwords = data["nwords"]
    epoch = data["epoch"]
    seed = data["seed"]
    
    cell = np.full([height, width], '')
    word_list = ['']*nwords
    for i, ori_i_j_word in enumerate(ori_i_j_words):
        ori = ori_i_j_word["ori"]
        i = ori_i_j_word["i"]
        j = ori_i_j_word["j"]
        word = ori_i_j_word["word"]
        w_len = len(word)
        if ori == 0:
            cell[i:i+w_len, j] = list(word)[0:w_len]
        if ori == 1:
            cell[i, j:j+w_len] = list(word)[0:w_len]
        word_list[i] = word
    word_list.sort(key=len)
    return cell, mask, word_list

def show_json(fpath):
    cell, mask, _ = decode_json(fpath)
    print(mask)
    show_2Darray(cell, mask)