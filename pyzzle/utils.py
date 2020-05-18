import copy
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    attrs : dict
    """
    with open(fpath, "rb") as f:
        data = json.load(f)
    width = data["width"]
    height = data["height"]
    ori_i_j_words = data["list"]
    mask = np.array(data["mask"])
    nwords = data["nwords"]
    
    cell = np.full([height, width], '')
    word_list = ['']*nwords
    for idx, ori_i_j_word in enumerate(ori_i_j_words):
        ori = ori_i_j_word["ori"]
        i = ori_i_j_word["i"]
        j = ori_i_j_word["j"]
        word = ori_i_j_word["word"]
        w_len = len(word)
        if ori == 0:
            cell[i:i+w_len, j] = list(word)[0:w_len]
        if ori == 1:
            cell[i, j:j+w_len] = list(word)[0:w_len]
        word_list[idx] = word
    word_list.sort(key=len)

    attrs = {}
    for key in ("dict_name", "name", "epoch", "nwords", "seed"):
        attrs[key] = data[key]
    return cell, mask, word_list, attrs

def show_json(fpath):
    cell, mask, _, _ = decode_json(fpath)
    show_2Darray(cell, mask)

def save_json_as_probrem_image(fpath, oname, label="word list", dpi=300):
    """
    fpath : str
        File path to json
    oname : str
        File name for output
    """
    cell, mask, word_list, attrs = decode_json(fpath)
    empty_cell = np.full(cell.shape, "", dtype="unicode")
    save_image(oname, empty_cell, word_list, mask=mask, title=attrs["name"], label=label, dpi=dpi)

def save_json_as_answer_image(fpath, oname, label="word list", dpi=300):
    """
    fpath : str
        File path to json
    oname : str
        File name for output
    """
    cell, mask, word_list, attrs = decode_json(fpath)
    save_image(oname, cell, word_list, mask=mask, title=attrs["name"], label=label, dpi=dpi)

def save_image(fpath, cell, word_list, mask=None, title="", label="word list", dpi=300):
    """
    Generate a puzzle image with word lists.
    
    Parameters
    ----------
    fpath : str
        Output file path
    cell : ndarray
        2D array for imaging
    word_list : array_like
        1D array of used words
    mask :
    title :
    label : str, default "[Word List]" 
        Title label for word lists
    dpi : int, default 300
        Dot-per-inch
    """
    # Generate puzzle image
    colors = np.where(cell == '', "#000000", "#FFFFFF")
    df = pd.DataFrame(cell)
    fig = plt.figure(figsize=(16,8), dpi=dpi)
    fig.set_facecolor('#EEEEEE')
    ax1 = fig.add_subplot(121)  # puzzle
    ax2 = fig.add_subplot(122)  # word list
    ax1.axis("off")
    ax2.axis("off")
    # Draw puzzle
    ax1_table = ax1.table(cellText=df.values, cellColours=colors, cellLoc="center", bbox=[0, 0, 1, 1])
    ax1_table.auto_set_font_size(False)
    ax1_table.set_fontsize(18)
    ax1.set_title(label=title, size=20)
    # Delete unmasked cells
    if mask is not None:
        wh_mask = np.where(mask == False)
        for i, j in list(zip(wh_mask[0], wh_mask[1])):
            del ax1_table._cells[i, j]
    # Draw word list
    if word_list == []:
        word_list = ['']
    word_list = sorted(word_list, key=len)
    rows = cell.shape[0]
    cols = np.ceil(len(word_list) / rows).astype("int")
    padnum = cols * rows - len(word_list)
    word_list += [''] * int(padnum)
    word_list = np.array(word_list).reshape(cols, rows).T
    ax2_table = ax2.table(cellText=word_list, cellColours=None, cellLoc="left", edges="open", bbox=[0, 0, 1, 1])
    ax2.set_title(label=label, size=20)
    ax2_table.auto_set_font_size(False)
    ax2_table.set_fontsize(18)
    plt.tight_layout()
    plt.savefig(fpath, dpi=dpi)
    plt.close()