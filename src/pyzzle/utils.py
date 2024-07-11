import copy
import json
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_is_logging_on = False
TRACE_LEVEL = 5


def in_ipynb():
    """Are we in a jupyter notebook?"""
    try:
        return 'ZMQ' in get_ipython().__class__.__name__
    except NameError:
        return False


def debug_on():
    """Turn debugging logging on."""
    logging_on(logging.DEBUG)


def trace_on():
    """Turn trace logging on."""
    logging_on(TRACE_LEVEL)


def logging_on(level=logging.WARNING):
    """Turn logging on."""
    global _is_logging_on

    if not _is_logging_on:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("[%(levelname)s: %(asctime)s :"
                                               " %(name)s] %(message)s",
                                               '%Y-%m-%d %H:%M:%S'))
        console.setLevel(level)
        logging.getLogger('').addHandler(console)
        _is_logging_on = True

    log = logging.getLogger('')
    log.setLevel(level)
    for h in log.handlers:
        h.setLevel(level)


def get_logger(name):
    """Return logger with null handler added if needed."""
    if not hasattr(logging.Logger, 'trace'):
        logging.addLevelName(TRACE_LEVEL, 'TRACE')

        def trace(self, message, *args, **kwargs):
            if self.isEnabledFor(TRACE_LEVEL):
                # Yes, logger takes its '*args' as 'args'.
                self._log(TRACE_LEVEL, message, args, **kwargs)

        logging.Logger.trace = trace

    log = logging.getLogger(name)
    return log


def logging_off():
    """Turn logging off."""
    logging.getLogger('').handlers = [logging.NullHandler()]


def show_2Darray(cell, mask=None, blank="", halfspace=True, stdout=False):
    """
    Display the puzzle.

    Parameters
    ----------
    cell : ndarray
        Numpy.ndarray for display
    mask : ndarray, optional
        Numpy.ndarray for mask
    blank : str
        Blank character
    halfspace : bool
        If True, use half space
    stdout: bool
        If True, display on stdout
    """
    array = copy.deepcopy(cell)
    if mask is not None:
        array[mask == True] = "■"
    if halfspace:
        blank_display = " "
    else:
        blank_display = "  "
    if stdout or not in_ipynb():
        array = np.where(array == blank, blank_display, array)
        print(array)
    else:
        from IPython.display import display
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
    ori_i_j_words = data["words"]
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
    for key in ("name", "epoch", "nwords", "seed"):
        attrs[key] = data[key]
    return cell, mask, word_list, attrs


def export_image(fig, axl, axr, cell, words, width, height, title="", fontsize=18, oname=None, dpi=300, answer=False):
    """
    Export a puzzle image. This can be used for square puzzles only.
    Parameters
    ----------
    axl : matplotlib ax
        Puzzle board ax.
    axr : matplotlib ax
        Word list ax.
    cell : numpy ndarray
        Puzzle board.
    words : ndarray
        Word list.
    width : int
        the width of puzzle.
    height : int
        the height of puzzle.
    title : str, default ""
        Puzzle name.
    oname : str
        Output file name.
    dpi : int, default 300
        Dot per Inch.
    answer : bool, default False
        If True, export with the answer.
    """
    
    import matplotlib.pyplot as plt
    import numpy as np

    from PuzzleImage import SkeltonImage

    si = SkeltonImage(blank="")
    axl = si.get_board(axl, cell, title=title, w_count=len(words), is_answer=answer)
    axr = si.get_wordlist(axr, words=words, width=width, height=height, fontsize=fontsize, draw_copyright=True)

    if oname is None:
        plt.show()
    else:
        fig.savefig(oname, dpi=dpi, bbox_inches='tight')
    plt.close()

def show_json(fpath):
    cell, mask, _, _ = decode_json(fpath)
    show_2Darray(cell, mask)


def save_json_as_problem_image(fpath, oname, label="word list", dpi=300):
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
    x = cell.shape[1]
    y = cell.shape[0]
    if y >= x:
        fig = plt.figure(figsize=(16,8), dpi=dpi)
        ax1 = fig.add_subplot(121)  # puzzle
        ax2 = fig.add_subplot(122)  # word list
    else:
        fig = plt.figure(figsize=(8,16), dpi=dpi)
        ax1 = fig.add_subplot(211)  # puzzle
        ax2 = fig.add_subplot(212)  # word list
    fig.set_facecolor('#EEEEEE')
    ax1.axis("off")
    ax2.axis("off")
    # Draw puzzle
    ax1_table = ax1.table(cellText=df.values, cellColours=colors, cellLoc="center", bbox=[0, 0, 1, 1])
    ax1_table.auto_set_font_size(False)
    ax1_table.set_fontsize(18)
    ax1.set_title(label=title, size=20)
    # Delete unmasked cells
    if mask is not None:
        wh_mask = np.where(mask == True)
        for i, j in list(zip(wh_mask[0], wh_mask[1])):
            ax1_table._cells[i, j].set_edgecolor('#FFFFFF')
            ax1_table._cells[i, j].set_facecolor('#FFFFFF')

    # Draw word list
    if len(word_list) == 0:
        word_list = ['']
    word_list = sorted(word_list)
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


def get_rect(cover):
    """
    Return coordinates of rectangular region from cover.

    Returns
    -------
    r_min : int
       Minimum number of rows
    r_max : int
       Maximum number of rows
    c_min : int
       Minimum number of cols
    c_min : int
       Maximum number of cols
    """
    if not np.any(cover):
        from pyzzle.Exception import ZeroSizePuzzleException
        raise ZeroSizePuzzleException("The puzzle has no contents.")
    rows = np.any(cover, axis=1)
    cols = np.any(cover, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    return r_min, r_max, c_min, c_max
