import copy
import json
import logging

import numpy as np
import pandas as pd
import japanize_matplotlib
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

def show_2Darray(cell, mask=None, blank="", stdout=False):
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
        array[mask == True] = "■"
    if stdout is True or in_ipynb() is False:
        array = np.where(array == blank, " ", array)
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
    for key in ("name", "epoch", "nwords", "seed"):
        attrs[key] = data[key]
    return cell, mask, word_list, attrs

def export_image(puzzle, words, title="", wn=15, oname='problem.png', draw_type=0, dpi=300, answer=False):
    """
    export the image of puzzle boards
    this fuction can be used when the board is square

    Parameters
    ----------
    puzzle : ndarray
        Array of words in the puzzle board
    words : ndarray
        The list of words in this puzzle
    title : str, default ""
        The name of this puzzle
    wn : int, default 15
        Square side length of the board
    oname : str, "problem.png"
        The name of output file
    draw_type : int, default 0
        The type of drawing (0:gray filling and outer frame  1:no filling and no outer frame)
    dpi : int, default 300
        The number of dpi
    answer : bool, default False
        Output the answer sheet or problem sheet (True:answer sheet  False:problem sheet)
    """
    words = np.array(sorted(words, key=lambda word: (len(word), word)))
    w_lens = np.vectorize(len)(words)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw=dict(width_ratios=[9,7], wspace=-0.1))
    ax1.axis("off")
    ax1.set(aspect="equal", xlim=(0,wn), ylim=(0,wn))
    ax2.axis("off")

    # Board creation
    for i in range(wn):
        for j in range(wn):
            # Horizontal line params
            y = wn - 1 - j
            xmin = i / wn
            xmax = (i+1) / wn
            # Vertical line params
            x = i + 1
            ymin = (wn-1-j) / wn
            ymax = (wn-1-j+1) / wn

            # thin lines
            if j <= wn-2 and puzzle[j,i] != '' and puzzle[j+1,i] != '':
                ax1.axhline(y=y, xmin=xmin, xmax=xmax, color='#EBEBEB', ls='-', lw=1)     
            if i <= wn-2 and puzzle[j,i] != '' and puzzle[j,i+1] !='':
                ax1.axvline(x=x, ymin=ymin, ymax=ymax, color='#EBEBEB', ls='-', lw=1)

            #　bold lines
            if j <= wn-2 and puzzle[j,i] != '' and puzzle[j+1,i] == '':
                ax1.axhline(y=y, xmin=xmin, xmax=xmax, color='k', ls='-',lw=1, zorder=4)
            if j <= wn-2 and puzzle[j,i] == '' and puzzle[j+1,i] != '':
                ax1.axhline(y=y, xmin=xmin, xmax=xmax, color='k', ls='-',lw=1, zorder=4)    
            if i <= wn-2 and puzzle[j,i] != '' and puzzle[j,i+1] == '':
                ax1.axvline(x=x, ymin=ymin, ymax=ymax, color='k', ls='-', lw=1, zorder=4)
            if i <= wn-2 and puzzle[j,i] == '' and puzzle[j,i+1] != '':
                ax1.axvline(x=i+1, ymin=ymin, ymax=ymax, color='k', ls='-', lw=1, zorder=4)

    if draw_type == 0:
        # Outer lines
        ax1.plot([0, 0, wn, wn, 0], [0, wn, wn, 0, 0], color='k', ls='-', lw=4, zorder=4)
        # Fill empty cells
        cmap = plt.cm.viridis
        cmap.set_over("#f5efe6", alpha=1)
        cmap.set_under("white", alpha=0)
        ax1.imshow(puzzle=="", extent=[0,wn,0,wn], cmap=cmap, vmin=0.5, vmax=0.6)
    
    if draw_type == 1:
        for j in range(wn):
            ymin = (wn-1-j+0.05) / wn
            ymax = (wn-1-j+1-0.05) / wn
            if puzzle[0,j] != '':
                ax1.axvline(x=0, ymin=ymin, ymax=ymax, color='k', ls='-', lw=4, zorder=4)
            if puzzle[wn-1,j] != '':
                ax1.axvline(x=wn, ymin=ymin, ymax=ymax, color='k', ls='-', lw=4, zorder=4)
        for i in range(wn):
            xmin = (i+0.05) / wn
            xmax = (i+1-0.05) / wn
            if puzzle[i,wn-1] != '':
                ax1.axhline(y=0, xmin=xmin, xmax=xmax, color='k', ls='-',lw=4, zorder=4)
            if puzzle[i,0] != '':
                ax1.axhline(y=wn, xmin=xmin, xmax=xmax, color='k', ls='-',lw=4, zorder=4)

    # puzzle title and copyright
    w_num = len(words)
    ax1.text(0.1, 15.2, f'{title}', size=16, ha='left', color='#1a1a1a')
    ax1.text(15, 15.1, f'{w_num}語', size=12, ha='right', color='#1a1a1a')
    ax2.text(0.95, -0.01, '© MakePuzz', size=18, ha='right', fontname='Yu Gothic', alpha=0.5, fontweight='bold')

    # Word list creation
    col_num = 3
    char_max_per_row = 21
    row_num = np.ceil(w_num/col_num).astype(int) # row_num = np.ceil(w_num/20)
    char_num_per_row = w_lens[row_num-1] + w_lens[2*row_num-1] + w_lens[w_num-1] + 2 + 4

    # penetrate check
    pene_words_count = 0
    if char_num_per_row > char_max_per_row:
        char_num_at_row_2to3 = (char_max_per_row - 2 - w_lens[row_num-1]) # Subtract the left column from the whole
        peneall = bool(char_num_at_row_2to3 < w_lens[w_num-1])
        while char_num_per_row > char_max_per_row:
            pene_words_count += 1
            char_num_per_row = w_lens[row_num-1] + w_lens[2*row_num-1] + w_lens[w_num-1-pene_words_count] + 2 + 4

    # no penetration
    if pene_words_count == 0:
        # row spacing
        if row_num <= 10:
            row_spacing = 0.05 + 0.05
        if row_num <= 15:
            row_spacing = 0.015 + 0.05
        if row_num > 15:
            row_spacing = 0.05
        row_num_at_col_1 = row_num
        row_num_at_col_3 = w_num - 2 * row_num

    # penetration
    if pene_words_count > 0:
        if peneall is True: # all penetration
            # row spacing
            row_num_plus_pene_num = row_num + pene_words_count
            if row_num_plus_pene_num <= 10:
                row_spacing = 0.05 + 0.05
            if row_num_plus_pene_num <= 15:
                row_spacing = 0.015 + 0.05
            if row_num_plus_pene_num > 15:
                row_spacing = 0.05
            
            if pene_words_count > (20-row_num): # row_num adjust
                row_num = 20 - pene_words_count
            row_num_at_col_1 = row_num
            row_num_at_col_3 = w_num - 2 * row_num - pene_words_count
        if peneall is False: # Penetration appears in the right two columns
            row_spacing = 0.05
            if pene_words_count > (20-row_num):
                row_num = 20 - pene_words_count # row_num adjust
            row_num_at_col_1 = 20
            row_num_at_col_3 = w_num - 20 - row_num - pene_words_count

    # checkbox params
    width = 0.015
    height = 0.015

    # 1st column
    box_x = 0.047
    ymax = 0.985
    label_x = 0.02
    w = 0
    for n in range(row_num_at_col_1):
        # checkbox
        box_y = 0.965 - row_spacing * n
        fancybox = mpatches.FancyBboxPatch((box_x,box_y), width, height, boxstyle=mpatches.BoxStyle("Round", pad=0.005), fc="#f5efe6", ec="darkgray", alpha=1)
        ax2.add_patch(fancybox)
        # word
        word_y = 0.97 - row_spacing * n
        ax2.text(0.08, word_y, words[w], size=18, ha='left', va='center')
        # label
        if w == 0 or w_lens[w] > w_lens[w-1]:
            ax2.text(label_x, word_y, str(w_lens[w]), fontsize=10, color='dimgray', ha='right')
        # label line
        if w_lens[w] > w_lens[w-1]:
            ax2.axvline(x=label_x+0.01, color='lightgray', ymin=0.96-row_spacing*(n-1), ymax=ymax, lw=2)
            ymax = 0.985 - row_spacing * n
        # end of label line
        elif n == row_num_at_col_1-1:
            ax2.axvline(x=label_x+0.01, color='lightgray', ymin=0.96-row_spacing*n, ymax=ymax, lw=2)
        w += 1

    # 2nd column
    col_spacing = (w_lens[row_num_at_col_1]-3) * 0.05
    box_x = 0.277 + col_spacing
    ymax = 0.985
    label_x = 0.25 + col_spacing
    for n in range(row_num):
        # checkbox
        box_y = 0.965 - row_spacing * n
        fancybox = mpatches.FancyBboxPatch((box_x,box_y), width, height, boxstyle=mpatches.BoxStyle("Round", pad=0.005), fc="#f5efe6", ec="darkgray", alpha=1)
        ax2.add_patch(fancybox)
        # word
        word_y = 0.97 - row_spacing * n
        ax2.text(0.31+col_spacing, word_y, words[w], size=18, ha='left', va='center')
        # label
        if w == 0 or w_lens[w] > w_lens[w-1]:
            ax2.text(label_x, word_y, str(w_lens[w]), fontsize=10, color='dimgray', ha='right')
        # label line
        if w_lens[w] > w_lens[w-1]:
            ax2.axvline(x=label_x+0.01, color='lightgray', ymin=0.96-row_spacing*(n-1), ymax=ymax, lw=2)
            ymax = 0.985 - row_spacing * n
        # end of label line
        elif n == row_num-1:
            ax2.axvline(x=label_x+0.01, color='lightgray', ymin=0.96-row_spacing*n, ymax=ymax, lw=2)
        w += 1

    # 3rd column
    col_spacing = (w_lens[w]-5) * 0.05
    box_x = 0.597 + col_spacing
    ymax = 0.985
    label_x = 0.57+col_spacing
    for n in range(row_num_at_col_3):
        # checkbox
        box_y = 0.965 - row_spacing * n
        fancybox = mpatches.FancyBboxPatch((box_x,box_y), width, height, boxstyle=mpatches.BoxStyle("Round", pad=0.005), fc="#f5efe6", ec="darkgray", alpha=1)
        ax2.add_patch(fancybox)
        # word
        word_y = 0.97 - row_spacing * n
        ax2.text(0.63+col_spacing, word_y, words[w], size=18, ha='left', va='center')
        # label
        if w == 0 or w_lens[w] > w_lens[w-1]:
            ax2.text(label_x, word_y, str(w_lens[w]), fontsize=10, color='dimgray', ha='right')
        if w_lens[w] > w_lens[w-1]:
            ax2.axvline(x=label_x+0.01, color='lightgray', ymin=0.96-row_spacing*(n-1), ymax=ymax, lw=2)
            ymax = 0.985 - row_spacing * n
        elif n == (row_num_at_col_3 - 1):
            ax2.axvline(x=label_x+0.01, color='lightgray', ymin=0.96-row_spacing*n, ymax=ymax, lw=2)
        w += 1

    # penetrating column
    if pene_words_count > 0 and peneall is True:
        box_x = 0.047
        col_spacing = 0
        ymax = 0.985 - row_spacing * row_num - 0.025
        label_x = 0.02
        for n in range(pene_words_count):
            # checkbox
            box_y = 0.965 - row_spacing * (row_num+n) - 0.025
            fancybox = mpatches.FancyBboxPatch((box_x,box_y), width, height, boxstyle=mpatches.BoxStyle("Round", pad=0.005),fc="#f5efe6", ec="darkgray", alpha=1)
            ax2.add_patch(fancybox)
            # word
            word_y = 0.97-row_spacing*(row_num+n)-0.025
            ax2.text(0.08, word_y, words[w], size=18, ha='left', va='center')
            # label
            if n == 0 or w_lens[w] > w_lens[w-1]:
                ax2.text(label_x, word_y, str(w_lens[w]), fontsize=10, color='dimgray',  ha='right')
            # label line
            if n != 0 and w_lens[w] > w_lens[w-1]:
                ax2.axvline(x=label_x+0.01, color='lightgray', ymin=0.96-row_spacing*(row_num+n-1)-0.025, ymax=ymax, lw=2)
                ymax = 0.985 - row_spacing * n - 0.025
            # end of label line
            elif n == pene_words_count-1:
                ax2.axvline(x=label_x+0.01, color='lightgray', ymin=0.96-row_spacing*(row_num+n)-0.025, ymax=ymax, lw=2)
            w += 1

    if pene_words_count > 0 and peneall is False:
        col_spacing = (w_lens[row_num_at_col_1]-3) * 0.05
        box_x = 0.277 + col_spacing
        ymax = 0.985 - row_spacing * row_num - 0.025
        label_x = 0.25+col_spacing
        for n in range(pene_words_count):
            # checkbox
            box_y = 0.965 - row_spacing * (row_num+n) - 0.025
            fancybox = mpatches.FancyBboxPatch((box_x,box_y), width, height, boxstyle=mpatches.BoxStyle("Round", pad=0.005), fc="#f5efe6", ec="darkgray", alpha=1)
            ax2.add_patch(fancybox)
            # word
            word_y = 0.97-row_spacing*(row_num+n)-0.025
            ax2.text(0.31+col_spacing, word_y, words[w], size=18, ha='left', va='center')
            # label
            if n == 0 or w_lens[w] > w_lens[w-1]:
                ax2.text(label_x, word_y, str(w_lens[w]), fontsize=10, color='dimgray', ha='right')
            if n != 0 and w_lens[w] > w_lens[w-1]:
                ax2.axvline(x=label_x+0.01, color='lightgray', ymin=0.96-row_spacing*(row_num+n-1)-0.025, ymax=ymax, lw=2)
                ymax = 0.985 - row_spacing * n - 0.025
            elif n == pene_words_count-1:
                ax2.axvline(x=label_x+0.01, color='lightgray', ymin=0.96-row_spacing*(row_num+n)-0.025, ymax=ymax, lw=2)
            w += 1

    if answer is False:
        fig.savefig(oname, dpi=dpi, bbox_inches='tight')
        return
    
    # Answer image
    # alphabet .35 .25, Hiwagana .15 .25
    for i in range(wn):
        for j in range(wn):
            x = i + 0.5
            y = wn - j - 0.6
            rotation = 0
            # The rotation process for vertical long tones
            if puzzle[j,i] == 'ー' and i >= 1 and puzzle[j,i-1] == '':
                x += 0.01
                y += 0.15
                rotation = 90
            ax1.text(x, y, puzzle[j,i], size=18, ha="center", va="center", rotation=rotation)
    fig.savefig(oname, dpi=dpi, bbox_inches='tight')
    return

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

def get_rect(cell):
    """
    Return coordinates of rectangular region

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
    rows = np.any(cell, axis=1)
    cols = np.any(cell, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    return r_min, r_max, c_min, c_max
