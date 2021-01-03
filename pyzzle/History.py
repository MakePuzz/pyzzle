from enum import Enum


class HistoryCode(Enum):
    """
    Enumeration of the history.

    The result number corresponds to the judgment result
    1. Add
    2. Drop
    3. Drop(in Kick process)
    4. Move
    """
    ADD = 1
    DROP = 2
    DROP_KICK = 3
    MOVE = 4


class HistoryItem(list):
    def __init__(self, code, ori, i, j, word):
        super().__init__([code, ori, i, j, word])
    @property
    def code(self):
        return self[0]
    @property
    def ori(self):
        return self[1]
    @property
    def i(self):
        return self[2]
    @property
    def j(self):
        return self[3]
    @property
    def word(self):
        return self[4]
