from enum import Enum


class History(Enum):
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


# class History(list):
#     def log(self, ori, i, j, k, code):
#         self.append()
