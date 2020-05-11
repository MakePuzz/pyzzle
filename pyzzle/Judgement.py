from enum import Enum


class Judgement(Enum):
    """
    Enumeration of the possible placement of the word to be placed on the board.

    The result number corresponds to the judgment result
    0. The word can be placed (only succeeded)
    1. The preceding and succeeding cells are already filled
    2. At least one place must cross other words
    3. Not a correct intersection
    4. The same word is in use
    5. The Neighbor cells are filled except at the intersection
    6. US/USA, DOMINICA/DOMINICAN problem
    7. The word overlap with the mask
    """
    THE_WORD_CAN_BE_PLACED = 0
    THE_PRECEDING_AND_SUCCEEDING_CELLS_ARE_ALREADY_FILLED = 1
    AT_LEAST_ONE_PLACE_MUST_CROSS_OTHER_WORDS = 2
    NOT_A_CORRECT_INTERSECTION = 3
    THE_SAME_WORD_IS_IN_USE = 4
    THE_NEIGHBOR_CELLS_ARE_FILLED_EXCEPT_AT_THE_INTERSECTION = 5
    US_USA_DOMINICA_DOMINICAN_PROBLEM = 6
    THE_WORD_OVERLAP_WITH_THE_MASK = 7