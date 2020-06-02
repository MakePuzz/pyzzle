import copy


class Optimizer:
    def __init__(self):
        self.method_list = ["local_search", "iterated_local_search"]
        self.method = ""

    @staticmethod
    def get_neighbor_solution(puzzle):
        """
        This method gets the neighborhood solution
        """
        # Copy the puzzle
        _puzzle = copy.deepcopy(puzzle)
        # Drop words until connectivity collapse
        _puzzle.collapse()
        # Kick
        _puzzle.kick()
        # Add as much as possible
        _puzzle.add_to_limit()
        return _puzzle

    def local_search(self, puzzle, epoch, show=True, move=False):
        """
        This method performs a local search
        """
        # Logging
        if puzzle.epoch == 0:
            puzzle.logging()
        # Copy
        _puzzle = copy.deepcopy(puzzle)
        if show is True:
            print(">>> Interim solution")
            _puzzle.show()
        goal_epoch = _puzzle.epoch + epoch
        for ep in range(epoch):
            _puzzle.epoch += 1
            print(f">>> Epoch {_puzzle.epoch}/{goal_epoch}")
            # Get neighbor solution by drop->kick->add
            new_puzzle = self.get_neighbor_solution(_puzzle)

            # Repeat if the score is high
            for func_num in range(len(_puzzle.obj_func)):
                prev_score = _puzzle.obj_func.get_score(_puzzle, func_num)
                new_score = new_puzzle.obj_func.get_score(new_puzzle, func_num)
                if new_score > prev_score:
                    print(f"    - Improved: {_puzzle.obj_func.get_score(_puzzle, all=True)} --> {new_puzzle.obj_func.get_score(new_puzzle, all=True)}")
                    _puzzle = copy.deepcopy(new_puzzle)
                    _puzzle.logging()
                    if show is True:
                        _puzzle.show()
                    break
                if new_score < prev_score:
                    _puzzle.logging()
                    print(f"    - Stayed: {_puzzle.obj_func.get_score(_puzzle, all=True)}")
                    break
            else:
                _puzzle = copy.deepcopy(new_puzzle)
                _puzzle.logging()
                print(f"    - Replaced(same score): {_puzzle.obj_func.get_score(_puzzle, all=True)} -> {new_puzzle.obj_func.get_score(new_puzzle, all=True)}")
                if show is True:
                    _puzzle.show()
        # Update previous puzzle
        puzzle.weight = copy.deepcopy(_puzzle.weight)
        puzzle.enable = copy.deepcopy(_puzzle.enable)
        puzzle.cell = copy.deepcopy(_puzzle.cell)
        puzzle.cover = copy.deepcopy(_puzzle.cover)
        puzzle.label = copy.deepcopy(_puzzle.label)
        puzzle.used_words = copy.deepcopy(_puzzle.used_words)
        puzzle.used_plc_idx = copy.deepcopy(_puzzle.used_plc_idx)
        puzzle.nwords = copy.deepcopy(_puzzle.nwords)
        puzzle.history = copy.deepcopy(_puzzle.history)
        puzzle.base_history = copy.deepcopy(_puzzle.base_history)
        puzzle.log = copy.deepcopy(_puzzle.log)
        puzzle.epoch = copy.deepcopy(_puzzle.epoch)
        puzzle.first_solved = copy.deepcopy(_puzzle.first_solved)
        puzzle.seed = copy.deepcopy(_puzzle.seed)
        puzzle.dic = copy.deepcopy(_puzzle.dic)
        puzzle.plc = copy.deepcopy(_puzzle.plc)

    def set_method(self, method_name):
        """
        This method sets the optimization method on the instance
        """
        if method_name not in self.method_list:
            raise ValueError(f"Optimizer doesn't have '{method_name}' method")
        self.method = method_name
