import copy, logging

LOG = logging.getLogger(__name__)


class Optimizer:
    method_list = ["local_search", "multi_start"]

    def __init__(self, method="local_search"):
        self.methods = {
            "local_search": self.local_search,
            "multi_start": self.multi_start
        }
        self.set_method(method)

    @staticmethod
    def get_neighbor_solution(puzzle, use_f=False):
        """
        This method gets the neighborhood solution
        """
        # Copy the puzzle
        _puzzle = copy.deepcopy(puzzle)
        if _puzzle.nwords >= 1:
            # Drop words until connectivity collapse
            _puzzle.collapse()
            # Kick
            _puzzle.kick()
        # Add as much as possible
        if use_f is True:
            _puzzle.add_to_limit_f()
        else:
            _puzzle.add_to_limit()
        return _puzzle

    @classmethod
    def local_search(self, puzzle, epoch, show=True, move=False, use_f=False):
        """
        This method performs a local search
        """
        # Logging
        if puzzle.epoch == 0:
            puzzle.logging()
        # Copy
        _puzzle = copy.deepcopy(puzzle)
        if show is True:
            LOG.info(">>> Interim solution")
            _puzzle.show()
        goal_epoch = _puzzle.epoch + epoch
        for _ in range(epoch):
            _puzzle.epoch += 1
            LOG.info(f">>> Epoch {_puzzle.epoch}/{goal_epoch}")
            # Get neighbor solution by drop->kick->add
            new_puzzle = self.get_neighbor_solution(_puzzle, use_f=use_f)

            # Repeat if the score is high
            for func_num in range(len(_puzzle.obj_func)):
                prev_score = _puzzle.obj_func.get_score(_puzzle, func_num)
                new_score = new_puzzle.obj_func.get_score(new_puzzle, func_num)
                if new_score > prev_score:
                    LOG.info(f"- Improved: {_puzzle.obj_func.get_score(_puzzle, all=True)}")
                    LOG.info(f"        --> {new_puzzle.obj_func.get_score(new_puzzle, all=True)}")
                    _puzzle = copy.deepcopy(new_puzzle)
                    _puzzle.logging()
                    if show is True:
                        _puzzle.show()
                    break
                if new_score < prev_score:
                    _puzzle.logging()
                    LOG.info(f"- Stayed: {_puzzle.obj_func.get_score(_puzzle, all=True)}")
                    break
            else:
                _puzzle = copy.deepcopy(new_puzzle)
                _puzzle.logging()
                LOG.info(f"- Replaced: {_puzzle.obj_func.get_score(_puzzle, all=True)}")
                if show is True:
                    _puzzle.show()
        return _puzzle

    def multi_start(self, puzzle, epoch, n=1, unique=False, show=True, use_f=False):
        puzzles = []
        for _n in range(n):
            LOG.info(f"> Node: {_n+1}")
            _puzzle = copy.deepcopy(puzzle)
            _puzzle = _puzzle.solve(epoch=epoch, optimizer="local_search", show=show, use_f=use_f)
            puzzles.append(_puzzle)
        for i, _puzzle in enumerate(puzzles):
            if i == 0:
                prime_puzzle = _puzzle
            else:
                if unique is True and _puzzle.is_unique is False:
                    continue
                if _puzzle >= prime_puzzle:
                    prime_puzzle = _puzzle
        return prime_puzzle

    def set_method(self, method_name):
        """
        This method sets the optimization method on the instance
        """
        if method_name not in self.method_list:
            raise ValueError(f"Optimizer doesn't have '{method_name}' method")
        self.method = method_name
        self.optimize = self.methods[method_name]
