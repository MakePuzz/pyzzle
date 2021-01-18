import copy
import logging
import time

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
        _puzzle = puzzle.copy(deep=True)
        if _puzzle.nwords >= 1:
            # Drop words until connectivity collapse
            _puzzle.collapse()
            # Kick
            _puzzle.kick()
        # Add as much as possible
        if use_f:
            _puzzle.add_to_limit_f()
        else:
            _puzzle.add_to_limit()
        return _puzzle

    def local_search(self, puzzle, epoch, time_limit=None, time_offset=0, show=True, shrink=False, move=False, use_f=False):
        """
        This method performs a local search
        """
        # Logging
        if puzzle.epoch == 0:
            puzzle.logging()
        # Copy
        _puzzle = puzzle.copy(deep=True)
        if show:
            LOG.info(">>> Interim solution")
            _puzzle.show()
        goal_epoch = _puzzle.epoch + epoch

        if time_limit is not None:
            start_time = time.time()

        for ep in range(epoch):
            if time_limit is not None:
                performance_time = time.time() - start_time
                duration_per_1ep = performance_time/(ep+1)
                if performance_time + duration_per_1ep + time_offset >= time_limit:
                    LOG.info(f"End with time limit. {performance_time + time_offset} sec (> {time_limit} sec)")
                    break
                LOG.info(f"{round(performance_time + time_offset,3)} sec (< {round(time_limit,3)} sec)")
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
                    _puzzle = new_puzzle.copy(deep=True)
                    _puzzle.logging()
                    if show:
                        _puzzle.show()
                    break
                if new_score < prev_score:
                    _puzzle.logging()
                    LOG.info(f"- Stayed: {_puzzle.obj_func.get_score(_puzzle, all=True)}")
                    break
            else:
                _puzzle = new_puzzle.copy(deep=True)
                _puzzle.logging()
                LOG.info(f"- Replaced: {_puzzle.obj_func.get_score(_puzzle, all=True)}")
                if show:
                    _puzzle.show()
            if shrink:
                _puzzle = _puzzle.shrink()
        return _puzzle

    def multi_start(self, puzzle, epoch, time_limit=None, time_offset=0, n=1, unique=False, show=True, shrink=False, use_f=False):
        puzzles = []
        if time_limit is not None:
            start_time = time.time()
        for _n in range(n):
            if time_limit is not None:
                performance_time = time.time() - start_time
                time_offset = performance_time + time_offset
                if performance_time >= time_limit:
                    break
            LOG.info(f"> Node: {_n+1}")
            _puzzle = puzzle.copy(deep=True)
            _puzzle = _puzzle.solve(epoch=epoch, optimizer="local_search", time_limit=time_limit, of=_puzzle.obj_func, time_offset=time_offset, show=show, shrink=shrink, use_f=use_f)
            puzzles.append(_puzzle)
        for i, _puzzle in enumerate(puzzles):
            if i == 0:
                prime_puzzle = _puzzle
            else:
                if unique and not _puzzle.is_unique:
                    continue
                if _puzzle >= prime_puzzle:
                    prime_puzzle = _puzzle
        return prime_puzzle

    def set_method(self, method_name="local_search"):
        """
        This method sets the optimization method on the instance
        """
        if method_name not in self.method_list:
            raise ValueError(f"Optimizer doesn't have '{method_name}' method")
        self.method = method_name
        self.optimize = self.methods[method_name]
