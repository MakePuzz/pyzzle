from abc import ABCMeta, abstractmethod
import copy
import logging
import time

LOG = logging.getLogger(__name__)


class Optimizer(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def optimize(self):
        pass


class LocalSearch(Optimizer):
    def __init__(self, show=True, shrink=False, move=False, use_f=False):
        self.show = show
        self.shrink = shrink
        self.move = move
        self.use_f = use_f
    
    def optimize(self, puzzle, epoch, time_limit=None, time_offset=0):
        """
        This method performs a local search
        """
        # Logging
        if puzzle.epoch == 0:
            puzzle.logging()
        # Copy
        _puzzle = puzzle.copy(deep=True)
        if self.show:
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
            new_puzzle = self.get_neighbor_solution(_puzzle)

            # Repeat if the score is high
            for func_num in range(len(_puzzle.obj_func)):
                prev_score = _puzzle.obj_func.get_score(_puzzle, func_num)
                new_score = new_puzzle.obj_func.get_score(new_puzzle, func_num)
                if new_score > prev_score:
                    LOG.info(f"- Improved: {_puzzle.obj_func.get_score(_puzzle, all=True)}")
                    LOG.info(f"        --> {new_puzzle.obj_func.get_score(new_puzzle, all=True)}")
                    _puzzle = new_puzzle.copy(deep=True)
                    _puzzle.logging()
                    if self.show:
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
                if self.show:
                    _puzzle.show()
            if self.shrink:
                _puzzle = _puzzle.shrink()
        return _puzzle

    def get_neighbor_solution(self, puzzle):
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
        if self.use_f:
            _puzzle.add_to_limit_f()
        else:
            _puzzle.add_to_limit()
        return _puzzle


class MultiStart(Optimizer):
    def __init__(self, n, show=True, shrink=False, move=False, use_f=False):
        self.n = n
        self.show = show
        self.shrink = shrink
        self.move = move
        self.use_f = use_f
        self.localsearch_optimizer = LocalSearch(show=show, shrink=shrink, move=move, use_f=use_f)

    def optimize(self, puzzle, epoch, time_limit=None, time_offset=0):
        puzzles = []
        if time_limit is not None:
            start_time = time.time()
        for _n in range(self.n):
            if time_limit is not None:
                performance_time = time.time() - start_time
                time_offset = performance_time + time_offset
                if performance_time >= time_limit:
                    break
            LOG.info(f"> Node: {_n+1}")
            _puzzle = puzzle.copy(deep=True)
            _puzzle = self.localsearch_optimizer.optimize(_puzzle, epoch, time_limit=time_limit, time_offset=time_offset)
            puzzles.append(_puzzle)
        return self.get_prime_puzzle(puzzles)

    def get_prime_puzzle(self, puzzles):
        for i, _puzzle in enumerate(puzzles):
            if i == 0:
                prime_puzzle = _puzzle
                continue
            if _puzzle >= prime_puzzle:
                prime_puzzle = _puzzle
        return prime_puzzle
