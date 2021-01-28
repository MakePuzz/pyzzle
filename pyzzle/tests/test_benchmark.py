import os
import unittest

import pytest

from pyzzle import Puzzle, Dictionary, Mask, Gravity
from pyzzle import utils


def init(width=15, height=15, dictionary="pokemon"):
    dic = Dictionary(f"{os.path.dirname(__file__)}/data/{dictionary}.txt")
    seed = 0
    puzzle = Puzzle(width=width, height=height, seed=seed)
    puzzle.import_dict(dic)
    return puzzle

def solve_local_search(width=15, height=15, dictionary="pokemon", epoch=30, obj_func=None, use_f=False):
    puzzle = init(width, height, dictionary)
    if obj_func is None:
        obj_func = ["uniqueness", "weight", "nwords", "area_rect"]
    puzzle = puzzle.solve(epoch=epoch, time_limit=None, optimizer="local_search", of=obj_func, show=False, use_f=use_f)        
    return puzzle

def test_init_15x15_pokemon_benchmark(benchmark):
    puzzle = benchmark.pedantic(init, kwargs=dict(width=15, height=15, dictionary="pokemon"),
                             rounds=5, iterations=1)
    assert True

def test_init_30x30_pokemon_benchmark(benchmark):
    puzzle = benchmark.pedantic(init, kwargs=dict(width=30, height=30, dictionary="pokemon"),
                             rounds=5, iterations=1)
    assert True

def test_solve_local_search_5x5_pokemon_ep10_benchmark(benchmark):
    puzzle = benchmark.pedantic(solve_local_search, kwargs=dict(width=5, height=5, dictionary="pokemon", epoch=10),
                             rounds=2, iterations=1)
    assert True

def test_solve_local_search_10x10_pokemon_ep3_benchmark(benchmark):
    puzzle = benchmark.pedantic(solve_local_search, kwargs=dict(width=10, height=10, dictionary="pokemon", epoch=3),
                             rounds=2, iterations=1)
    assert True