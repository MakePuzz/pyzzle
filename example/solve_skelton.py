#%%
import copy

import numpy as np

# from pyzzle.solver import SkeltonSolver


class BranchPoint(tuple):
    def __new__(cls, array):
        return tuple.__new__(cls, array)
    def __repr__(self):
        return f"({', '.join(map(str,self))})" 
    
    @property
    def ori(self):
        return self[0]
    @property
    def i(self):
        return self[1]
    @property
    def j(self):
        return self[2]
    @property
    def word(self):
        return self[3]


class Branch(list):
    def __init__(self, branch_point, maximum_length=None):
        super().__init__([branch_point])
        self.maximum_length = maximum_length
        self.completed = False
    
    def __repr__(self):
        return f"{self.__class__.__name__}(◉-{'-'.join(map(str,self))})"
    def split(self):
        cp = copy.deepcopy(self)
        return self, cp
    @property
    def ori(self):
        return list(map(lambda pt: pt.ori, self))
    @property
    def i(self):
        return list(map(lambda pt: pt.i, self))
    @property
    def j(self):
        return list(map(lambda pt: pt.j, self))
    @property
    def word(self):
        return list(map(lambda pt: pt.word, self))
    @property
    def depth(self):
        return len(self)
    @property
    def is_completed(self):
        return self.completed
    def get(self, idx):
        try:
            return self[idx]
        except IndexError:
            return BranchPoint(None, None, None, None)

class SolutionTree(list):
    def __init__(self, branch):
        super().__init__([branch])
    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}({(os.linesep+' '*len(name)+' ').join(map(str,self))})"
    def crop(self, depth):
        return list(map(lambda branch: branch.get(depth), self))
    @property
    def is_completed(self):
        for branch in self:
            if not branch.is_completed:
                return False
        return True

# %%
ptA1 = BranchPoint([0, 1, 2, "hoge"])
ptA2 = BranchPoint([1, 2, 3, "fuga"])
branchA = Branch(ptA1)
branchA.append(ptA2)

ptB1 = BranchPoint([0, 1, 2, "hoge"])
ptB2 = BranchPoint([1, 2, 4, "piyo"])
ptB3 = BranchPoint([0, 3, 1, "foo"])
branchB = Branch(ptB1)
branchB.append(ptB2)
branchB.append(ptB3)

tree = SolutionTree(branchA)
tree.append(branchB)
tree

#%%
import scipy.ndimage as ndi
class SkeltonSolver:
    def __init__(self, cover, words):
        """
        Parameters
        ----------
        cover : array_like
        
        words : array_like
        """
        self.trees = []
        self.cover = np.array(cover)
        self.words = np.array(words)
        self.nwords = len(self.words)
        self.wlens = np.array(list(map(len, self.words)))
        self.vertical_cover = ndi.binary_opening(self.cover, structure=[[1],[1]]).astype(int)
        self.vertical_lbl, _ = ndi.label(self.vertical_cover, structure=[[0,1,0],[0,1,0],[0,1,0]])
        self.horizontal_cover = ndi.binary_opening(self.cover, structure=[[1, 1]]).astype(int)
        self.horizontal_lbl, _ = ndi.label(self.horizontal_cover, structure=[[0,0,0],[1,1,1],[0,0,0]])

    def solve(self):
        """
        Returns
        -------
        cells : numpy ndarray
            Array containing the solutions.
        """
        print("========================================================")
        start_words = self.get_starting_words()
        comps = Puzzle.get_word_compositions(puzzle.cover)
        comp = comps[len(start_words[0])][0]
        for word in start_words:
            pt = BranchPoint([comp[0], comp[1], comp[2], word])
            self.trees.append(SolutionTree(Branch(pt)))
        
        for t, tree in enumerate(self.trees):
            print(f"--------------------")
            print(f"      Tree {t}      ")
            print(f"--------------------")
            tree = self.grow_tree(tree)
        print("========================================================")

        solved_branches = []
        for tree in self.trees:
            for branch in tree:
                if branch.depth == self.nwords:
                    solved_branches.append(branch)
        solved_branches = self.get_unique_list(solved_branches)
        return solved_branches, self.trees
    
    def get_unique_list(self, seq):
        seen = []
        return [x for x in seq if x not in seen and not seen.append(x)]
        
    def grow_tree(self, tree):
        if tree.is_completed:
            print("Growed")
            return tree
        for branch in tree:
            # print(f"|==== Grow branch (completed: {branch.is_completed})")
            if branch.is_completed:
                continue
            if branch.depth == self.nwords:
                branch.completed = True
                break
            tmp_words = copy.deepcopy(self.words)
            for uword in branch.word:
                tmp_words = np.delete(tmp_words, np.where(tmp_words == uword)[0])
            for i in range(self.nwords-branch.depth):
                cell = self.construct_cell_from_branch(branch)
                tmp_wlens = np.array(list(map(len, tmp_words)))

                # get an edge
                edges = self.find_edges(cell, self.cover)
                if len(edges) == 0:
                    branch.completed = True
                    # print("Break. エッジが存在しない")
                    break
                edge = edges[0]

                # get useable_words
                useable_words = []
                for potential_word in tmp_words[tmp_wlens == edge["len"]]:
                    use_this = True
                    for ind, char in edge["cross"].items():
                        if char != "" and potential_word[ind] != char:
                            use_this = False
                            break
                    if use_this:
                        useable_words.append(potential_word)
                if useable_words == []:
                    branch.completed = True
                    # print("Break. エッジはあるがマッチするワードが存在しない")
                    break
                org_branch = copy.deepcopy(branch)
                for k, useable_word in enumerate(useable_words):
                    pt = BranchPoint([edge["ori"], edge["i"], edge["j"], useable_word])
                    if k == 0:
                        tmp_words = np.delete(tmp_words, np.where(tmp_words == useable_word)[0])
                        branch.append(pt)
                    else:
                        new_branch = copy.deepcopy(org_branch)
                        new_branch.append(pt)
                        tree.append(new_branch)
        tree = self.grow_tree(tree)
        return tree
        
    def find_edges(self, cell, cover):
        """
        Returns a set of indices, orientations, lengths, and cross infomations 
        of corossing words.

        Parameters
        ----------
        cell : numpy ndarray
            Cell array.
        cover : numpy ndarray
            Cover array.
        
        Returns
        -------
        edges : list
            Edges.
        """
        crosses = np.where((cell != "") * (cover == 2))
        cell = np.pad(cell, 1, mode="constant", constant_values="P")
        cover = np.pad(cover, 1, mode="constant", constant_values=0)
        edges = []
        for ci, cj in zip(crosses[0], crosses[1]):
            ci += 1 # +1 because padding
            cj += 1 # +1 because padding
            composition = {"ori": None, "i": None, "j": None, "len": None, "cross":{}}
            upper = (cell[ci-1, cj] == "") * (cover[ci-1, cj] >= 1)
            under = (cell[ci+1, cj] == "") * (cover[ci+1, cj] >= 1)
            connect_vertically = upper + under
            left = (cell[ci, cj-1] == "") * (cover[ci, cj-1] >= 1)
            right = (cell[ci, cj+1] == "") * (cover[ci, cj+1] >= 1)
            connect_horizontally = left + right
            if not connect_vertically and not connect_horizontally:
                continue
            if connect_vertically:
                composition["ori"] = 0
                word_indices = np.where(self.vertical_lbl == self.vertical_lbl[ci-1, cj-1])
            if connect_horizontally:
                composition["ori"] = 1
                word_indices = np.where(self.horizontal_lbl == self.horizontal_lbl[ci-1, cj-1])
            composition["i"] = word_indices[0][0]
            composition["j"] = word_indices[1][0]
            composition["len"] = word_indices[0].size
            is_cross = (cover[word_indices[0]+1, word_indices[1]+1] == 2)
            cross_char = cell[word_indices[0]+1, word_indices[1]+1][is_cross]
            for ind, char in zip(np.where(is_cross)[0], cross_char):
                composition["cross"][ind] = char
            edges.append(composition)
        return edges
    
    def construct_cell_from_branch(self, branch):
        cell = np.full(self.cover.shape, "")
        for pt in branch:
            if pt.ori == 0:
                cell[pt.i:pt.i+len(pt.word), pt.j] = list(pt.word)
            if pt.ori == 1:
                cell[pt.i, pt.j:pt.j+len(pt.word)] = list(pt.word)
        return cell

    def get_starting_words(self):
        """
        Returns
        -------
        words : numpy ndarray
            Array containing the words with the fewest words of the same length.
        """
        import collections
        c = collections.Counter(self.wlens)
        only1_length = list(c.keys())[np.argmin(list(c.values()))]
        words = self.words[(np.array(self.wlens) == only1_length)]
        return words


#%%
from pyzzle import Puzzle
puzzle = Puzzle.from_json("json/v005sample.json")

#%%
solver = SkeltonSolver(puzzle.cover, puzzle.uwords[:puzzle.nwords])

branches, trees = solver.solve()
branches

# %%
cell = solver.construct_cell_from_branch(branches[0])
cell
# %%
solved_cell = Puzzle.from_cell(cell)
solved_cell
# %%
puzzle
# %%
trees
# %%
np.all(solved_cell.cell == puzzle.cell)