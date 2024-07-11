import numpy as np


class Gravity(np.ndarray):
    hokkaido = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,0,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,0,0,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,0,0,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,0,0,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,0,0,0,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,0,0,0,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,0,0,0,0,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,0,0,0,0,0,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,0,0,0,0,0,0,0,0,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,8,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,0,0,0,0,0,0,0,0,0,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,8,8,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,1,1,1,1,1,8,0,8,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,1,1,1,8,0,8,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,1,8,8],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,1],
        [1,1,1,1,1,1,8,8,1,1,1,1,1,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,1,1],
        [1,1,1,1,1,1,8,0,8,1,1,1,1,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,1,1,1,1],
        [1,1,1,1,1,1,8,0,0,8,8,8,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,1,1,1,1,1],
        [1,1,1,1,1,1,1,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,8,8,8,8,8,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,8,8,0,0,0,8,8,8,8,0,0,0,8,1,1,1,8,0,0,0,0,0,0,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,8,0,0,0,8,1,1,1,8,0,0,8,1,1,1,1,1,8,0,0,0,0,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,8,0,0,8,1,1,1,1,1,8,8,1,1,1,1,1,1,1,1,8,0,0,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,8,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,8,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,8,0,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,8,0,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,8,0,0,0,0,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,8,0,0,8,1,8,8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,8,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,8,0,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,8,0,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    ])
    
    
    
    hokkaido_2 = np.array([
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,90,90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,85,85,85,85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,70,70,70,70,85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,85,70,50,70,85,85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,70,50,70,70,85,85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,70,70,50,70,70,85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,85,70,50,50,70,85,85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,70,50,50,50,70,85,85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,70,50,40,50,70,70,85,85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,70,50,40,40,50,70,70,85,85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,85,70,50,40,40,50,50,70,70,70,85,85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,90, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,70,70,50,40,40,50,50,50,50,70,70,85,85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,90,90, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,70,50,50,40,40,40,40,40,50,50,70,70,70,85,85,85,85,85, 0, 0, 0, 0, 0,90,90,90, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,70,50,40,40,40,40,40,40,40,50,50,50,70,70,70,70,70,85,85, 0, 0, 0,85,85,85, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,70,50,40,40,40,40,40,40,40,40,40,50,50,50,50,50,70,70,70,70,70,70,70,70,85, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,70,50,30,30,30,30,30,30,30,30,30,30,30,30,30,50,50,50,50,50,50,50,50,70,85, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,70,70,50,30,30,30,30,30,30,30,30,30,30,30,30,30,30,40,40,40,40,40,40,50,70,85, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,70,70,50,50,30,30,30,30,30,30,30,30,30,30,30,30,30,30,40,40,40,40,40,40,50,70,85,85, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,70,50,50,30,30,30,10,10,10,10,10,10,10,30,30,30,30,30,40,40,40,40,40,40,50,70,85,85, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,70,50,30,30,30,30,10,10,10,10,10,10,10,30,30,30,30,30,40,40,40,40,40,40,50,70,85,90,90, 0,90,90],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,70,50,30,30,30,30,10,10,10, 5, 5,10,10,30,30,30,30,30,40,40,40,40,40,40,50,70,85,90,90,90,90, 0],
        [ 0, 0, 0, 0, 0, 0,90,90, 0, 0, 0, 0, 0,85,70,50,30,30,30,30,10,10,10, 5, 5,10,10,30,30,30,30,30,40,40,40,40,50,50,50,70,85,90,90,90, 0, 0],
        [ 0, 0, 0, 0, 0, 0,90,90,90, 0, 0, 0, 0,85,70,50,30,30,30,30,10,10,10,10,10,10,10,30,30,30,50,50,50,50,50,50,50,70,70,70,85,90, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0,90,90,90,85,85,85,85,85,70,50,30,30,30,30,10,10,10,10,10,10,30,30,30,50,50,70,70,70,70,70,70,70,85,85,85, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0,85,70,70,70,70,70,70,70,50,30,30,30,30,10,10,10,10,10,10,30,50,50,50,70,70,85,85,85,85,85,85,85, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0,85,70,50,50,50,50,50,50,50,30,30,30,30,30,30,30,30,30,30,30,50,50,70,70,85,85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0,85,85,70,50,50,50,50,50,50,50,50,50,50,50,50,40,40,40,40,40,50,50,70,70,85,85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0,85,85,85,70,70,50,50,50,70,70,70,70,70,70,70,50,50,40,40,40,50,50,70,70,85,85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0,85,85,85,85,85,70,70,70,70,70,85,85,85,85,85,70,70,50,50,50,40,50,70,70,85,85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0,85,85,85,85,85,85,85,85,85,85,70,85,85, 0, 0, 0,85,85,70,70,70,50,50,50,70,85,85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0,85,85,85,85,85, 0, 0, 0,85,85,85,85, 0, 0, 0, 0, 0,85,85,85,70,70,70,50,70,85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0,85,85,85,85, 0, 0, 0, 0, 0,85,85, 0, 0, 0, 0, 0, 0, 0, 0,85,85,85,70,70,70,85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0,90,90,90,90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,85,70,70,85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0,90,90,90,90,90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,85,85,90,85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0,90,90,90,90,90,90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,90,90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0,90,90,90,90,90,90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0,90,90,90,90,90,90,90,90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0,90,90,90,90, 0,90,90,90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0,90,90,90,90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0,90,90,90,90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0,90,90,90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0,90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])