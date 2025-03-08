from cube import Cube
import numpy as np

def test_cube():
    size = 3
    env = Cube(size)
    obs = env.reset()
    env.render_3d()
    # for a cube of size N, there are 6*N possible moves
    # for i in range(6*size):
    #     obs,_,_,_ = env.step(i)
    #     env.render_3d()
    state = env.shuffle(100)
    env.render_3d()


test_cube()