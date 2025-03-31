import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Cube(gym.Env):
    """
    Rubik's Cube environment with standard face-based moves and a 3D render.
    Faces: U, D, F, B, L, R.
    Rotations: direction=+1 => 90° clockwise as viewed from that face.
    """

    def __init__(self, n=3):
        super().__init__()
        self.size = n
        self.state = {
            'U': np.full((n, n), 0),  # Up
            'D': np.full((n, n), 1),  # Down
            'F': np.full((n, n), 2),  # Front
            'B': np.full((n, n), 3),  # Back
            'L': np.full((n, n), 4),  # Left
            'R': np.full((n, n), 5),  # Right
        }

        # 3 axis × n layers × 2 directions
        self.action_space = spaces.Discrete(3 * n * 2)

        # Simple color map for each integer in the faces
        self.color_map = {
            0: 'white',   # U
            1: 'yellow',  # D
            2: 'green',   # F
            3: 'blue',    # B
            4: 'orange',  # L
            5: 'red',     # R
        }
    def get_state(self):
        flat = np.concatenate([self.state[face].flatten() for face in self.state])
        return flat

    # ------------------------
    #   FACE-BASED ROTATIONS
    # ------------------------
    def rotate_face(self, face, direction=1):
        """Rotate a face by ±90°. np.rot90 is CCW, so we pass -direction."""
        self.state[face] = np.rot90(self.state[face], -direction)

    def rotate_layer(self, axis, layer, direction=1):
        """
        Rotate a slice (layer) around an axis:
          x => L(0)/R(n-1)
          y => U(0)/D(n-1)
          z => F(0)/B(n-1)

        direction=+1 => 90° clockwise if you're facing that face from outside.
        """
        n = self.size

        if axis == 'z':
            # FRONT / BACK
            if layer == 0:
                self.rotate_face('F', direction)
            elif layer == n-1:
                self.rotate_face('B', -direction)

            # Cycle among U, L, D, R at row = layer
            top_row_u    = self.state['U'][layer, :].copy()
            left_col_l   = self.state['L'][:, n-1-layer].copy()
            bottom_row_d = self.state['D'][n-1-layer, :].copy()
            right_col_r  = self.state['R'][:, layer].copy()

            if direction == 1:  # F clockwise
                self.state['U'][layer, :]           = left_col_l[::-1]
                self.state['L'][:, n-1-layer]       = bottom_row_d
                self.state['D'][n-1-layer, :]       = right_col_r[::-1]
                self.state['R'][:, layer]           = top_row_u
            else:  # F counterclockwise
                self.state['U'][layer, :]           = right_col_r
                self.state['R'][:, layer]           = bottom_row_d[::-1]
                self.state['D'][n-1-layer, :]       = left_col_l
                self.state['L'][:, n-1-layer]       = top_row_u[::-1]

        elif axis == 'x':
            # LEFT / RIGHT
            if layer == 0:
                self.rotate_face('L', direction)
            elif layer == n-1:
                self.rotate_face('R', -direction)

            # Extract columns from each face
            colU = self.state['U'][:, layer].copy()
            colB = self.state['B'][:, n-1-layer].copy()
            colD = self.state['D'][:, layer].copy()
            colF = self.state['F'][:, layer].copy()

            if direction == 1:  # L clockwise
                self.state['U'][:, layer] = colB
                self.state['B'][:, n-1-layer] = colD#[::-1]
                self.state['D'][:, layer] = colF[::-1]
                self.state['F'][:, layer] = colU[::-1]
            else:  # L counterclockwise
                self.state['U'][:, layer] = colF[::-1]
                self.state['F'][:, layer] = colD[::-1]
                self.state['D'][:, layer] = colB#[::-1]
                self.state['B'][:, n-1-layer] = colU

        elif axis == 'y':
            # UP / DOWN
            if layer == 0:
                self.rotate_face('U', direction)
            elif layer == n-1:
                self.rotate_face('D', -direction)

            # Cycle among B, R, F, L at row = layer
            row_b = self.state['B'][layer, :].copy()
            row_r = self.state['R'][layer, :].copy()
            row_f = self.state['F'][layer, :].copy()
            row_l = self.state['L'][layer, :].copy()

            if direction == 1:  # U clockwise
                self.state['B'][layer, :] = row_r
                self.state['R'][layer, :] = row_f
                self.state['F'][layer, :] = row_l
                self.state['L'][layer, :] = row_b

            else:  # U counterclockwise
                self.state['B'][layer, :] = row_l
                self.state['L'][layer, :] = row_f
                self.state['F'][layer, :] = row_r
                self.state['R'][layer, :] = row_b

    # ----------------------
    #     GYM FUNCTIONS
    # ----------------------
    def step(self, action):
        block_size  = 2 * self.size  # each axis has 2*n combos
        axis_index  = action // block_size
        remainder   = action % block_size
        layer       = remainder // 2
        dir_bit     = remainder % 2
        direction   = 1 if dir_bit == 0 else -1

        axes = ['x', 'y', 'z']
        axis = axes[axis_index]

        self.rotate_layer(axis, layer, direction)
        return self.state, 0.0, False, {}

    def reset(self):
        self.__init__(self.size)
        return self.state
    
    def shuffle(self,n=100):
        for _ in range(n):
            action = np.random.randint(self.action_space.n)
            # print(action)
            self.step(action)
        return self.state

    def render(self):
        """Simple text-based face dump."""
        for face, mat in self.state.items():
            print(face + ':')
            print(mat)
            print()

    # --------------------------
    #   3D RENDERING FUNCTION
    # --------------------------
    def render_3d(self):
        """
        Shows each face in 3D. If one face seems 'upside-down', tweak its
        'origin', 'right_vec', or 'down_vec' below until it looks correct.
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_box_aspect((1,1,1))
        ax.set_axis_off()

        def add_square_3d(corners, facecolor):
            poly = Poly3DCollection([corners])
            poly.set_facecolor(facecolor)
            poly.set_edgecolor('black')
            ax.add_collection3d(poly)

        n = self.size
        half = n / 2.0

        # For each face, define an origin (top-left corner),
        # and row/column vectors (down_vec, right_vec).
        #
        # The definitions below should display all faces right-side up:
        face_configs = {
            'U': {
                'origin':    np.array([-half, +half, +half]),
                'right_vec': np.array([1.0, 0.0, 0.0]),
                'down_vec':  np.array([0.0, 0.0, -1.0])
            },
            'D': {
                'origin':    np.array([-half, -half, -half]),
                'right_vec': np.array([1.0, 0.0, 0.0]),
                'down_vec':  np.array([0.0, 0.0, +1.0])
            },
            'F': {
                'origin':    np.array([-half, +half, +half]),
                'right_vec': np.array([1.0, 0.0, 0.0]),
                'down_vec':  np.array([0.0, -1.0, 0.0])
            },
            'B': {
                # Adjusted to avoid appearing flipped:
                'origin':    np.array([+half, +half, -half]),
                'right_vec': np.array([-1.0, 0.0, 0.0]),
                'down_vec':  np.array([0.0, -1.0, 0.0])
            },
            'R': {
                'origin':    np.array([+half, +half, +half]),
                'right_vec': np.array([0.0, 0.0, -1.0]),
                'down_vec':  np.array([0.0, -1.0, 0.0])
            },
            'L': {
                # Adjusted so that the left face isn't flipped:
                'origin':    np.array([-half, +half, -half]),
                'right_vec': np.array([0.0, 0.0, +1.0]),
                'down_vec':  np.array([0.0, -1.0, 0.0])
            }
        }

        # Draw each face as n×n squares
        for face_name, face_mat in self.state.items():
            cfg = face_configs[face_name]
            origin = cfg['origin']
            rv = cfg['right_vec']
            dv = cfg['down_vec']

            for i in range(n):        # row index => 'down' steps
                for j in range(n):    # col index => 'right' steps
                    color_idx = face_mat[i, j]
                    color = self.color_map[color_idx]

                    # top-left corner of cell (i, j)
                    corner = origin + dv*i + rv*j
                    p0 = corner
                    p1 = corner + rv
                    p2 = corner + rv + dv
                    p3 = corner + dv

                    add_square_3d([p0, p1, p2, p3], color)

        # Adjust view
        max_range = n
        ax.set_xlim(-max_range, +max_range)
        ax.set_ylim(-max_range, +max_range)
        ax.set_zlim(-max_range, +max_range)

        plt.show()
