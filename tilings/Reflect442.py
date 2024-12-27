import torch
import numpy as np


from escher.OTE.ote import AffineTrans, MapType, SparseSystem, Constraints

from escher.OTE.tilings.ReflectSquare import ReflectSquareConstraints


class Reflect442Constraints(Constraints):
    """
    Orbifold signature "*442"
    """

    def __init__(self, vertices, sides):
        self.sides = sides
        topright = sides["top"][-1]
        topleft = sides["top"][0]
        bottomleft = sides["bottom"][-1]
        bottomright = sides["bottom"][0]
        corners = [topleft, topright, bottomleft, bottomright]
        corner_coords = [
            [-0.5, -0.5],
            [0, 0],
            [-1, -1],
            [0, -1],
        ]

        bottom = sides["bottom"]
        top = sides["top"]
        left = sides["left"]
        right = sides["right"]

        sp = SparseSystem(vertices.shape[0])
        n_up = np.array([0, 1])
        n_right = np.array([1, 0])
        lines = [bottom[:-1], right[:-1], top[:-1], left[:-1]]
        cons = [n_up, n_right, (n_up - n_right) / 2, (n_up - n_right) / 2]

        for line, con in zip(lines, cons):
            sp.generate_straight_line_constraint(line, con)

        for corner, coord in zip(corners, corner_coords):
            sp.generate_fixed_constraints(np.array([corner]), np.array([coord]))

        self.reflect_square_constraints = ReflectSquareConstraints(vertices, sides)
        super().__init__(sp)
        self.update_scaling()

    def update_scaling(self):
        self.ad_hoc_scaling = np.sqrt(8) / 2

    def get_global_transformation_type(self):
        return MapType.IDENTITY

    def get_torus_directions(self):
        return np.array([0, 4]), np.array([4, 0])

    def get_torus_cover(self, vertices, sides):
        I = np.identity(2)
        Reflect_xy = np.array([[0, 1], [1, 0]])
        first = AffineTrans(I, np.array([0, 0]), 0)
        second = AffineTrans(Reflect_xy, np.array([0, 0]), 1)
        A = [first, second]

        self.reflect_square_constraints = ReflectSquareConstraints(vertices, sides)
        A_reflect = self.reflect_square_constraints.get_torus_cover(vertices, sides)

        B = list()
        i = 0
        scale_and_translate = AffineTrans(2 * np.eye(2), np.array([1, 1]), 0, (0, 0))
        for transfo_1 in A:
            for transfo_2 in A_reflect:
                B.append(
                    transfo_2.compose(
                        scale_and_translate.compose(transfo_1, 0, (0, 0)), i, (0, 0)
                    )
                )
                i += 1
        return B

    def get_boundary(self):
        sides = self.sides
        return [
            (sides["top"], sides["bottom"][-1::-1]),
            (sides["left"], sides["right"][-1::-1]),
        ]

    def tiling_coloring_number(self):
        return 8

    def get_horizontal_symmetry_orientation(self):
        return torch.Tensor(np.eye(2))
