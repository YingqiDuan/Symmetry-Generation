import torch
import numpy as np


from escher.OTE.ote import AffineTrans, MapType, SparseSystem, Constraints


class Reflect333Constraints(Constraints):
    def __init__(self, vertices, sides):
        self.sides = sides
        topright = sides["top"][-1]
        topleft = sides["top"][0]
        bottomleft = sides["bottom"][-1]
        bottomright = sides["bottom"][0]
        corners = [topleft, topright, bottomleft, bottomright]
        corner_coords = [
            [-np.sqrt(3) / 4, -1 / 4],
            [0, 0],
            [-np.sqrt(3) / 2, -1 / 2],
            [0, -1],
        ]

        bottom = sides["bottom"]
        top = sides["top"]
        left = sides["left"]
        right = sides["right"]

        sp = SparseSystem(vertices.shape[0])
        n_1 = np.array([1, 0])
        n_2 = np.array([1 / 2, -np.sqrt(3) / 2])
        n_3 = np.array([-1 / 2, -np.sqrt(3) / 2])

        n_1 = n_1 / np.linalg.norm(n_1)
        n_2 = n_2 / np.linalg.norm(n_2)
        n_3 = n_3 / np.linalg.norm(n_3)

        lines = [bottom[:-1], right[:-1], top[:-1], left[:-1]]
        cons = [n_3, n_1, n_2, n_2]
        for line, con in zip(lines, cons):
            sp.generate_straight_line_constraint(line, con)

        for corner, coord in zip(corners, corner_coords):
            sp.generate_fixed_constraints(np.array([corner]), np.array([coord]))

        super().__init__(sp)

    def update_scaling(self):
        self.ad_hoc_scaling = np.sqrt(4 / (np.sqrt(3) / 4))

    def get_global_transformation_type(self):
        return MapType.IDENTITY

    def get_torus_directions(self):
        return np.array([np.sqrt(3), 0]), np.array([np.sqrt(3) / 2, 3 / 2])

    def get_torus_cover(self, vertices, sides):
        I = np.identity(2)
        A = [AffineTrans(I, np.array([0, 0]), 0)]

        reflect1 = np.array([[1 / 2, np.sqrt(3) / 2], [np.sqrt(3) / 2, -1 / 2]])
        A.append(AffineTrans(reflect1, np.array([0, 0]), 1))

        reflect2 = np.array([[1 / 2, -np.sqrt(3) / 2], [-np.sqrt(3) / 2, -1 / 2]])
        A.append(AffineTrans(reflect2, np.array([0, 0]), 1))

        Reflect_y = np.array([[-1, 0], [0, 1]])
        Reflect_y = AffineTrans(Reflect_y, np.array([0, 0]), 2)
        B = list()
        i = 3
        for transfo_1 in A:
            B.append(Reflect_y.compose(transfo_1, i, (0, 0)))
            i += 1
        A = A + B
        return A

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
