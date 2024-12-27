import math
import torch
import numpy as np


from escher.OTE.ote import AffineTrans, MapType, SparseSystem, Constraints


class OrbifoldIConstraints(Constraints):
    def __init__(self, vertices, sides):
        self.sides = sides
        topleft, topright = sides["top"][0], sides["top"][-1]
        bottomleft, bottomright = sides["bottom"][-1], sides["bottom"][0]
        corners = [topleft, topright, bottomleft, bottomright]
        corner_coords = [[-1, 1], [1, 1], [-1, -1], [1, -1]]

        left, right = sides["left"][1:-1], sides["right"][1:-1]
        top, bottom = np.flip(sides["top"])[1:-1], np.flip(sides["bottom"])[1:-1]

        sp = SparseSystem(vertices.shape[0])

        sp.generate_rotation_constraints(
            left, top, math.pi / 2, np.array(corner_coords[0])
        )
        sp.generate_rotation_constraints(
            right, bottom, math.pi / 2, np.array(corner_coords[3])
        )

        for corner, coord in zip(corners, corner_coords):
            sp.generate_fixed_constraints(np.array([corner]), np.array([coord]))

        super().__init__(sp)

    def update_scaling(self):
        self.ad_hoc_scaling = 1

    def get_global_transformation_type(self):
        return MapType.IDENTITY

    def get_torus_directions(self):
        return np.array([0, 4]), np.array([4, 0])

    def get_torus_cover(self, vertices, sides):
        first = AffineTrans(
            np.eye(2), np.array([self.tile_width / 2, -self.tile_width / 2]), 0, (0, 0)
        )
        R90 = AffineTrans(np.array([[0, 1], [-1, 0]]), np.array([0, 0]))
        A = [first]
        for i in range(3):
            A.append(R90.compose(A[-1], i + 1, (0, 0)))
        return A

    def get_boundary(self):
        sides = self.sides
        return [
            (sides["top"], sides["left"][-1::-1]),
            (sides["bottom"], sides["right"][-1::-1]),
        ]

    def tiling_coloring_number(self):
        return 8

    def get_horizontal_symmetry_orientation(self):
        theta = np.pi / 4
        return torch.Tensor(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
