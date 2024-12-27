import math
import torch
import numpy as np


from escher.OTE.ote import AffineTrans, MapType, SparseSystem, Constraints


class OrbifoldIIConstraints(Constraints):
    def __init__(self, vertices, sides):
        self.sides = sides
        topleft, topright = sides["top"][0], sides["top"][-1]
        bottomleft, bottomright = sides["bottom"][-1], sides["bottom"][0]
        corners = [topleft, topright, bottomleft, bottomright]
        corner_coords = [[0, 1], [math.sqrt(3), 0], [-math.sqrt(3), 0], [0, -1]]

        left, right = sides["left"][1:-1], sides["right"][1:-1]
        top, bottom = np.flip(sides["top"])[1:-1], np.flip(sides["bottom"])[1:-1]

        sp = SparseSystem(vertices.shape[0])

        sp.generate_rotation_constraints(
            left, top, 2 * math.pi / 3, np.array([0.0, 1.0])
        )
        sp.generate_rotation_constraints(
            right, bottom, 2 * math.pi / 3, np.array([0.0, -1.0])
        )

        for corner, coord in zip(corners, corner_coords):
            sp.generate_fixed_constraints(np.array([corner]), np.array([coord]))

        super().__init__(sp)
        self.update_scaling()

    def update_scaling(self):
        self.ad_hoc_scaling = np.sqrt(2 / math.sqrt(3))

    def tiling_coloring_number(self):
        return 3

    def get_global_transformation_type(self):
        return MapType.IDENTITY

    def get_torus_directions(self):
        return np.array([2 * math.sqrt(3), 0]), np.array(
            [math.sqrt(3), -self.tile_width / 2 - 2]
        )

    def get_torus_cover(self, vertices, sides):
        theta = 2 * math.pi / 3
        R120 = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        )
        R120 = AffineTrans(R120, np.array([0, 0]))

        first = AffineTrans(np.eye(2), np.array([0, -self.tile_width / 2]), 0, (0, 0))

        A = [first]
        for i in range(2):
            A.append(R120.compose(A[-1], i + 1, (0, 0)))

        return A

    def get_boundary(self):
        sides = self.sides
        return [(sides["top"], sides["left"]), (sides["bottom"], sides["right"])]

    def tiling_coloring_number(self):
        return 8

    def get_horizontal_symmetry_orientation(self):
        theta = np.pi / 4
        return torch.Tensor(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
