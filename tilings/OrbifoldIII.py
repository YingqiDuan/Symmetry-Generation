import math
import torch
import numpy as np


from escher.OTE.ote import AffineTrans, MapType, SparseSystem, Constraints


class OrbifoldIIIConstraints(Constraints):
    def __init__(self, vertices, sides):
        self.sides = sides

        topleft, topright = sides["top"][0], sides["top"][-1]
        bottomleft, bottomright = sides["bottom"][-1], sides["bottom"][0]
        corners = [topleft, topright, bottomleft, bottomright]
        corner_coords = [[0.0, math.sqrt(3)], [1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]]

        left, right = sides["left"][1:-1], sides["right"][1:-1]
        top, bottom = np.flip(sides["top"])[1:-1], np.flip(sides["bottom"])[1:-1]

        sp = SparseSystem(vertices.shape[0])

        sp.generate_rotation_constraints(
            left, top, 2 * math.pi / 6, np.array([0.0, math.sqrt(3)])
        )
        sp.generate_rotation_constraints(right, bottom, math.pi, np.array([0.0, 0.0]))

        for corner, coord in zip(corners, corner_coords):
            sp.generate_fixed_constraints(np.array([corner]), np.array([coord]))

        super().__init__(sp)
        self.update_scaling()
        self.symmetric_experiment = False
        self.symmetric_copies = 6

    def update_scaling(self):
        self.ad_hoc_scaling = np.sqrt(4 / math.sqrt(3))

    def tiling_coloring_number(self):
        return 6

    def get_global_transformation_type(self):
        return MapType.IDENTITY

    def get_torus_directions(self):
        return np.array([3, math.sqrt(3)]), np.array([3, -math.sqrt(3)])

    def get_torus_cover(self, vertices, sides):
        theta = math.pi / 3
        R60 = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        )
        R60 = AffineTrans(R60, np.array([0, 0]))

        first = AffineTrans(np.eye(2), np.array([0, -math.sqrt(3)]), 0, (0, 0))

        A = [first]
        for i in range(5):
            A.append(R60.compose(A[-1], i + 1, (0, 0)))

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

    def get_torus_cover_symmetry(self, vertices, sides):
        first = AffineTrans(np.eye(2), np.array([0, 0]), 0, (0, 0))
        second = AffineTrans(np.eye(2), 0.5 * np.array([3, math.sqrt(3)]), 1, (0, 0))
        third = AffineTrans(np.eye(2), 0.5 * np.array([3, -math.sqrt(3)]), 2, (0, 0))
        fourth = AffineTrans(np.eye(2), 0.5 * np.array([6, 0]), 3, (0, 0))
        return [first, second, third, fourth]

    def get_symmetry_map(self, vertices, sides):
        if not self.symmetric_experiment:
            self.symmetric_experiment = True
            self.get_torus_cover_base = self.get_torus_cover
            self.get_torus_cover = self.get_torus_cover_symmetry

        A = self.get_torus_cover_base(vertices, sides)
        scaledown = AffineTrans(np.eye(2) * 0.5, np.array([0, 0]), 0, (0, 0))
        A = [scaledown.compose(a, i, (0, 0)) for i, a in enumerate(A)]
        return A
