import torch
import numpy as np


from escher.OTE.ote import AffineTrans, MapType, SparseSystem, Constraints


class TorusConstraints(Constraints):
    def __init__(self, vertices, sides):
        self.sides = sides
        topright = sides["top"][-1]
        topleft = sides["top"][0]
        bottomleft = sides["bottom"][-1]
        bottomright = sides["bottom"][0]
        corners = [topleft, topright, bottomleft, bottomright]
        corner_coords = [[-1, 1], [1, 1], [-1, -1], [1, -1]]

        bottom = np.flip(sides["bottom"])
        top = sides["top"]
        left = sides["left"]
        right = np.flip(sides["right"])

        sp = SparseSystem(vertices.shape[0])
        sp.generate_translation_constraint(bottom[1:-1], top[1:-1], np.array([0, 2]))
        sp.generate_translation_constraint(left[1:-1], right[1:-1], np.array([2, 0]))

        for corner, coord in zip(corners, corner_coords):
            sp.generate_fixed_constraints(np.array([corner]), np.array([coord]))

        super().__init__(sp)

    def update_scaling(self):
        self.ad_hoc_scaling = 1

    def get_global_transformation_type(self):
        return MapType.SKEW

    def get_torus_directions(self):
        return np.array([0, 4]), np.array([4, 0])

    def get_torus_cover(self, vertices, sides):
        I = np.identity(2)
        return [
            AffineTrans(I, np.array([0, 0]), 0),
            AffineTrans(I, np.array([0, 2]), 1),
            AffineTrans(I, np.array([2, 0]), 2),
            AffineTrans(I, np.array([2, 2]), 3),
        ]

    def get_boundary(self):
        sides = self.sides
        return [
            (sides["top"], sides["bottom"][-1::-1]),
            (sides["left"], sides["right"][-1::-1]),
        ]

    def tiling_coloring_number(self):
        return 8

    def get_horizontal_symmetry_orientation(self):
        theta = np.pi / 4
        return torch.Tensor(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
