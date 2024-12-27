import torch
import math
import numpy as np
from abc import ABC, abstractmethod


def map(vertices, A):
    A = A.T.to(vertices.device).type(vertices.dtype)
    return vertices @ A


class MapType:
    IDENTITY = 0
    NON_ISOTROPIC_SCALE = 1
    SKEW = 2


class GlobalDeformation:
    def __init__(
        self,
        init_rotation=torch.eye(2),
        device="cpu",
        singular_value_bound=2,
        random_init=False,
    ):
        device = torch.device(device)
        self.init_rotation = init_rotation.to(device)
        self.singular_value_bound = singular_value_bound

        if random_init:
            self.theta1 = torch.nn.Parameter(torch.rand(1, device=device) * 5)
            self.theta2 = torch.nn.Parameter(torch.rand(1, device=device) * 5)
            self.singular_value = torch.nn.Parameter(
                torch.rand(1, device=device) * 1000
            )
        else:
            self.theta1 = torch.nn.Parameter(torch.zeros(1, device=device))
            self.theta2 = torch.nn.Parameter(torch.zeros(1, device=device))
            self.singular_value = torch.nn.Parameter(torch.ones(1, device=device))

    def _rotation_from_angle(self, theta):
        cs, sn = torch.cos(theta), torch.sin(theta)
        r1 = torch.cat([cs, -sn], dim=0)
        r2 = torch.cat([sn, cs], dim=0)
        return torch.stack((r1, r2), dim=1)

    def get_matrix(self, global_rotation: bool, map_type: MapType):
        R2 = (
            self._rotation_from_angle(self.theta1)
            if global_rotation
            else torch.eye(2, device=self.theta1.device)
        )
        if map_type != MapType.IDENTITY:
            sv = (
                torch.sigmoid(self.singular_value)
                * (self.singular_value_bound - 1 / self.singular_value_bound)
                + 1 / self.singular_value_bound
            )
        else:
            sv = torch.ones(1, device=self.theta1.device)
        D = torch.diag(torch.cat([sv, 1 / sv]))
        R1 = (
            self._rotation_from_angle(self.theta1)
            if map_type == MapType.SKEW
            else torch.eye(2, device=R2.device)
        )
        return self.init_rotation @ R2 @ D @ R1


class AffineTrans:
    def __init__(self, A, b, orientation_index=0, translation_index=(0, 0)):
        self.A = A.cpu().detach().numpy() if torch.is_tensor(A) else A
        self.b = b.cpu().detach().numpy() if torch.is_tensor(b) else b
        self.orientation_index = orientation_index
        self.translation_index = translation_index

    def map(self, vertices):
        if torch.is_tensor(vertices):
            A = torch.from_numpy(self.A.T).to(vertices.device).float()
            if vertices.ndimension() == 3:
                A = A.unsqueeze(0)
            b = torch.from_numpy(self.b).to(vertices.device).type(vertices.dtype)
            return vertices @ A + b
        else:
            return vertices @ self.A.T + self.b

    def compose(self, other, orientation_index, translation_index):
        combined_A = self.A @ other.A
        combined_b = other.b @ self.A.T + self.b
        return AffineTrans(
            combined_A,
            combined_b,
            orientation_index,
            translation_index,
        )


class SparseSystem:
    def __init__(self, n_vertices):
        self.I = []
        self.J = []
        self.V = []
        self.b = []
        self.n_vertices = n_vertices

    def add(self, I, J, V, b):
        self.I.append(I)
        self.J.append(J)
        self.V.append(V)
        self.b.append(b)

    def aggregate(self):
        offset = 0
        new_I = []
        for i in self.I:
            i_flat = i.ravel()
            new_I.append(i_flat + offset)
            offset += i_flat.max() + 1
        J = [j.flatten() for j in self.J]
        V = [v.flatten() for v in self.V]
        b = [bb.flatten() for bb in self.b]
        return (
            np.concatenate(new_I),
            np.concatenate(J, axis=0),
            np.concatenate(V, axis=0),
            np.concatenate(b, axis=0),
        )

    def generate_rotation_constraints(self, side1, side2, r1, delta):
        c, s = math.cos(r1), math.sin(r1)
        alpha = -delta
        alpha[0] += c * delta[0] - s * delta[1]
        alpha[1] += s * delta[0] + c * delta[1]
        left_x, left_y = side1, side1 + self.n_vertices
        top_x, top_y = side2, side2 + self.n_vertices

        J_1 = np.stack([left_x, left_y, top_x], axis=1)
        I_1 = np.array(range(J_1.shape[0]))
        b_1 = np.full(I_1.max() + 1, alpha[0])
        I_1 = np.stack([I_1, I_1, I_1], axis=1)
        ones = np.ones(J_1.shape[0])
        V_1 = np.stack([c * ones, -s * ones, -ones], axis=1)

        J_2 = np.stack([left_x, left_y, top_y], axis=1)
        I_2 = np.arange(J_2.shape[0])
        b_2 = np.full(I_2.max() + 1, alpha[1])
        I_2 = np.stack([I_2, I_2, I_2], axis=1) + I_1.max() + 1
        ones = np.ones(J_2.shape[0])
        V_2 = np.stack([s * ones, c * ones, -ones], axis=1)

        I = np.concatenate((I_1, I_2), axis=0)
        J = np.concatenate((J_1, J_2), axis=0)
        V = np.concatenate((V_1, V_2), axis=0)
        b = np.concatenate((b_1, b_2), axis=0)

        assert b.shape == I.max() + 1
        self.add(I, J, V, b)
        return I, J, V, b

    def generate_straight_line_constraint(self, side, normal):
        side_x, side_y = side, side + self.n_vertices
        J = np.stack([side_x[1:], side_x[:-1], side_y[1:], side_y[:-1]], axis=1)
        I = np.arange(J.shape[0])
        I = np.stack([I, I, I, I], axis=1)
        b = np.zeros(J.shape[0])
        ones = np.ones(J.shape[0])
        V = np.stack(
            [normal[0] * ones, -normal[0] * ones, normal[1] * ones, -normal[1] * ones],
            axis=1,
        )
        self.add(I, J, V, b)
        return I, J, V, b

    def generate_relative_sum_constraint(self, side1, p1, side2, p2, x_axis, sum=0):
        sign = 1
        side1_x, side1_y = side1, side1 + self.n_vertices
        side2_x, side2_y = side2, side2 + self.n_vertices
        p1_x, p1_y = p1, p1 + self.n_vertices
        p2_x, p2_y = p2, p2 + self.n_vertices

        if x_axis:
            J = np.stack(
                [
                    side1_x,
                    np.full(side1_x.shape[0], p1_x),
                    side2_x,
                    np.full(side1_x.shape[0], p2_x),
                ],
                axis=1,
            )
        else:
            J = np.stack(
                [
                    side1_y,
                    np.full(side1_y.shape[0], p1_y),
                    side2_y,
                    np.full(side1_y.shape[0], p2_y),
                ],
                axis=1,
            )

        I = np.arange(J.shape[0])
        b = np.full(I.max() + 1, sum)
        I = np.stack([I, I, I, I], axis=1)
        ones = np.ones(J.shape[0])
        V = np.stack([ones * sign, -ones * sign, ones, -ones], axis=1)
        assert b.shape[0] == I.max() + 1
        self.add(I, J, V, b)
        return I, J, V, b

    def generate_relative_translation_constraint(
        self, side1, p1, side2, p2, x_axis, reflect=False, shift=0
    ):
        sign = -1 if reflect else 1
        side1_x, side1_y = side1, side1 + self.n_vertices
        side2_x, side2_y = side2, side2 + self.n_vertices
        p1_x, p1_y = p1, p1 + self.n_vertices
        p2_x, p2_y = p2, p2 + self.n_vertices

        if x_axis:
            J = np.stack(
                [
                    side1_x,
                    np.full(side1_x.shape[0], p1_x),
                    side2_x,
                    np.full(side1_x.shape[0], p2_x),
                ],
                axis=1,
            )
        else:
            J = np.stack(
                [
                    side1_y,
                    np.full(side1_y.shape[0], p1_y),
                    side2_y,
                    np.full(side2_y.shape[0], p2_y),
                ],
                axis=1,
            )

        I = np.arange(J.shape[0])
        b = np.full(I.max() + 1, shift)
        I = np.stack([I, I, I, I], axis=1)
        ones = np.ones(J.shape[0])
        V = np.stack([-ones * sign, ones * sign, ones, -ones], axis=1)
        assert b.shape[0] == I.max() + 1
        self.add(I, J, V, b)
        return I, J, V, b

    def generate_translation_constraint(self, side1, side2, delta, reflect_x=False):
        side1_x, side1_y = side1, side1 + self.n_vertices
        side2_x, side2_y = side2, side2 + self.n_vertices
        sgn = -1 if reflect_x else 1

        J_1 = np.stack([side1_x, side2_x], axis=1)
        I_1 = np.arange(J_1.shape[0])
        b_1 = np.full(I_1.max() + 1, delta[0])
        I_1 = np.stack([I_1, I_1], axis=1)
        ones = np.ones(J_1.shape[0])
        V_1 = np.stack([-ones, sgn * ones], axis=1)

        J_2 = np.stack([side1_y, side2_y], axis=1)
        I_2 = np.arange(J_2.shape[0])
        b_2 = np.full(I_2.max() + 1, delta[1])
        I_2 = np.stack([I_2, I_2], axis=1) + I_1.max() + 1
        ones = np.ones(J_2.shape[0])
        V_2 = np.stack([-ones, ones], axis=1)

        I = np.concatenate([I_1, I_2])
        J = np.concatenate([J_1, J_2])
        V = np.concatenate([V_1, V_2])
        b = np.concatenate([b_1, b_2])
        assert b.shape[0] == I.max() + 1
        self.add(I, J, V, b)
        return I, J, V, b

    def generate_fixed_constraints_x(self, side, points_x):
        J = side
        I = np.arange(J.shape[0])
        b = np.array(points_x)
        V = np.ones(J.shape[0])
        assert b.shape[0] == I.max() + 1
        self.add(I, J, V, b)
        return I, J, V, b

    def generate_fixed_constraints_y(self, side, points_y):
        J = side + self.n_vertices
        I = np.arange(J.shape[0])
        b = np.array(points_y)
        V = np.ones(J.shape[0])
        assert b.shape[0] == I.max() + 1
        self.add(I, J, V, b)
        return I, J, V, b

    def generate_fixed_constraints(self, side, points):
        assert len(points.shape) == 2
        self.generate_fixed_constraints_x(side, points[:, 0])
        self.generate_fixed_constraints_y(side, points[:, 1])


class Constraints(ABC):
    def __init__(self, sp: SparseSystem):
        I, J, vals, b = sp.aggregate()
        self.cIJ = np.stack([I, J], axis=1)
        self.cV = vals
        self.b = b
        self.tile_width = 2
        self.ad_hoc_scaling = 1.0

    def get_torus_directions(self):
        return np.array([self.tile_width, 0]), np.array([0, self.tile_width])

    def get_torus_directions_with_scaling(self):
        vec_1, vec_2 = self.get_torus_directions()
        return vec_1 * self.ad_hoc_scaling, vec_2 * self.ad_hoc_scaling

    def get_torus_cover_with_scaling(self, vertices, sides):
        torus_cover = self.get_torus_cover(vertices, sides)
        scaling = AffineTrans(np.eye(2) * self.ad_hoc_scaling, [0, 0], 0, (0, 0))
        return [
            scaling.compose(gen, gen.orientation_index, (0, 0)) for gen in torus_cover
        ]

    @abstractmethod
    def tiling_coloring_number(self):
        pass

    @abstractmethod
    def get_boundary(self):
        pass

    @abstractmethod
    def get_torus_cover(self, vertices, sides):
        pass

    @abstractmethod
    def get_horizontal_symmetry_orientation(self):
        pass

    @abstractmethod
    def get_global_transformation_type() -> MapType:
        pass

    def get_torus_tiling_width(self):
        return 2 * self.tile_width

    def get_tiling(self, half_grid_size, vertices, sides):
        I = np.eye(2)
        vec1, vec2 = self.get_torus_directions_with_scaling()
        torus_maps = self.get_torus_cover_with_scaling(vertices, sides)
        maps = [
            AffineTrans(I, sign_i * i * vec1 + sign_j * j * vec2, None, None).compose(
                gen, gen.orientation_index, (i, j)
            )
            for i in range(half_grid_size + 1)
            for j in range(half_grid_size + 1)
            for sign_i in [-1, 1]
            if not (i == 0 and sign_i == -1)
            for sign_j in [-1, 1]
            if not (j == 0 and sign_j == -1)
            for gen in torus_maps
        ]
        return maps

    def get_more_tiling(self, half_grid_size, previous_half_grid_size, vertices, sides):
        I = np.eye(2)
        vec1, vec2 = self.get_torus_directions_with_scaling()
        torus_maps = self.get_torus_cover_with_scaling(vertices, sides)
        maps = [
            AffineTrans(I, i * vec1 + j * vec2, None, None).compose(
                gen, gen.orientation_index, (i, j)
            )
            for i in range(-half_grid_size, half_grid_size + 1)
            for j in range(-half_grid_size, half_grid_size + 1)
            if not (
                -previous_half_grid_size <= i <= previous_half_grid_size
                and -previous_half_grid_size <= j <= previous_half_grid_size
            )
            for gen in torus_maps
        ]
        return maps

    def get_num_orientation(self, vertices, sides):
        return len(self.get_torus_cover(vertices, sides))


class OTESolver:
    def __init__(self, edge_pairs, V, constraints: Constraints):
        self.n_verts = V.shape[0]
        IJ, wInds, wFacs = [], [], []

        for i in range(edge_pairs.shape[0]):
            IJ.append((edge_pairs[i, 0], edge_pairs[i, 1]))
            IJ.append((edge_pairs[i, 1], edge_pairs[i, 0]))
            wInds.extend([i, i])
            wFacs.extend([1, 1])
            for j in range(2):
                IJ.append((edge_pairs[i, j], edge_pairs[i, j]))
                wInds.append(i)
                wFacs.append(-1)

        IJ = np.asarray(IJ)
        IJ = np.concatenate((IJ, IJ + IJ.max() + 1), axis=0)
        wInds.extend(wInds)
        wFacs.extend(wFacs)
        self.wInds = wInds
        self.wFacs = wFacs

        cIJ = constraints.cIJ.copy()
        cV = constraints.cV.copy()
        b = constraints.b.copy()

        cIJ[:, 0] += IJ[:, 0].max() + 1

        cIJt = np.stack([cIJ[:, 1], cIJ[:, 0]], axis=1)
        cIJ = np.concatenate([cIJ, cIJt], axis=0)
        cV = np.tile(cV, 2)

        self.cV = cV
        IJ = np.concatenate([IJ, cIJ], axis=0)

        self.IJ = np.concatenate((np.zeros([IJ.shape[0], 1]), IJ), axis=1)
        self.b = np.concatenate((np.zeros((V.shape[0] * 2)), b), 0)

    def build_KKT_matrix(self, w):
        Wvals = w[self.wInds] * torch.tensor(self.wFacs).unsqueeze(1)
        vals = torch.cat([Wvals.squeeze(), torch.from_numpy(self.cV)], dim=0)
        KKT = torch.sparse_coo_tensor(np.transpose(self.IJ), vals).coalesce()
        return KKT

    def solve(self, w):
        KKT = self.build_KKT_matrix(w).to(torch.float64)
        b = torch.from_numpy(self.b).unsqueeze(0).unsqueeze(2).to(torch.float64)

        A = KKT.to_dense()
        X_dense = torch.linalg.solve(A, b.squeeze())[0]
        X = X_dense if X_dense.dim() == 1 else X_dense[0]

        success = True
        residual = torch.linalg.norm(A @ X - b.squeeze())
        if residual > 1e-8:
            print(f"linear system not solved, error: {residual.item():.2e}")
            success = False

        mapped = torch.zeros([self.n_verts, 2], dtype=X.dtype)
        mapped[:, 0] = X[: self.n_verts]
        mapped[:, 1] = X[self.n_verts : self.n_verts * 2]
        return mapped, X, success
