import numpy as np
import torch
from torch.nn import Module
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import torchvision
import trimesh
from math import cos, pi, sin


def square_mesh(resolution, num_labels=1):
    # Create a grid of points in [-1,1] x [-1,1]
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    xv, yv = np.meshgrid(x, y)
    points = np.column_stack((xv.ravel(), yv.ravel()))

    # faces_1
    faces_1 = np.concatenate(
        [np.array([[i, i + 1, i + resolution + 1]]) for i in range(resolution**2)]
    )
    mask_1 = np.array(
        [
            (i % resolution != resolution - 1) and (i + resolution + 1 < resolution**2)
            for i in range(resolution**2)
        ]
    )
    faces_1[resolution - 2] = [
        resolution - 2,
        resolution - 1,
        resolution - 2 + resolution,
    ]
    idx_1 = resolution**2 - 2 * resolution
    faces_1[idx_1] = [idx_1, idx_1 + 1, idx_1 + resolution]
    faces_1 = faces_1[mask_1]

    # faces_2
    faces_2 = np.concatenate(
        [
            np.array([[i + resolution, i, i + resolution + 1]])
            for i in range(resolution**2)
        ]
    )
    mask_2 = np.array(
        [
            (i % resolution != resolution - 1) and (i + resolution + 1 < resolution**2)
            for i in range(resolution**2)
        ]
    )
    faces_2[resolution - 2] = [
        resolution - 1,
        resolution - 1 + resolution,
        resolution - 2 + resolution,
    ]
    idx_2 = resolution**2 - 2 * resolution
    faces_2[idx_2] = [
        resolution**2 - resolution + 1,
        resolution**2 - resolution,
        idx_2 + 1,
    ]
    faces_2 = faces_2[mask_2]

    faces = np.concatenate([faces_1, faces_2])

    #  Label faces
    mask = []
    for tri in faces:
        pa, pb, pc = points[tri]

        if num_labels in (1, 2):
            # Face is labeled True if all x <= y; otherwise False
            mask.append(not ((pa[0] > pa[1]) or (pb[0] > pb[1]) or (pc[0] > pc[1])))
        else:
            # num_labels is a perfect square => grid-based split
            grid_size = int(np.sqrt(num_labels))
            bin_size = 1.0 / grid_size
            # Rescale vertices from [-1,1] to [0,1]
            pa, pb, pc = (pa + 1) / 2, (pb + 1) / 2, (pc + 1) / 2

            # Determine the "max" bin index for this triangle (upper-left corner in bin space)
            x_bin = min(
                max(pa[0] // bin_size, pb[0] // bin_size, pc[0] // bin_size),
                grid_size - 1,
            )
            y_bin = min(
                max(pa[1] // bin_size, pb[1] // bin_size, pc[1] // bin_size),
                grid_size - 1,
            )

            # 2D bins → single label index
            mask.append(int(x_bin + y_bin * grid_size))

    mask = np.array(mask)

    # Split faces by labels
    if num_labels == 1:
        faces_split = [faces]
    elif num_labels == 2:
        faces_split = [faces[mask], faces[~mask]]
    else:
        faces_split = [faces[mask == label] for label in range(num_labels)]

    return points, faces, faces_split, mask


def save_mesh(fname, vertices, triangles, uvs, texture_image=None):
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uvs)
    mesh.export(fname, file_type="obj")

    if texture_image is not None:
        save_path = Path(fname).parent / "material_0.png"
        tex_img = texture_image.detach().cpu().squeeze()

        if tex_img.shape[2] < 5 and tex_img.shape[0] > 5:
            tex_img = tex_img.permute(2, 0, 1)

        tex_img = tex_img.flip([1])

        torchvision.utils.save_image(tex_img, save_path.as_posix())


def split_square_boundary(V, bdry):
    x, y = V[bdry].T
    top = take_piece_and_sort(x, y, bdry)
    right = take_piece_and_sort(-y, x, bdry)
    bottom = take_piece_and_sort(-x, -y, bdry)
    left = take_piece_and_sort(y, -x, bdry)
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def take_piece_and_sort(sort_coord, choose_coord, bdry):
    inds = np.argsort(sort_coord)
    chosen = choose_coord[inds] == choose_coord.max()
    return bdry[inds[chosen]]


def check_triangle_orientation(V, T):
    V = V.cpu().detach().numpy() if isinstance(V, torch.Tensor) else V
    T = T.cpu().detach().numpy() if isinstance(T, torch.Tensor) else T

    pa, pb, pc = (
        V[T[:, 0]].astype(np.float64),
        V[T[:, 1]].astype(np.float64),
        V[T[:, 2]].astype(np.float64),
    )

    det = np.cross(pb - pa, pc - pa)

    if det.max() < 0:
        det = -det

    min_det = det.min()

    if min_det < -1e-6:
        print(min_det)
        print(f"min: {min_det}")
        with open("triangle_orientation_error.pkl", "wb") as f:
            pickle.dump({"V": V, "T": T, "min": min_det}, f)
            print("Triangle orientation is wrong!")


def vertex_augmentation(mapped, ROTATION_MATRIX, BATCH_SIZE=1):
    batch_vertices = []
    for imb in range(BATCH_SIZE):
        if imb == 0:
            rmat = torch.eye(2)
            delta = torch.tensor([0.0, 0.0])
        else:
            theta = torch.randn(1) * (pi / 4)
            rmat = torch.tensor([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
            delta = torch.randn([1, 2]) * 0.1

        rmat = rmat.cuda()
        delta = delta.cuda()

        rmat = ROTATION_MATRIX @ rmat

        vertices = mapped @ rmat.t() + delta

        mm = (vertices**2).sum(dim=1).max().sqrt()
        vertices = 0.9 * vertices / mm

        ones = torch.ones(vertices.shape[0], 1, device=vertices.device)
        vertices = torch.cat((vertices, ones), dim=1)

        batch_vertices.append(vertices)

    return torch.stack(batch_vertices)


class EqualAreaLoss(Module):
    def __init__(self):
        super().__init__()
        self.vals = []

    @staticmethod
    def compute_area(V: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        # Extract vertices for each face
        a, b, c = V[faces[:, 0]], V[faces[:, 1]], V[faces[:, 2]]

        # Pad to convert to homogeneous coordinates if necessary
        a, b, c = (
            a.pad((0, 1), "constant", 1),
            b.pad((0, 1), "constant", 1),
            c.pad((0, 1), "constant", 1),
        )

        # Compute vectors for two edges of each face
        ab, ac = b - a, c - a

        # Compute cross product and area
        cross_prod = torch.cross(ab, ac, dim=1)
        area = torch.norm(cross_prod, dim=1) * 0.5

        return area.sum()

    def equal_area_loss(self, V: torch.Tensor, faces_split: list) -> torch.Tensor:
        # Compute areas for each subset of faces
        areas = torch.stack([self.compute_area(V, faces) for faces in faces_split])

        # Calculate mean area
        mean_area = areas.mean()

        # Compute loss as the sum of squared deviations from the mean area
        loss = torch.sum((areas - mean_area.detach()) ** 2)

        # Store areas for visualization
        self.vals.append(areas.detach().cpu().numpy())

        return loss

    def save_curves(self, path: str) -> None:
        if not self.vals:
            return

        plt.figure(figsize=(10, 5))
        for area_set in zip(*self.vals):
            plt.plot(area_set, alpha=0.6)
        plt.xlabel("Iteration")
        plt.ylabel("Area")
        plt.title("Area Distribution Over Iterations")
        plt.savefig(path)
        plt.close()
