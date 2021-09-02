# Run with `python3 cubify_nerf.py --config configs/lego.txt`
import numpy as np
import torch
from tqdm import tqdm

# install from https://github.com/tatsy/torchmcubes
from torchmcubes import marching_cubes
import run_nerf, nerf, volsdf

import pytorch3d.ops
import pytorch3d.io
import pytorch3d.structures

from pathlib import Path

def compute_mlp_at_grid(
    predict_mlp_fn, device, bbox_bounds, num_points_per_axis, hwf=None):
    """
    Compute NeRF's density or VolSDF's SDF at a uniform coordinate grid.

    predict_mlp_fn:
        callable
        Input: tensor of xyz coordinates of shape (..., 3).
        Output: tensor of MLP regression predictions (density or SDF) of shape (...).
    nerf_network:
        callable
        As in `render_kwargs_test['network_fine']` from `nerf.create_model()`.
    device:
        str
        A PyTorch device.
    bbox_bounds:
        `tuple` of `tuple` of `float`
        Tuple ((xmin, xmax), (ymin, ymax), (zmin, zmax)) denoting grid start and end for all axes.
    num_points_per_axis:
        `int`
        How many points in the grid to sample over each axis.
    """
    # Construct the coordinates grid, which is the input to NeRF
    grids_1d = [torch.linspace(
        *bounds_1d, num_points_per_axis, device=device) for bounds_1d in bbox_bounds]
    meshgrids = torch.meshgrid(grids_1d)
    meshgrids = torch.stack(meshgrids, -1)

    # NDC
    if hwf is not None:
        meshgrids = run_nerf.ndc_rays(*hwf, 1.0, meshgrids)

    # NeRF's output that will be later fed to marching cubes -- an occupancy voxel grid
    predictions = torch.empty_like(meshgrids[..., 0])

    # Iterate the grid and simply
    with torch.no_grad():
        # Iterate through x-constant planes
        for plane_coords, plane_predictions in zip(tqdm(meshgrids), predictions):
            prediction = predict_mlp_fn(plane_coords)
            plane_predictions.copy_(prediction)

    return predictions


def main():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = run_nerf.config_parser()
    args = parser.parse_args()
    assert args.ft_path is not None

    ########### Constants for configs/lego.txt ###########
    # TODO: move them to config

    # Where to save the mesh
    DESTINATION_FILE = Path("./mesh.obj")

    # Bounding box limits
    bbox = ((-1.5, 2.0), (-1.7, 1.0), (-1.0, 1.08)) # lego
    # bbox = ((-3.5, 3.5), (-2.3, 2.7), (-8.5, -0.05)) # trex
    # How many points in the grid to sample
    N = 275

    # Tells marching cubes at which density value to discretize the voxel grid (in case of NeRF)
    DENSITY_THRESHOLD = 0.4
    ########### End constants  ###########

    if DESTINATION_FILE.exists():
        raise FileExistsError(DESTINATION_FILE)

    if args.dataset_type != 'llff' or args.no_ndc:
        hwf = None
    else:
        _, _, _, _, _, _, hwf, _, _, _ = run_nerf.load_dataset(args)

    if args.model == 'nerf':
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = \
            nerf.create_model(args)

        model = render_kwargs_test['network_fine']
        embed_fn, input_ch = nerf.get_embedder(args.multires, args.i_embed)

        def predict_mlp_fn(coords):
            # query the density from NeRF by coordinates, without ray integration
            coords = embed_fn(coords)
            prediction = model.predict_alpha(coords)[..., 0]
            prediction = 1.0 - torch.exp(-prediction.relu())
            return prediction

        marching_cubes_threshold = DENSITY_THRESHOLD

    elif args.model == 'volsdf':
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = \
            volsdf.create_model(args)

        model = render_kwargs_test['network_fn']

        def predict_mlp_fn(coords):
            sdf, _ = model.f.predict_sdf(coords)
            return sdf[..., 0]

        marching_cubes_threshold = 0.0

    density_or_sdf = compute_mlp_at_grid(predict_mlp_fn, args.device, bbox, N, hwf)

    # Run marching cubes on the density (or SDF) grid that we just received
    verts, faces = marching_cubes(density_or_sdf.cpu(), marching_cubes_threshold)
    print(
        f"The computed mesh is these tensors:\n"
        f"  Vertices: {verts.dtype}, {verts.shape}\n"
        f"  Faces:    {faces.dtype}, {faces.shape}")

    # For some reason, `marching_cubes()` outputs dimensions in the wrong order
    verts = verts[:, (2, 1, 0)]

    # Rescale mcubes output coordinates to match NeRF
    for verts_dimension, (bbox_min, bbox_max) in zip(verts.t(), bbox):
        verts_dimension *= (bbox_max - bbox_min) / (N - 1)
        verts_dimension += bbox_min

    # Save the output of marching cubes to disk
    print(f"Saving the mesh to {DESTINATION_FILE}")
    meshes = pytorch3d.structures.Meshes([verts], [faces])
    pytorch3d.io.IO().save_mesh(meshes, DESTINATION_FILE)

if __name__ == "__main__":
    main()
