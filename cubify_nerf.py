# Run with `python3 cubify_nerf.py --config configs/lego.txt`
import numpy as np
import torch
from tqdm import tqdm

# install from https://github.com/tatsy/mcubes_pytorch/
from mcubes_module import mcubes_cpu as marching_cubes
import run_nerf
import run_nerf_helpers

import pytorch3d.ops
import pytorch3d.io
import pytorch3d.structures

from pathlib import Path

def compute_nerf_opacity_at_grid(
    embed_fn, nerf_network, device, bbox_bounds, num_points_per_axis, hwf=None):
    """
    embed_fn:
        callable
        As returned by `run_nerf.get_embedder()`.
    nerf_network:
        callable
        As in `render_kwargs_test['network_fine']` from `run_nerf.create_nerf()`.
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
        *bounds_1d, num_points_per_axis, device=run_nerf.device) for bounds_1d in bbox_bounds]
    meshgrids = torch.meshgrid(grids_1d)
    meshgrids = torch.stack(meshgrids, -1)

    # NDC
    if hwf is not None:
        meshgrids = run_nerf_helpers.ndc_rays(*hwf, 1.0, meshgrids)

    # NeRF's output that will be later fed to marching cubes -- an occupancy voxel grid
    opacity = torch.empty_like(meshgrids[..., 0])

    # Iterate the grid and simply query the opacity from NeRF, without ray integration
    with torch.no_grad():
        # Iterate through x-constant planes
        for plane_coords, plane_opacity in zip(tqdm(meshgrids), opacity):
            plane_coords = embed_fn(plane_coords)

            prediction = nerf_network.predict_alpha(plane_coords)[..., 0]
            prediction = 1.0 - torch.exp(-prediction.relu())

            plane_opacity.copy_(prediction)

    return opacity


def main():
    parser = run_nerf.config_parser()
    args = parser.parse_args()

    ########### Constants for configs/lego.txt ###########
    # TODO: move them to config

    # Where to save the mesh
    DESTINATION_FILE = Path("./mesh.obj")

    # Bounding box limits
    # bbox = ((-1.5, 2.0), (-1.7, 1.0), (-1.0, 1.0)) # lego
    bbox = ((-3.5, 3.5), (-2.3, 2.7), (-8.5, -0.05)) # trex
    # How many points in the grid to sample
    N = 125

    # Tells marching cubes at which opacity value to discretize the voxel grid
    OPACITY_THRESHOLD = 0.4
    ########### End constants  ###########

    if DESTINATION_FILE.exists():
        raise FileExistsError(DESTINATION_FILE)

    # Create NeRF model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = run_nerf.create_nerf(args)
    nerf = render_kwargs_test['network_fine']
    embed_fn, input_ch = run_nerf.get_embedder(args.multires, args.i_embed)

    if args.dataset_type != 'llff' or args.no_ndc:
        hwf = None
    else:
        _, _, _, _, _, _, hwf, _, _, _ = run_nerf.load_dataset(args)

    opacity = compute_nerf_opacity_at_grid(embed_fn, nerf, run_nerf.device, bbox, N, hwf)

    # Run marching cubes on the opacity grid that we just received
    verts, faces = marching_cubes(opacity.cpu(), OPACITY_THRESHOLD)
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
    meshes = pytorch3d.structures.Meshes([verts], [faces])
    pytorch3d.io.IO().save_mesh(meshes, DESTINATION_FILE)

if __name__ == "__main__":
    main()
