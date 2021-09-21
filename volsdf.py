import numpy as np
import torch
from torch import nn
import radam # pip install git+https://github.com/LiyuanLucasLiu/RAdam/

from run_nerf import DEBUG
import nerf

import os
import contextlib

# https://github.com/lioryariv/idr/blob/main/code/model/implicit_differentiable_renderer.py
class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0
    ):
        super().__init__()

        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = nerf.get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def predict_sdf(self, coord, compute_normal=False):
        """
        coord:
            `torch.Tensor`, shape = `(..., 3)`
        compute_normal:
            `bool`
            If `True`, also return SDF gradient.
        """
        if compute_normal:
            coord.requires_grad_(True)
            maybe_enable_grad = torch.enable_grad
        else:
            maybe_enable_grad = contextlib.nullcontext

        with maybe_enable_grad():
            prediction = self.forward(coord)
            sdf, latent_features = prediction[..., :1], prediction[..., 1:]

            if compute_normal:
                # Prevent `coord.grad` from getting overwritten/accumulated
                old_coord_grad = coord.grad # almost always `None`

                d_output = torch.ones_like(sdf)
                sdf.backward(
                    d_output, create_graph=self.training, inputs=[coord])
                sdf_normal = coord.grad

                coord.grad = old_coord_grad

                return sdf, latent_features, sdf_normal
            else:
                return sdf, latent_features


# https://github.com/lioryariv/idr/blob/44959e7aac267775e63552d8aac6c2e9f2918cca/code/model/implicit_differentiable_renderer.py#L99
class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = nerf.get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = x.sigmoid()
        # Widen the sigmoid output a little bit
        # to let the network yield 0.0 and 1.0 easier
        widen_factor = 0.04
        x = x * (1 + widen_factor * 2) - widen_factor

        return x


class VolSDF(nn.Module):
    def __init__(self, args):
        super().__init__()

        assert args.use_viewdirs, "NYI"

        self.embedding_coords_fn, embedding_dims_coords = \
            nerf.get_embedder(args.multires, args.i_embed)
        self.embedding_viewdirs_fn, embedding_dims_viewdirs = \
            nerf.get_embedder(args.multires_views, args.i_embed)

        self.r = args.r

        # TODO move to args
        feature_vector_size = 256

        self.f = ImplicitNetwork(
            feature_vector_size=feature_vector_size,
            d_in=3,
            d_out=1,
            dims=[256] * 8,
            geometric_init=True,
            bias=1.0,
            skip_in=(4,),
            weight_norm=True,
            multires=args.multires)

        self.L = RenderingNetwork(
            feature_vector_size=feature_vector_size,
            mode='idr',
            d_in=3+3+3, # coordinate, SDF normal, view direction
            d_out=3,
            dims=[256] * 4, # TODO move to args
            weight_norm=True,
            multires_view=args.multires_views)

        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, coord, view_direction):
        sdf, latent_features, sdf_normal = self.f.predict_sdf(coord, compute_normal=True)
        sdf = torch.minimum(sdf, self.r - coord.norm(2, dim=-1, keepdim=True))

        light_field = self.L(coord, sdf_normal, view_direction, latent_features)

        def laplace_cdf(x, beta):
            retval = torch.empty_like(x)

            mask = x > 0
            retval[mask] = 1.0 - 0.5 * (x[mask] / -beta).exp()

            mask = ~mask
            retval[mask] = 0.5 * (x[mask] / beta).exp()

            return retval

        # alpha and beta are Laplace CDF parameters
        alpha = self.beta.reciprocal() # hard setting from the paper
        density = alpha * laplace_cdf(-sdf, self.beta)

        return torch.cat((light_field, density), dim=-1), sdf_normal


def render_rays(ray_batch, # of length <= `args.chunk`
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                white_bkgd=False,
                raw_noise_std=0.,
                r=3.0):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [N, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: not used, only needed for saving weights, TODO remove.
      network_query_fn: instance of `VolSDF` wrapped in `nerf.batchify()`
      N_samples: int. Number samples along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: random perturbation for density values (as in NeRF).
    Returns a dict with these keys:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray.
      raw: [num_rays, num_samples, 4]. Raw predictions from model (R, G, B, density).
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[..., 0:3], ray_batch[..., 3:6] # [N_rays, 3] each
    viewdirs = ray_batch[..., -3:] if ray_batch.shape[-1] > 8 else None # [N_rays, 3]

    # see "Modeling the background" in Section B.3
    M = 2 * r
    near, far = 0, M

    # Normalize ray directions. Just to be safe -- not sure if this is really
    # needed (e.g. this wasn't done in 'nerf-pytorch')
    rays_d_norm = rays_d.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    rays_d = rays_d / rays_d_norm

    # TODO implement proper sampling Τ (see Algorithm 1)

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    # TODO expand only after applying positional encoding to avoid extra computations
    viewdirs = viewdirs[:, None].expand(pts.shape)

    # Predict light field and density at points sampled on rays
    rgb_and_density, _ = network_query_fn(pts, viewdirs)

    points_for_eikonal_loss = torch.empty((N_rays, 3), device=pts.device).uniform_(-r/2, r/2)
    _, _, sdf_normal_for_eikonal_loss = network_fn.f.predict_sdf(
        points_for_eikonal_loss, compute_normal=True)
    # rgb_and_density: [N_rays, N_samples, 4]
    # sdf_normal: [N_rays, N_samples, 3]

    # Convert predictions to pixel values with numerical integration
    rgb_map, disp_map, acc_map, weights, depth_map = nerf.raw2outputs(
        rgb_and_density, z_vals, rays_d,
        rgb_activation_fn=lambda x: x, density_activation_fn=lambda x: x,
        raw_noise_std=raw_noise_std, white_bkgd=white_bkgd)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = rgb_and_density
    ret['sdf_normal'] = sdf_normal_for_eikonal_loss

    for k in ret:
        if DEBUG and (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def create_model(args):
    model = VolSDF(args)

    # Create optimizer
    grad_vars = model.parameters()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    elif args.optimizer == 'radam':
        optimizer = radam.RAdam(params=grad_vars, lr=args.lrate)

    start = 0

    # Load checkpoints
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        experiment_dir = os.path.join(basedir, expname)
        ckpts = [os.path.join(experiment_dir, f) for f in sorted(os.listdir(experiment_dir)) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=args.device)

        def remove_prefix(state_dict, prefix):
            if all(k.startswith(prefix) for k in state_dict):
                return {k[len(prefix):] : v for k, v in state_dict.items()}
            else:
                assert all(not k.startswith(prefix) for k in state_dict)
                return state_dict

        for k in 'network_fn_state_dict',:
            ckpt[k] = remove_prefix(ckpt[k], 'module.')

        start = ckpt['global_step'] + 1
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': nerf.batchify(model, args.netchunk),
        'perturb' : args.perturb,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'render_rays_fn': render_rays,
        'r': args.r,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    # render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, optimizer
