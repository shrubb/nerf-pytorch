# TODO remove
DEBUG = False

import os
import numpy as np
import imageio
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import nerf
import volsdf

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data, load_blender_poses, save_blender_poses
from load_LINEMOD import load_LINEMOD_data

from pathlib import Path

np.random.seed(0)

# Misc
loss_functions = {
	'MSE': lambda x, y : torch.mean((x - y) ** 2), # legacy from NeRF
	'L1': torch.nn.functional.l1_loss,
}
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


def batchify_rays(rays_flat, chunk=1024*32, render_rays_fn=None, **render_kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays_fn(rays_flat[i:i+chunk], **render_kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d=None):
    # rays_o, rays_d: (rays_shape x 3), e.g. (480 x 640 x 3)

    if rays_d is not None:
        # Shift ray origins to near plane
        t = -(near + rays_o[...,2]) / rays_d[...,2]
        rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]
    rays_o = torch.stack([o0,o1,o2], -1)

    if rays_d is None:
        return rays_o
    else:
        d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
        d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
        d2 = -2. * near / rays_o[...,2]
        rays_d = torch.stack([d0,d1,d2], -1)

        return rays_o, rays_d


def render(
    H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
    near=0., far=1., use_viewdirs=False, c2w_staticcam=None, **render_kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **render_kwargs)
    # TODO remove this
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    extras = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [extras]


def render_path(
    render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0,
    fixed_viewdir=None):
    """
    Render a batch of full images.

    fixed_viewdir:
        If a 4 x 4 matrix, use this view direction for all frames.
        If `None`, use actual view direction (camera position).
    """
    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()

        if fixed_viewdir is not None:
            rgb, disp, acc, _ = render(
                H, W, K, chunk=chunk, c2w=fixed_viewdir, c2w_staticcam=c2w[:3,:4], **render_kwargs)
        else:
            rgb, disp, acc, _ = render(
                H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)

        rgb.clamp_(0.0, 1.0)

        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            acc.clamp_(0.0, 1.0)
            image = np.concatenate((rgbs[-1], acc.cpu().numpy()[..., None]), axis=-1)
            imageio.imwrite(Path(savedir) / f"{i:03}.png", image)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def load_dataset(args):
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(
            args.datadir, args.factor, recenter=args.recenter, bd_factor=args.rescale_factor,
            spherify=args.spherify, r=args.r if args.rescale_factor < 0 else None)

        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    return images, poses, render_poses, i_train, i_val, i_test, hwf, near, far, K

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--model", type=str, choices=['nerf', 'volsdf'], default='nerf',
                        help='Kind of model to use')
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--batch_size", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--num_iterations", type=int, default=200000 + 1,
                        help='total number of optimization steps')
    parser.add_argument("--loss", type=str, choices=['MSE', 'L1'], default='MSE',
                        help="Type of loss function")
    parser.add_argument("--optimizer", type=str, choices=['adam', 'radam'], default='adam',
                        help='Optimizer')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--eikonal_loss_weight", type=float, default=0.1,
                        help="VolSDF only: lambda from equation 17")
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help='Device; "cpu", "cuda", or "cuda:N"')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--r", type=float, default=3.0,
                        help="VolSDF only: 'r', scene boundary parameter from equation (19).")
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--fixed_viewdir_in_test",
                        type=lambda matrix: torch.Tensor(
                            [list(map(float, row.split(', '))) for row in matrix.split('; ')]),
                        default=None,
                        help='In val/test, use this view direction (3 x 4 matrix). '
                        'If not specified, use actual view direction (camera position).')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_poses", type=str, default='dataset_path',
                        help="Which poses to render during evaluation ('--render_only'). Can be "
                             "'test' (dataset's test split), "
                             "'dataset_path' (other dataset-defined poses, e.g. smooth spherical "
                             "trajectory in case of 'blender' datasets), or "
                             "a path to .json file with poses.")
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')
    parser.add_argument("--recenter", type=str, default='mean',
                        choices=['none', 'mean', 'intersection'],
                        help="Only for --dataset_type=llff.\n"
                             "If 'none', leave camera poses as is.\n"
                             "If 'mean', move the scene origin to the average "
                             "camera position.\n"
                             "If 'intersection', compute the least squares solution for the "
                             "intersection point of cameras' principal axes and move the scene "
                             "origin there.\n"
                             "See also: '--r'.")
    parser.add_argument("--rescale_factor", type=float, default=0.75,
                        help="Only for --dataset_type=llff.\n"
                             "Rescale the entire scene (i.e. cameras' coordinates) by approx. "
                             "this value.\n"
                             "If negative value is given, follow the camera normalization "
                             "procedure described in VolSDF section B.1, using '--r'.")

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=5000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    images, poses, render_poses, i_train, i_val, i_test, hwf, near, far, K = \
        load_dataset(args)
    H, W, focal = hwf

    if args.render_poses == 'dataset_path':
        # Poses already loaded above
        render_poses = render_poses
    elif args.render_poses == 'test':
        # Switch to dataset's test split
        render_poses = np.array(poses[i_test])
    elif args.render_poses == 'train':
        # ...or train split (useful for removing view-dependent effects from the trainset)
        render_poses = np.array(poses[i_train])
    else:
        # Switch to custom poses loaded from file
        render_poses, _ = load_blender_poses(args.render_poses)

    # Create log dir, copy the config file, initialize Tensorboard
    print(f"Experiment name: {args.expname}")
    experiment_dir = Path(args.basedir) / args.expname
    experiment_dir.mkdir(parents=True, exist_ok=True)
    with open(experiment_dir / "args.txt", 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        with open(experiment_dir / "config.txt", 'w') as file:
            with open(args.config, 'r') as original_config_file:
                file.write(original_config_file.read())
    tensorboard_writer = SummaryWriter(experiment_dir)

    # Create nerf model
    model_specific_module = {
        'nerf': nerf,
        'volsdf': volsdf
    }[args.model]
    render_kwargs_train, render_kwargs_test, start, optimizer = \
        model_specific_module.create_model(args)

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(args.device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')

        with torch.no_grad():
            if args.render_poses == 'test':
                images = images[i_test]
            else:
                images = None

            if args.render_poses in ('test', 'train'):
                poses_name = args.render_poses
            elif args.render_poses == 'dataset_path':
                poses_name = 'path'
            else:
                poses_name = Path(args.render_poses).with_suffix('').name

            testsavedir = experiment_dir / 'renderonly_{}_{:06d}'.format(poses_name, start)
            testsavedir.mkdir(parents=True, exist_ok=True)
            print('test poses shape', render_poses.shape)

            # Save camera parameters too
            save_blender_poses(testsavedir / "cameras.json", render_poses.cpu().numpy(), hwf)

            render_kwargs_test['network_fn'].eval()
            rgbs, _ = render_path(
                render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images,
                savedir=testsavedir, render_factor=args.render_factor,
                fixed_viewdir=args.fixed_viewdir_in_test)
            print('Done rendering', testsavedir)

            return

    # Prepare raybatch tensor if batching random rays
    batch_size = args.batch_size
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(args.device)
    poses = torch.Tensor(poses).to(args.device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(args.device)

    loss_function = loss_functions[args.loss]

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    for global_step in range(start, args.num_iterations + 1):
        # First, validate
        render_kwargs_test['network_fn'].eval()

        if global_step % args.i_weights == 0 and global_step > 0 or global_step == args.num_iterations:
            dict_to_save = {
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if 'network_fine' in render_kwargs_train:
                dict_to_save['network_fine_state_dict'] = \
                    render_kwargs_train['network_fine'].state_dict()

            path = experiment_dir / '{:06d}.tar'.format(global_step)
            torch.save(dict_to_save, path)
            print('Saved checkpoints at', path)

        if global_step % args.i_video == 0 and global_step > 0 or global_step == args.num_iterations:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            movie_prefix = '{}_spiral_{:06d}'.format(args.expname, global_step)
            imageio.mimwrite(experiment_dir / f'{movie_prefix}_rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(experiment_dir / f'{movie_prefix}_disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            del rgbs, disps # hopefully, save a tiny bit of GPU memory...

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if global_step % args.i_testset == 0 and global_step > 0 or global_step == args.num_iterations:
            testsavedir = experiment_dir / 'testset_{:06d}'.format(global_step)
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(
                    torch.Tensor(poses[i_test]).to(args.device), hwf, K, args.chunk,
                    render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if global_step % args.i_img == 0 and global_step > 0 or global_step == args.num_iterations:
            # Log a rendered validation view to Tensorboard
            val_image_indices = [i_val[0], i_val[-1]]
            targets = images[val_image_indices] # N x H x W x C, np.float64, 0..1
            targets = torch.Tensor(targets).float()
            current_val_poses = poses[val_image_indices, :3,:4]
            with torch.no_grad():
                # rgbs: N x H x W x C, np.float32, 0..1
                rgbs, disps = render_path(current_val_poses, hwf, K, args.chunk, render_kwargs_test)

                mse = loss_functions['MSE'](torch.Tensor(rgbs), targets)
                psnr = mse2psnr(mse)

            tensorboard_image = np.concatenate((rgbs, targets.cpu()), axis=2)
            tensorboard_image = tensorboard_image.reshape(-1, *tensorboard_image.shape[-2:])
            tensorboard_writer.add_image(
                'Render, validation', tensorboard_image, global_step,
                dataformats='HWC')

            tensorboard_writer.add_scalar(f"MSE, validation", mse, global_step)
            tensorboard_writer.add_scalar(f"PSNR, validation", psnr, global_step)

            del rgbs, disps # hopefully, save a tiny bit of GPU memory...

            # if args.N_importance > 0:
            #     with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
            #         tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
            #         tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
            #         tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])

        render_kwargs_test['network_fn'].train()

        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+batch_size] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += batch_size
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(args.device)
            pose = poses[img_i, :3,:4]

            if batch_size is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if global_step < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if global_step == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[batch_size], replace=False)  # (batch_size,)
                select_coords = coords[select_inds].long()  # (batch_size, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (batch_size, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (batch_size, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (batch_size, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                retraw=True, **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = loss_function(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss) if args.loss == 'MSE' else np.nan

        # Add eikonal loss
        if args.model == 'volsdf':
            assert extras['sdf_normal'].shape[-1] == 3
            eikonal_loss = ((extras['sdf_normal'].norm(2, dim=-1) - 1.0) ** 2).mean()
            loss += args.eikonal_loss_weight * eikonal_loss

        if 'rgb0' in extras:
            img_loss0 = loss_function(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0) if args.loss == 'MSE' else np.nan

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if global_step % args.i_print == 0 or global_step == args.num_iterations:
            tqdm.write(f"[TRAIN] Iter: {global_step} Loss: {loss}  PSNR: {psnr}")

        tensorboard_writer.add_scalar(f"{args.loss}, train", img_loss, global_step)
        # if args.N_importance > 0:
        #     tf.contrib.summary.scalar('psnr0', psnr0)
        if args.model == 'volsdf':
            tensorboard_writer.add_scalar(f"Eikonal loss, train", eikonal_loss, global_step)
        tensorboard_writer.add_scalar(f"Learning rate", new_lrate, global_step)
        tensorboard_writer.add_scalar(f"Step time", dt, global_step)


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
