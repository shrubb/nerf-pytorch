import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def load_blender_poses(path):
    with open(path, 'r') as f:
        metadata = json.load(f)

    H, W = 800, 800
    camera_angle_x = float(metadata['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    camera_poses = [frame['transform_matrix'] for frame in metadata['frames']]
    camera_poses = np.float32(camera_poses)

    return camera_poses, [H, W, focal]

def save_blender_poses(path, camera_poses, hwf, names=None):
    """
    Write camera transformations to disk in 'nerf-pytorch''s "blender" json format.
    """
    H, W, focal = hwf
    camera_angle_x = 2 * np.arctan(0.5 * W / focal)
    camera_angle_y = 2 * np.arctan(0.5 * H / focal)

    camera_poses = [
        np.concatenate((camera[:3, :4], np.float32([[0,0,0,1]]))) for camera in camera_poses]

    output_json = {
        "image_height": H,
        "image_width": W,
        "camera_angle_x": camera_angle_x,
        "camera_angle_y": camera_angle_y,
        "frames": [
            {
                "file_path": names[idx] if names is not None else f"./{idx:03}.png",
                "rotation": 0.012566370614359171, # don't know what this is, took it from lego
                "transform_matrix": camera.tolist()
            } for idx, camera in enumerate(camera_poses)
        ]
    }

    with open(path, 'w') as f:
        json.dump(output_json, f, indent=4)

def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()


    return imgs, poses, render_poses, [H, W, focal], i_split


