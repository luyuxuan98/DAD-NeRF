import json
import numpy as np
import pdb
import torch

from ray_utils import get_rays, get_ray_directions, get_ndc_rays


BOX_OFFSETS = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                                device='cuda')


def hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        # 位运算符，算一堆异或（三维坐标idx值）
        xor_result ^= coords[..., i]*primes[i]
    #                      00000000111111111111         和这玩意做一个&取与就是在取余，位操作比数学的方案要快
    return torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result


def get_bbox3d_for_blenderobj(camera_transforms, H, W, near=2.0, far=6.0):
    '''
    camera_angle_x = float(camera_transforms['camera_angle_x'])
    focal = 0.5*W/np.tan(0.5 * camera_angle_x)
    '''
    focal = camera_transforms['focal_len']

    # ray directions in camera coordinates
    directions = get_ray_directions(H, W, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]

    points = []

    for frame in camera_transforms["frames"]:
        c2w = torch.FloatTensor(frame["transform_matrix"])
        rays_o, rays_d = get_rays(directions, c2w)
        
        def find_min_max(pt):
            for i in range(3):
                if(min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if(max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        for i in [0, W-1, H*W-W, H*W-1]:
            min_point = rays_o[i] + near*rays_d[i]
            max_point = rays_o[i] + far*rays_d[i]
            points += [min_point, max_point]
            find_min_max(min_point)
            find_min_max(max_point)

    return (torch.tensor(min_bound)-torch.tensor([1.0,1.0,1.0]), torch.tensor(max_bound)+torch.tensor([1.0,1.0,1.0]))


def get_bbox3d_for_llff(poses, hwf, near=0.0, far=1.0):
    H, W, focal = hwf
    H, W = int(H), int(W)
    
    # ray directions in camera coordinates
    directions = get_ray_directions(H, W, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]

    points = []
    poses = torch.FloatTensor(poses)
    for pose in poses:
        rays_o, rays_d = get_rays(directions, pose)
        rays_o, rays_d = get_ndc_rays(H, W, focal, 1.0, rays_o, rays_d)

        def find_min_max(pt):
            for i in range(3):
                if(min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if(max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        for i in [0, W-1, H*W-W, H*W-1]:
            min_point = rays_o[i] + near*rays_d[i]
            max_point = rays_o[i] + far*rays_d[i]
            points += [min_point, max_point]
            find_min_max(min_point)
            find_min_max(max_point)

    return (torch.tensor(min_bound)-torch.tensor([0.1,0.1,0.0001]), torch.tensor(max_bound)+torch.tensor([0.1,0.1,0.0001]))


def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    box_min, box_max = bounding_box
    # print('bounding_box:', bounding_box)

    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        print("ALERT: some points are outside bounding box. Clipping them!")
        # pdb.set_trace()
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max-box_min)/resolution
    
    # print('xyz.shape:', xyz.shape)
    # print('xyz:', xyz)

    bottom_left_idx = torch.floor((xyz-box_min)/grid_size).int()

    # print('bottom_left_idx.shape:', bottom_left_idx.shape)
    # print('bottom_left_idx:', bottom_left_idx)
    # 转化为实际的坐标值
    voxel_min_vertex = bottom_left_idx*grid_size + box_min

    # print('voxel_min_vertex.shape:', voxel_min_vertex.shape)
    # print('voxel_min_vertex:', voxel_min_vertex)

    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0,1.0,1.0])*grid_size

    # hashed_voxel_indices = [] # B x 8 ... 000,001,010,011,100,101,110,111
    # for i in [0, 1]:
    #     for j in [0, 1]:
    #         for k in [0, 1]:
    #             vertex_idx = bottom_left_idx + torch.tensor([i,j,k])
    #             # vertex = bottom_left + torch.tensor([i,j,k])*grid_size
    #             hashed_voxel_indices.append(hash(vertex_idx, log2_hashmap_size))


    # print('BOX_OFFSETS.shape:', BOX_OFFSETS.shape)
    # print('BOX_OFFSETS:', BOX_OFFSETS)

    # 把idx的最左下坐标扩充为三维空间中确定一个立方体的8个顶点的坐标
    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
    # print('voxel_indices.shape:', voxel_indices.shape)
    # print('voxel_indices:', voxel_indices)
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    # print('hashed_voxel_indices.shape:', hashed_voxel_indices.shape)
    # print('hashed_voxel_indices:', hashed_voxel_indices)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices



if __name__=="__main__":
    with open("data/nerf_synthetic/chair/transforms_train.json", "r") as f:
        camera_transforms = json.load(f)
    
    bounding_box = get_bbox3d_for_blenderobj(camera_transforms, 800, 800)
