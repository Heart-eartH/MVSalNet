import torch

from pytorch3d.structures import Pointclouds


device = torch.device("cuda:0")

def my_rotate(in_data,in_depth,jiao,renderer):
    batch,c,h,w = in_depth.shape
    x,y = torch.meshgrid(torch.arange(w),torch.arange(h))

    in_data = in_data.permute(0,3,2,1).float()
    in_depth = in_depth.permute(0,3,2,1)
    in_depth[..., 0][in_depth[..., 0] == 0] = 1020

    verts = torch.zeros((batch,h,w,3),).to(device)

    verts[:,:,:,0]=x-w/2
    verts[:,:,:,1]=h/2-y
    verts[:,:,:,2]=-in_depth[:,:,:,0]+127.5

    verts = verts.reshape(batch,h*w,3)

    x= torch.unsqueeze(x,2)
    x = torch.unsqueeze(x,0)

    R_depth = ((x - w / 2)* torch.sin(torch.tensor(jiao * 3.14159 / 180))).to(device) \
              + (in_depth - 127.5)* torch.cos(torch.tensor(jiao * 3.14159 / 180).to(device))

    rgb = torch.cat((in_data.reshape(batch,h*w,3),R_depth.reshape(batch,h*w,1)),dim=2)

    point_cloud = Pointclouds(points=verts, features=rgb)

    images = renderer(point_cloud)

    images[..., 3][images[..., 3] == 0] = 1020

    return images[..., :3], torch.unsqueeze(images[..., 3], 3)

def rotate_back(in_data,in_depth,renderer):

    batch,c,h,w = in_data.shape

    x,y = torch.meshgrid(torch.arange(w),torch.arange(h))
    in_data = in_data.permute(0,3,2,1).float()
    in_depth = in_depth.permute(0,2,1,3)

    verts = torch.zeros((batch,h,w,3),).to(device)
    verts[:,:,:,0]=x-w/2
    verts[:,:,:,1]=h/2-y
    verts[:,:,:,2]=-in_depth[:,:,:,0]
    verts = verts.reshape(batch,h*w,3)

    rgb = in_data.reshape(batch,h*w,1)

    point_cloud = Pointclouds(points=verts, features=rgb)

    images = renderer(point_cloud)

    return images
