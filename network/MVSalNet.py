import torch
import torch.nn as nn
from module.BaseBlocks import BasicConv2d
from utils.tensor_ops import cus_sample, upsample_add
from backbone.ResNet import Backbone_ResNet50_in3
from module.MyModules import transDDPM
from py3d_rotate import my_rotate,rotate_back
from MVSal_trans import SwinTransformer
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    NormWeightedCompositor,)

device = torch.device("cuda:0")

def to_nor(tensor, mean, std):
    tensor = tensor.clone()
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]
    tensor.sub_(mean).div_(std)
    return tensor
def initNetParams(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

class MVSalNet_Res50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample
        self.encoder2, self.encoder4, self.encoder8, self.encoder16, self.encoder32 = Backbone_ResNet50_in3(
            pretrained=pretrained)

        self.encoder2_R, self.encoder4_R, self.encoder8_R, self.encoder16_R, self.encoder32_R = Backbone_ResNet50_in3(
            pretrained=pretrained)

        self.encoder2_L, self.encoder4_L, self.encoder8_L, self.encoder16_L, self.encoder32_L = Backbone_ResNet50_in3(
            pretrained=pretrained)


        self.trans32 = nn.Conv2d(2048, 64, kernel_size=1)
        self.trans16 = nn.Conv2d(1024, 64, kernel_size=1)
        self.trans8 = nn.Conv2d(512, 64, kernel_size=1)
        self.trans4 = nn.Conv2d(256, 64, kernel_size=1)
        self.trans2 = nn.Conv2d(64, 64, kernel_size=1)


        self.transtrans32 = SwinTransformer(img_size=10, patch_size=1,window_size=5, in_chans=2048,)
        self.transtrans16 = SwinTransformer(img_size=20, patch_size=1,window_size=10, in_chans=1024,)
        self.transtrans8 = SwinTransformer(img_size=40, patch_size=1,window_size=20, in_chans=512,)

        self.upconv32 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)


        self.selfdc_32 = transDDPM(144, 64)
        self.selfdc_16 = transDDPM(144, 64)
        self.selfdc_8 = transDDPM(144, 64)
        initNetParams(self.selfdc_32)
        initNetParams(self.selfdc_16)
        initNetParams(self.selfdc_8)

        self.classifier = nn.Conv2d(32, 1, 1)

        self.trans32_L = nn.Conv2d(2048, 64, kernel_size=1)
        self.trans16_L = nn.Conv2d(1024, 64, kernel_size=1)
        self.trans8_L = nn.Conv2d(512, 64, kernel_size=1)
        self.trans4_L = nn.Conv2d(256, 64, kernel_size=1)
        self.trans2_L = nn.Conv2d(64, 64, kernel_size=1)

        self.transtrans32_L = SwinTransformer(img_size=10, patch_size=1,window_size=5, in_chans=2048,)
        self.transtrans16_L = SwinTransformer(img_size=20, patch_size=1,window_size=10, in_chans=1024,)
        self.transtrans8_L = SwinTransformer(img_size=40, patch_size=1,window_size=20, in_chans=512,)

        self.upconv32_L = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv16_L = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8_L = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4_L = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2_L = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1_L = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.selfdc_32_L = transDDPM(144, 64)
        self.selfdc_16_L = transDDPM(144, 64)
        self.selfdc_8_L = transDDPM(144, 64)
        initNetParams(self.selfdc_32_L)
        initNetParams(self.selfdc_16_L)
        initNetParams(self.selfdc_8_L)

        self.classifier_L = nn.Conv2d(32, 1, 1)

        self.trans32_R = nn.Conv2d(2048, 64, kernel_size=1)
        self.trans16_R = nn.Conv2d(1024, 64, kernel_size=1)
        self.trans8_R = nn.Conv2d(512, 64, kernel_size=1)
        self.trans4_R = nn.Conv2d(256, 64, kernel_size=1)
        self.trans2_R = nn.Conv2d(64, 64, kernel_size=1)

        self.transtrans32_R = SwinTransformer(img_size=10, patch_size=1,window_size=5, in_chans=2048,)
        self.transtrans16_R = SwinTransformer(img_size=20, patch_size=1,window_size=10, in_chans=1024,)
        self.transtrans8_R = SwinTransformer(img_size=40, patch_size=1,window_size=20, in_chans=512,)

        self.upconv32_R = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv16_R = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8_R = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4_R = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2_R = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1_R = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.selfdc_32_R = transDDPM(144, 64)
        self.selfdc_16_R = transDDPM(144, 64)
        self.selfdc_8_R = transDDPM(144, 64)
        initNetParams(self.selfdc_32_R)
        initNetParams(self.selfdc_16_R)
        initNetParams(self.selfdc_8_R)

        self.classifier_R = nn.Conv2d(32, 1, 1)

        self.Conv1_f_1=BasicConv2d(1,32,kernel_size=3, stride=1, padding=1)
        self.Conv1_f_2=BasicConv2d(32,32,kernel_size=3, stride=1, padding=1)
        self.Conv1_f_3=BasicConv2d(32,1,kernel_size=1, stride=1)

        self.Conv2_f_1=BasicConv2d(1,32,kernel_size=3, stride=1, padding=1)
        self.Conv2_f_2=BasicConv2d(32,32,kernel_size=3, stride=1, padding=1)
        self.Conv2_f_3=BasicConv2d(32,1,kernel_size=1, stride=1)

        self.Conv3_f_1=BasicConv2d(2,32,kernel_size=3, stride=1, padding=1)
        self.Conv3_f_2=BasicConv2d(32,32,kernel_size=3, stride=1, padding=1)
        self.Conv3_f_3=nn.Conv2d(32, 1, 1)

        self.R1, self.T1 = look_at_view_transform(1000, 0, -30)
        self.R2, self.T2 = look_at_view_transform(1000, 0, 30)
        self.cameras1 = FoVOrthographicCameras(
            device=device,
            R=self.R1,
            T=self.T1,
            max_y=160,
            min_y=-160,
            max_x=160,
            min_x=-160, )
        self.cameras2 = FoVOrthographicCameras(
            device=device,
            R=self.R2,
            T=self.T2,
            max_y=160,
            min_y=-160,
            max_x=160,
            min_x=-160, )

        self.raster_settings = PointsRasterizationSettings(
            image_size=(320, 320),
            points_per_pixel=5)
        self.rasterizer1 = PointsRasterizer(cameras=self.cameras1, raster_settings=self.raster_settings)
        self.renderer1 = PointsRenderer(
            rasterizer=self.rasterizer1,
            compositor=NormWeightedCompositor())
        self.rasterizer2 = PointsRasterizer(cameras=self.cameras2, raster_settings=self.raster_settings)
        self.renderer2 = PointsRenderer(
            rasterizer=self.rasterizer2,
            compositor=NormWeightedCompositor())

    def forward(self, in_data, in_depth):
        xuanzhuanjiao=30
        in_data_R,depth_R=my_rotate(in_data,in_depth*255,xuanzhuanjiao,self.renderer1)
        in_data_L,depth_L=my_rotate(in_data,in_depth*255,-xuanzhuanjiao,self.renderer2)

        in_data_R = in_data_R.permute(0,3,1,2)
        in_data_L = in_data_L.permute(0,3,1,2)

        in_data_R = to_nor(in_data_R,[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        in_data = to_nor(in_data,[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        in_data_L = to_nor(in_data_L,[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        in_data_2 = self.encoder2(in_data)
        in_data_2_R = self.encoder2_R(in_data_R)
        in_data_2_L = self.encoder2_L(in_data_L)
        del in_data
        del in_data_R
        del in_data_L

        in_data_4 = self.encoder4(in_data_2)
        in_data_4_R = self.encoder4_R(in_data_2_R)
        in_data_4_L = self.encoder4_L(in_data_2_L)

        in_data_8 = self.encoder8(in_data_4)
        in_data_8_R = self.encoder8_R(in_data_4_R)
        in_data_8_L = self.encoder8_L(in_data_4_L)

        in_data_16 = self.encoder16(in_data_8)
        in_data_16_R = self.encoder16_R(in_data_8_R)
        in_data_16_L = self.encoder16_L(in_data_8_L)

        in_data_32 = self.encoder32(in_data_16)
        in_data_32_R = self.encoder32_R(in_data_16_R)
        in_data_32_L = self.encoder32_L(in_data_16_L)

        in_data_8_aux = self.transtrans8(in_data_8)
        in_data_16_aux = self.transtrans16(in_data_16)
        in_data_32_aux = self.transtrans32(in_data_32)

        in_data_2 = self.trans2(in_data_2)
        in_data_4 = self.trans4(in_data_4)
        in_data_8 = self.trans8(in_data_8)
        in_data_16 = self.trans16(in_data_16)
        in_data_32 = self.trans32(in_data_32)

        out_data_32 = self.upconv32(in_data_32)
        del in_data_32
        out_data_16 = self.upsample_add(self.selfdc_32(out_data_32, in_data_32_aux), in_data_16)
        del out_data_32, in_data_32_aux, in_data_16
        out_data_16 = self.upconv16(out_data_16)
        out_data_8 = self.upsample_add(self.selfdc_16(out_data_16, in_data_16_aux), in_data_8)
        del out_data_16, in_data_16_aux, in_data_8
        out_data_8 = self.upconv8(out_data_8)
        out_data_4 = self.upsample_add(self.selfdc_8(out_data_8, in_data_8_aux), in_data_4)
        del out_data_8, in_data_8_aux, in_data_4
        out_data_4 = self.upconv4(out_data_4)
        out_data_2 = self.upsample_add(out_data_4, in_data_2)
        del out_data_4, in_data_2
        out_data_2 = self.upconv2(out_data_2)
        out_data_1 = self.upconv1(self.upsample(out_data_2, scale_factor=2))
        del out_data_2
        out_data = self.classifier(out_data_1)
        del out_data_1

        in_data_8_aux_R = self.transtrans8_R(in_data_8_R)
        in_data_16_aux_R = self.transtrans16_R(in_data_16_R)
        in_data_32_aux_R = self.transtrans32_R(in_data_32_R)

        in_data_2_R = self.trans2_R(in_data_2_R)
        in_data_4_R = self.trans4_R(in_data_4_R)
        in_data_8_R = self.trans8_R(in_data_8_R)
        in_data_16_R = self.trans16_R(in_data_16_R)
        in_data_32_R = self.trans32_R(in_data_32_R)

        out_data_32_R = self.upconv32_R(in_data_32_R)
        del in_data_32_R
        out_data_16_R = self.upsample_add(self.selfdc_32_R(out_data_32_R, in_data_32_aux_R), in_data_16_R)
        del out_data_32_R, in_data_32_aux_R, in_data_16_R
        out_data_16_R = self.upconv16_R(out_data_16_R)
        out_data_8_R = self.upsample_add(self.selfdc_16_R(out_data_16_R, in_data_16_aux_R), in_data_8_R)
        del out_data_16_R, in_data_16_aux_R, in_data_8_R
        out_data_8_R = self.upconv8_R(out_data_8_R)
        out_data_4_R = self.upsample_add(self.selfdc_8_R(out_data_8_R, in_data_8_aux_R), in_data_4_R)
        del out_data_8_R, in_data_8_aux_R, in_data_4_R
        out_data_4_R = self.upconv4_R(out_data_4_R)
        out_data_2_R = self.upsample_add(out_data_4_R, in_data_2_R)
        del out_data_4_R, in_data_2_R
        out_data_2_R = self.upconv2_R(out_data_2_R)
        out_data_1_R = self.upconv1_R(self.upsample(out_data_2_R, scale_factor=2))
        del out_data_2_R
        out_data_R = self.classifier_R(out_data_1_R)
        del out_data_1_R

        in_data_8_aux_L = self.transtrans8_L(in_data_8_L)
        in_data_16_aux_L = self.transtrans16_L(in_data_16_L)
        in_data_32_aux_L = self.transtrans32_L(in_data_32_L)

        in_data_2_L = self.trans2_L(in_data_2_L)
        in_data_4_L = self.trans4_L(in_data_4_L)
        in_data_8_L = self.trans8_L(in_data_8_L)
        in_data_16_L = self.trans16_L(in_data_16_L)
        in_data_32_L = self.trans32_L(in_data_32_L)


        out_data_32_L = self.upconv32_L(in_data_32_L)
        del in_data_32_L
        out_data_16_L = self.upsample_add(self.selfdc_32_L(out_data_32_L, in_data_32_aux_L), in_data_16_L)
        del out_data_32_L, in_data_32_aux_L, in_data_16_L
        out_data_16_L = self.upconv16_L(out_data_16_L)
        out_data_8_L = self.upsample_add(self.selfdc_16_L(out_data_16_L, in_data_16_aux_L), in_data_8_L)
        del out_data_16_L, in_data_16_aux_L, in_data_8_L
        out_data_8_L = self.upconv8_L(out_data_8_L)
        out_data_4_L = self.upsample_add(self.selfdc_8_L(out_data_8_L, in_data_8_aux_L), in_data_4_L)
        del out_data_8_L, in_data_8_aux_L, in_data_4_L
        out_data_4_L = self.upconv4_L(out_data_4_L)
        out_data_2_L = self.upsample_add(out_data_4_L, in_data_2_L)
        del out_data_4_L, in_data_2_L
        out_data_2_L = self.upconv2_L(out_data_2_L)
        out_data_1_L = self.upconv1_L(self.upsample(out_data_2_L, scale_factor=2))
        del out_data_2_L
        out_data_L = self.classifier_L(out_data_1_L)
        del out_data_1_L

        out_data_L=rotate_back(out_data_L-torch.min(out_data_L),depth_L,self.renderer1)+torch.min(out_data_L)
        out_data_R=rotate_back(out_data_R-torch.min(out_data_R),depth_R,self.renderer2)+torch.min(out_data_R)
        out_data_R = out_data_R.permute(0,3,1,2)
        out_data_L = out_data_L.permute(0,3,1,2)

        out_data_RL=out_data_R+out_data_L
        del out_data_R,out_data_L
        out_data_E1_mul=out_data*(out_data_RL.sigmoid())
        out_data_E1_add=out_data+out_data_RL

        out_data_E2_mul=self.Conv1_f_1(out_data_E1_mul)
        del out_data_E1_mul
        out_data_E3_mul=self.Conv1_f_2(out_data_E2_mul)
        del out_data_E2_mul
        out_data_E4_mul=self.Conv1_f_3(out_data_E3_mul)
        del out_data_E3_mul
        out_data_E2_add=self.Conv2_f_1(out_data_E1_add)
        del out_data_E1_add
        out_data_E3_add=self.Conv2_f_2(out_data_E2_add)
        del out_data_E2_add
        out_data_E4_add=self.Conv2_f_3(out_data_E3_add)
        del out_data_E3_add

        out_data_fuse_1=torch.cat((out_data_E4_add,out_data_E4_mul),1)
        del out_data_E4_add,out_data_E4_mul
        out_data_fuse_2=self.Conv3_f_1(out_data_fuse_1)
        del out_data_fuse_1
        out_data_fuse_3=self.Conv3_f_2(out_data_fuse_2)
        del out_data_fuse_2
        out_data_fuse_4=self.Conv3_f_3(out_data_fuse_3)
        del out_data_fuse_3

        return out_data_fuse_4.sigmoid(),out_data.sigmoid(),out_data_RL.sigmoid()

