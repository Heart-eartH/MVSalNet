import argparse
import os
import os.path as osp
from datetime import datetime
from distutils.util import strtobool
import torch
from PIL import Image
from torchvision import transforms
import network
from utils.misc import check_dir_path_valid
from torch.utils.data import DataLoader, Dataset

my_parser = argparse.ArgumentParser(
    prog="main script",
    allow_abbrev=False,
)
my_parser.add_argument("--param_path", required=True, type=str)
my_parser.add_argument("--model", required=True, type=str)
my_parser.add_argument("--testset", required=True, type=str)
my_parser.add_argument(
    "--has_masks",
    default=True,
    type=lambda x: bool(strtobool(str(x))),
)
my_parser.add_argument("--save_pre", default=True, type=lambda x: bool(strtobool(str(x))))
my_parser.add_argument("--save_path", default="pre_result", type=str)
my_parser.add_argument("--data_mode", default="RGBD", choices=["RGB", "RGBD"], type=str)
my_parser.add_argument("--use_gpu", default=True, type=lambda x: bool(strtobool(str(x))))
my_args = my_parser.parse_args()
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader

from config import arg_config
def _make_test_dataset(root, prefix=(".jpg", ".png")):
    img_path = os.path.join(root, "Image")
    depth_path = os.path.join(root, "Depth")
    img_list = [os.path.splitext(f)[0] for f in os.listdir(depth_path) if f.endswith(prefix[1])]
    return [
        (
            os.path.join(img_path, img_name + prefix[0]),
            os.path.join(depth_path, img_name + prefix[1]),
        )
        for img_name in img_list
    ]
class TestImageFolder(Dataset):
    def __init__(self, root, in_size, prefix):
        self.imgs = _make_test_dataset(root, prefix=prefix)

        self.test_img_trainsform = transforms.Compose(
            [
                transforms.Resize((in_size, in_size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        self.test_depth_transform = transforms.Compose(
            [transforms.Resize((in_size, in_size), interpolation=Image.BILINEAR), transforms.ToTensor(),]
        )

    def __getitem__(self, index):
        img_path,depth_path = self.imgs[index]

        img = Image.open(img_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")
        if img.size != depth.size:
            depth = depth.resize(img.size, resample=Image.BILINEAR)
        img_name = (img_path.split(os.sep)[-1]).split(".")[0]

        img = self.test_img_trainsform(img)
        depth = self.test_depth_transform(depth)
        return img, img_name,depth

    def __len__(self):
        return len(self.imgs)

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super(DataLoaderX, self).__iter__())


def _make_loader(dataset, shuffle=True, drop_last=False):
    return DataLoaderX(
        dataset=dataset,
        batch_size=1,
        num_workers=arg_config["num_workers"],
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
    )
def create_loader(data_path, prefix=(".jpg", ".png")):
    test_set = TestImageFolder(data_path, in_size=arg_config["input_size"], prefix=prefix)
    loader = _make_loader(test_set, shuffle=False, drop_last=False)

    return loader

class Tester:
    def __init__(self, args):
        if args.use_gpu and torch.cuda.is_available():
            self.dev = torch.device("cuda:0")
        else:
            self.dev = torch.device("cpu")

        self.to_pil = transforms.ToPILImage()
        self.data_mode = args.data_mode
        self.model_name = args.model

        self.te_data_path = args.testset
        self.image_dir = os.path.join(self.te_data_path, "Image")
        if self.data_mode == "RGBD":
            self.depth_dir = os.path.join(self.te_data_path, "Depth")
        else:
            self.depth_dir = ""

        self.has_masks = args.has_masks
        if self.has_masks:
            self.mask_dir = os.path.join(self.te_data_path, "Mask")
        else:
            self.mask_dir = ""
        check_dir_path_valid([self.te_data_path, self.image_dir, self.mask_dir])

        self.save_pre = args.save_pre
        if self.save_pre:
            self.save_path = args.save_path
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        self.net = getattr(network, self.model_name)(pretrained=False).to(self.dev).half()
        self.resume_checkpoint(load_path=args.param_path)
        self.net.eval()

        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize((320, 320), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        if self.data_mode == "RGBD":
            self.depth_transform = transforms.Compose(
                [transforms.Resize((320, 320), interpolation=Image.BILINEAR), transforms.ToTensor()]
            )
        self.te_loader= create_loader(data_path=args.testset)

    def test(self):
        for te_data in self.te_loader:
            rgb_tensor,name,depth_tensor=te_data
            with torch.no_grad():
                pred_tensor,_,_ = self.net(rgb_tensor.to(self.dev, non_blocking=True).half(), depth_tensor.to(self.dev, non_blocking=True).half())

            pred_tensor = pred_tensor.squeeze(0).cpu().detach()

            pred_pil = self.to_pil(pred_tensor.float())

            if self.save_pre:
                pred_pil.save(osp.join(self.save_path, name[0]+'.png'))



    def resume_checkpoint(self, load_path):

        if os.path.exists(load_path) and os.path.isfile(load_path):
            print(f" =>> loading checkpoint '{load_path}' <<== ")
            checkpoint = torch.load(load_path, map_location=self.dev)
            self.net.load_state_dict(checkpoint)
            print(f" ==> loaded checkpoint '{load_path}' " f"(only has the net's weight params) <<== ")
        else:
            raise Exception(f"{load_path} ERROR")


if __name__ == "__main__":

    print(f" ===========>> {datetime.now()}: 初始化开始 <<=========== ")
    init_start = datetime.now()
    tester = Tester(args=my_args)
    print(f" ==>> 初始化完毕，用时：{datetime.now() - init_start} <<== ")

    print(f" ===========>> {datetime.now()}: 开始测试 <<=========== ")
    tester.test()
    print(f" ===========>> {datetime.now()}: 结束测试 <<=========== ")