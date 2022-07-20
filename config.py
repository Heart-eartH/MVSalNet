import os

__all__ = ["proj_root", "arg_config"]

proj_root = os.path.dirname(__file__)

rgbdtr_path='/home/zjy/data_list/rgbd_train_jw.lst'
nlprte_path='/home/zjy/data_list/nlpr_test_jw.lst'
njudte_path = '/home/Documents/data_list/njud_test_jw.lst'
lfsd_path = '/home/Downloads/titan3/LFSD'
rgbd135_path = '/home/Downloads/titan3/RGBD-135'



arg_config = {

    "model": "MVSalNet_Res50",
    "suffix": "7Datasets",
    "resume": False,
    "use_aux_loss": True,
    "save_pre": False,
    "epoch_num": 40,
    "lr": 0.005,
    "data_mode": "RGBD",
    "rgbd_data": {
        "tr_data_path": rgbdtr_path,
        "te_data_list": {
            "nlpr": nlprte_path,
        },
    },
    "print_freq": 10,
    "prefix": (".jpg", ".png"),

    "reduction": "mean",
    "optim": "sgd_trick",
    "weight_decay": 5e-4,
    "momentum": 0.9,
    "nesterov": False,
    "lr_type": "poly",
    "lr_decay": 0.9,
    "batch_size": 4,
    "num_workers": 8,
    "input_size": 320,
}
