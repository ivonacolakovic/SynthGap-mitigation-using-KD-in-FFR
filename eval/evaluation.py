import logging
import os
# FIXME
os.environ["WANDB__SERVICE_WAIT"] = "300"

#import cv2
import sys
import torch
sys.path.append('/nas-ctm01/homes/icolakovic/exp1')

from utils.utils_callbacks import CallBackVerification
from utils.utils_logging import init_logging

from config.config import config as cfg

from backbones.iresnet import iresnet100, iresnet50, iresnet34
from backbones.mobilefacenet import MobileFaceNet

if __name__ == "__main__":
    gpu_id = 0
    log_root = logging.getLogger()
    init_logging(log_root, 0, cfg.output,logfile="test1.log")
    callback_verification = CallBackVerification(1, 0, cfg.val_targets, cfg.rec)
    output_folder=cfg.output
    
    weights=os.listdir(output_folder)
    for w in weights:
        print(w)
        if "backbone" in w:
            if cfg.network == "iresnet100":
                backbone = iresnet100(num_features=cfg.embedding_size).to(f"cuda:{gpu_id}")
            elif cfg.network == "iresnet50":
                backbone = iresnet50(num_features=cfg.embedding_size).to(f"cuda:{gpu_id}")
            elif cfg.network == "iresnet34":
                backbone = iresnet34(num_features=cfg.embedding_size).to(f"cuda:{gpu_id}")
            elif cfg.network == "mobilenet":
                backbone = MobileFaceNet(embedding_size=cfg.embedding_size).to(f"cuda:{gpu_id}")


            backbone.load_state_dict(torch.load(os.path.join(output_folder,w)))
            #model = torch.nn.DataParallel(backbone, device_ids=[gpu_id])
            model = backbone.cuda()
            callback_verification(int(w.split("backbone")[0]),model)

