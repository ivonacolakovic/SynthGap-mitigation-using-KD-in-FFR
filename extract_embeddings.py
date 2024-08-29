import argparse
import logging
import os
import time
import math
import torch
import torch.nn.functional as F
import torch.utils.data.distributed
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from config.config import config as cfg
from utils.dataset import MXFaceDataset, MXFaceDataset, MXSyntheticFaceDataset, MXFaceDataset_rec
from utils.utils_logging import init_logging
from backbones.iresnet import iresnet100, iresnet50, iresnet34

torch.backends.cudnn.benchmark = True

def contract_space(features):
    cont_features=np.concatenate((features, features, features, features), axis=0)
    features=features.flatten()

    e_1=features[np.size(features)-1]
    e_2=features[np.size(features)-2]

    if e_1==0:
      if e_2==0:
        return cont_features
      elif e_2<0:
        theta=3/2*np.pi
        CIRC_SCALE=-e_2
      else:
        theta=1/2*np.pi
        CIRC_SCALE=e_2
    else:
      # theta should be a value between 0 and 2*pi rad
      theta=math.atan(e_2/e_1)
      if e_1 < 0:
        theta=theta+np.pi
      elif e_2 < 0:
        theta=theta+2*np.pi

      CIRC_SCALE=e_2/math.sin(theta)

    # determining the angle and coordinates for the contracted space
    new_theta_0=theta/cfg.cont_factor
    new_theta_90=new_theta_0+np.pi/2
    new_theta_180=new_theta_90+np.pi/2
    new_theta_270=new_theta_180+np.pi/2

    new_e1_0=math.cos(new_theta_0)*CIRC_SCALE
    new_e2_0=math.sin(new_theta_0)*CIRC_SCALE
    new_e1_90=math.cos(new_theta_90)*CIRC_SCALE
    new_e2_90=math.sin(new_theta_90)*CIRC_SCALE
    new_e1_180=math.cos(new_theta_180)*CIRC_SCALE
    new_e2_180=math.sin(new_theta_180)*CIRC_SCALE
    new_e1_270=math.cos(new_theta_270)*CIRC_SCALE
    new_e2_270=math.sin(new_theta_270)*CIRC_SCALE

    # contracting the space
    cont_features[0, np.size(features)-1]=new_e1_0
    cont_features[0, np.size(features)-2]=new_e2_0
    cont_features[1, np.size(features)-1]=new_e1_90
    cont_features[1, np.size(features)-2]=new_e2_90
    cont_features[2, np.size(features)-1]=new_e1_180
    cont_features[2, np.size(features)-2]=new_e2_180
    cont_features[3, np.size(features)-1]=new_e1_270
    cont_features[3, np.size(features)-2]=new_e2_270

    return cont_features

def save_contracted(cont_features, extra_img_path):
    offsets=[0, 90, 180, 270]

    for offs in range(len(offsets)):
        off_img_path=extra_img_path.replace(cfg.ethnicity, cfg.ethnicity+"/off_"+str(offsets[offs]))

        if not os.path.exists(cfg.cont_path_baseline+off_img_path.replace(off_img_path.split("/")[4],"")[0:-1]):
            os.makedirs(cfg.cont_path_baseline+off_img_path.replace(off_img_path.split("/")[4],"")[0:-1])
        np.save(cfg.cont_path_baseline+off_img_path, cont_features[offs,:])

def main(args):
    local_rank=args.local_rank
    torch.cuda.set_device(local_rank)
    rank=0

    # check if the path where the features will be saved exists and create it if not
    if not os.path.exists(cfg.out_fts) and rank == 0:
        os.makedirs(cfg.out_fts)
    else:
        time.sleep(2)

    # important for the outputted log file
    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output, logfile="emb_extraction.log")

    # FIXME Change here
    if cfg.dataset == 'competition':
        trainset = MXSyntheticFaceDataset(root_dir=cfg.rec, local_rank=local_rank, from_file=cfg.from_file, transform = None)
    elif cfg.dataset == 'competition_baseline' or cfg.dataset == 'emoreIresNet':
        trainset = MXFaceDataset_rec(root_dir=cfg.rec, local_rank=local_rank, transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ]))
    elif cfg.dataset=='balanced':
        trainset = MXFaceDataset(root_dir=cfg.rec, ethnicity=cfg.ethnicity, local_rank=local_rank, is_train=False, to_sample=cfg.to_sample)
    elif 'synthetic' in cfg.dataset:
        trainset = MXSyntheticFaceDataset(root_dir=cfg.rec, local_rank=local_rank, from_file=cfg.from_file, transform = None, is_train=False)

    # initilization of the sampler and loader; 'drop_last' should be set to false to extract all the embeddings
    train_sampler = torch.utils.data.RandomSampler(trainset)
    train_loader = DataLoader(dataset=trainset, batch_size=cfg.batch_size,
        sampler = train_sampler, num_workers=4, pin_memory=True, drop_last=False, prefetch_factor=2) 
    
    # load model
    if cfg.network == "iresnet100":
        backbone = iresnet100(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
        # FIXME Verify dropout rate
    elif cfg.network == "iresnet50":
        backbone = iresnet50(dropout=0.4,num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    elif cfg.network == "iresnet34":
        backbone = iresnet34(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    else:
        backbone = None
        logging.info("load backbone failed!")
        exit()

    # load the weights of the already trained model
    try:
        backbone_pth=os.path.join(cfg.backbone_pth)
        backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))

        if rank == 0:
            logging.info("backbone resume loaded successfully!")
    except (FileNotFoundError, KeyError, IndexError, RuntimeError):
        logging.info("load backbone resume init, failed!")

    # set the backbone to evaluation mode
    backbone.eval()

    for _, (img, label, extra_img_path) in enumerate(train_loader):
        # assign image and label information to cuda
        img=img.cuda(local_rank, non_blocking=True)
        label=label.cuda(local_rank, non_blocking=True)

        # extract the features from the images
        features=F.normalize(backbone(img))
        features=features.cpu().detach().numpy()

        for j in range(label.size()[0]):
            dir_to_make = ''
            # check if the path where the features will be saved exists and create it if not
            if cfg.dataset=='balanced':
                dir_to_make = cfg.out_fts+extra_img_path[j].replace(extra_img_path[j].split("/")[3],"")[0:-1]
                path_to_save = cfg.out_fts+extra_img_path[j]
            elif 'synthetic' in cfg.dataset:             
                dir_to_make = cfg.out_fts+"/".join(extra_img_path[j].split('/')[:-1])+'/'
                path_to_save = dir_to_make + extra_img_path[j].split('/')[-1]
            if not os.path.exists(dir_to_make):
                os.makedirs(dir_to_make)

            # save each embedding in an appropriate folder
            np.save(path_to_save, features[j,:])
            #print(extra_img_path[j])
            #print(dir_to_make)
            #print(path_to_save)
           
            # perform baseline contraction (no PCA) and save the correspondent features
            #cont_features=contract_space(features[j,:].reshape(-1,cfg.embedding_size))
            #save_contracted(cont_features, extra_img_path[j])

    logging.info("Features extracted successfully! Saved original and contracted baseline features!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch margin penalty loss training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    args_ = parser.parse_args()
    main(args_)
