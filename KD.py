import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss, MSELoss
from utils.augmentation import get_conventional_aug_policy
from torch.utils.data import DataLoader
from utils import losses
from config.config import config as cfg
from utils.dataset import MXFaceDataset,MXFaceDataset, MXSyntheticFaceDataset, MXFaceDataset_rec
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging

from backbones.iresnet import iresnet100, iresnet50, iresnet34
from backbones.mobilefacenet import MobileFaceNet

torch.backends.cudnn.benchmark = True

def main(args):
    local_rank = args.local_rank
    torch.cuda.set_device(0)
    rank = 0
    world_size = 1

    if cfg.KD_method!="naive":
        if cfg.KD_method=="triplet_loss":
            cfg.output_KD="/nas-ctm01/homes/mecaldeira/output/KD/triplet_loss/lr_"+str(cfg.lr_fc)+"/"+cfg.arc_method+"/"
            if cfg.consider_id:
                cfg.output_KD=cfg.output_KD + "phi_"+str(cfg.phi) + "_"
            if cfg.consider_race:
                cfg.output_KD=cfg.output_KD + "tllbd_"+str(cfg.fc_lbd) + "_"
            cfg.output_KD=cfg.output_KD + "/lbd_" + str(cfg.loss_lambda) + "/" + cfg.network + "/_" + str(cfg.to_sample)
        elif cfg.KD_method=="arc":
            cfg.output_KD="/nas-ctm01/homes/mecaldeira/output/KD/arc/lr_"+str(cfg.lr_fc)+"/"+cfg.arc_method + "/" + str(cfg.loss_lambda) + "/" + cfg.network + "/_" + str(cfg.to_sample)
        else:
            cfg.output_KD='/nas-ctm01/homes/mecaldeira/output/KD/lbd_'+ str(cfg.loss_lambda) + "/" + cfg.network + "/" + cfg.KD_method + "/" + cfg.cont_setting + "/_" + str(cfg.to_sample)
    else:
        cfg.output_KD='/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/output/KD/'+cfg.loss+'/'+ str(cfg.dataset_type) + "/" + cfg.network + "/" + cfg.KD_method + "/_" + str(cfg.to_sample)



    # cfg.output_KD="/nas-ctm01/homes/mecaldeira/output/KD/baseline/arc/lr_"+str(cfg.lr_fc)+"/"+cfg.arc_method + "/" + cfg.network + "/_" + str(cfg.to_sample)
    # cfg.path_teacher_fts="/nas-ctm01/datasets/public/BIOMETRICS/BalancedFace_embeddings/baseline_embeddings/fully_connected/ArcFace/lr_"+str(cfg.lr_fc)+"/"+cfg.arc_method



    if not os.path.exists(cfg.output_KD) and rank == 0:
        os.makedirs(cfg.output_KD)
    else:
        time.sleep(2)

    transform = get_conventional_aug_policy(cfg.augmentation)

    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output_KD, logfile="train_KD.log")

    if cfg.dataset == 'competition':
        if transform is not None:
            trainset = MXSyntheticFaceDataset(root_dir=cfg.rec, local_rank=local_rank,from_file=cfg.from_file,transform=transform)
        else:
            trainset = MXSyntheticFaceDataset(root_dir=cfg.rec, local_rank=local_rank,from_file=cfg.from_file)
    elif cfg.dataset == 'competition_baseline':
        trainset = MXFaceDataset_rec(root_dir=cfg.rec, local_rank=local_rank, transform=transform)
    elif cfg.dataset == 'balanced':
        trainset = MXFaceDataset(root_dir=cfg.rec, ethnicity=cfg.ethnicity, local_rank=local_rank, is_train=True, to_sample=cfg.to_sample)
    elif 'synthetic' in cfg.dataset:
        trainset = MXSyntheticFaceDataset(root_dir=cfg.rec, local_rank=local_rank, from_file=cfg.from_file, transform = None, is_train=False)

    train_sampler = torch.utils.data.RandomSampler(trainset)

    train_loader = DataLoader(dataset=trainset, batch_size=cfg.batch_size,
        sampler = train_sampler ,num_workers=4, pin_memory=True, drop_last=True,prefetch_factor=2)
    
    # load model
    if cfg.network == "iresnet100":
        backbone = iresnet100(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
        # FIXME Verify dropout rate
    elif cfg.network == "iresnet50":
        backbone = iresnet50(dropout=0.4,num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    elif cfg.network == "iresnet34":
        backbone = iresnet34(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    elif cfg.network == "mobilenet":
        backbone = MobileFaceNet(embedding_size=cfg.embedding_size).to(local_rank)
    else:
        backbone = None
        logging.info("load backbone failed!")
        exit()

    if args.resume:
        try:
            backbone_pth = os.path.join(cfg.output, str(cfg.global_step) + "backbone.pth")
            backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))

            if rank == 0:
                logging.info("backbone resume loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("load backbone resume init, failed!")

    backbone.train()

    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)
    
    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=cfg.lr_func)        

    # get header => in the KD setting, the number of output features (classes) is defined as the number of sampled ids
    if cfg.loss == "ElasticArcFace":
        header = losses.ElasticArcFace(in_features=cfg.embedding_size, out_features=cfg.to_sample, s=cfg.s, m=cfg.m,std=cfg.std).to(local_rank)
    elif cfg.loss == "ElasticArcFacePlus":
        header = losses.ElasticArcFace(in_features=cfg.embedding_size, out_features=cfg.to_sample, s=cfg.s, m=cfg.m,
                                    std=cfg.std, plus=True).to(local_rank)
    elif cfg.loss == "ElasticCosFace":
        header = losses.ElasticCosFace(in_features=cfg.embedding_size, out_features=cfg.to_sample, s=cfg.s, m=cfg.m,std=cfg.std).to(local_rank)
    elif cfg.loss == "ElasticCosFacePlus":
        header = losses.ElasticCosFace(in_features=cfg.embedding_size, out_features=cfg.to_sample, s=cfg.s, m=cfg.m,
                                    std=cfg.std, plus=True).to(local_rank)
    elif cfg.loss == "ArcFace":
        header = losses.ArcFace(in_features=cfg.embedding_size, out_features=cfg.to_sample, s=cfg.s, m=cfg.m).to(local_rank)
    elif cfg.loss == "CosFace":
        header = losses.CosFace(in_features=cfg.embedding_size, out_features=cfg.to_sample, s=cfg.s, m=cfg.m).to(local_rank)
    elif cfg.loss == "CurricularFace":
        header = losses.CurricularFace(in_features=cfg.embedding_size, out_features=cfg.to_sample, s=cfg.s, m=cfg.m).to(local_rank)
    elif cfg.loss == "RaceFace":
        header = losses.RaceFace(in_features=cfg.embedding_size, out_features=cfg.to_sample, s=cfg.s, m=cfg.m).to(local_rank)
    elif cfg.loss == "AdaFace":
        header = losses.AdaFace(embedding_size=cfg.embedding_size, classnum=cfg.to_sample, s=cfg.s, m=cfg.m).to(local_rank)
    else:
        print("Header not implemented")

    if args.resume:
        try:
            header_pth = os.path.join(cfg.output, str(cfg.global_step) + "header.pth")
            header.load_state_dict(torch.load(header_pth, map_location=torch.device(local_rank)))

            if rank == 0:
                logging.info("header resume loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("header resume init, failed!")
    
    header.train()

    opt_header = torch.optim.SGD(
    params=[{'params': header.parameters()}],
    lr=cfg.lr / 512 * cfg.batch_size * world_size,
    momentum=0.9, weight_decay=cfg.weight_decay)

    scheduler_header = torch.optim.lr_scheduler.LambdaLR(
    optimizer=opt_header, lr_lambda=cfg.lr_func)

    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size / world_size * cfg.num_epoch)
    if rank == 0: logging.info("Total Step is: %d" % total_step)

    if args.resume:
        rem_steps = (total_step - cfg.global_step)
        cur_epoch = cfg.num_epoch - int(cfg.num_epoch / total_step * rem_steps)
        logging.info("resume from estimated epoch {}".format(cur_epoch))
        logging.info("remaining steps {}".format(rem_steps))
        
        start_epoch = cur_epoch
        scheduler_backbone.last_epoch = cur_epoch

        # --------- this could be solved more elegant ----------------
        opt_backbone.param_groups[0]['lr'] = scheduler_backbone.get_lr()[0]
        
        scheduler_header.last_epoch = cur_epoch
        opt_header.param_groups[0]['lr'] = scheduler_header.get_lr()[0]

        print("last learning rate: {}".format(scheduler_header.get_lr()))
        # ------------------------------------------------------------

    callback_verification = CallBackVerification(cfg.eval_step, rank, cfg.val_targets, cfg.rec)
    callback_logging = CallBackLogging(500, rank, total_step, cfg.batch_size, world_size, writer=None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output_KD)

    loss = AverageMeter()
    global_step = cfg.global_step

    criterion = CrossEntropyLoss()
    criterion_KD = MSELoss()

    logging.info("Train starting for %s samples.", cfg.ethnicity)
    for epoch in range(start_epoch, cfg.num_epoch):
        running_loss = 0
        running_loss_CEL = 0
        running_loss_lbdKD = 0
        steps = 0
        for _, (img, label, extra_img_path) in enumerate(train_loader):
            global_step += 1
            img = img.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)
            features = F.normalize(backbone(img))

            teacher_fts=np.zeros((len(label), cfg.embedding_size))
            for sample in range(len(label)):
                if cfg.KD_method != "triplet_loss" and cfg.KD_method != "naive" and cfg.KD_method != "arc":
                    extra_img_path[sample]=extra_img_path[sample].replace("Indian", "Indian/off_"+str(cfg.off_indian))
                    extra_img_path[sample]=extra_img_path[sample].replace("Caucasian", "Caucasian/off_"+str(cfg.off_caucasian))
                    extra_img_path[sample]=extra_img_path[sample].replace("Asian", "Asian/off_"+str(cfg.off_asian))
                    extra_img_path[sample]=extra_img_path[sample].replace("African", "African/off_"+str(cfg.off_african))
                teacher_fts[sample,:]=np.load(cfg.path_teacher_fts+extra_img_path[sample]+".npy").reshape(-1, cfg.embedding_size)
            
            teacher_fts=torch.from_numpy(teacher_fts).float()
            teacher_fts=teacher_fts.cuda(local_rank, non_blocking=True)

            if cfg.loss != 'AdaFace':
                thetas = header(features, label)
            else:
                norm = torch.norm(features, 2, 1, True)
                output = torch.div(features, norm)
                thetas = header(output, norm, label)

            loss_CEL = criterion(thetas, label)

            loss_KD = cfg.loss_lambda*criterion_KD(teacher_fts, features)
            loss_v = loss_CEL+loss_KD
            loss_v.backward()

            clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)

            opt_backbone.step()
            opt_header.step()
            
            opt_backbone.zero_grad()
            opt_header.zero_grad()

            loss.update(loss_v.item(), 1)
            running_loss += loss_v.item()
            running_loss_CEL += loss_CEL.item()
            running_loss_lbdKD += loss_KD.item()
            steps+=1
            callback_logging(global_step, loss, epoch)

        logging.info("Loss check: CEL -> %f, KD -> %f, total -> %f.", running_loss_CEL, running_loss_lbdKD, running_loss)

        callback_verification(global_step, backbone)
        scheduler_backbone.step()

        scheduler_header.step()

        callback_checkpoint(global_step, backbone, header)

    if cfg.KD_method!="triplet_loss" and cfg.KD_method!="arc" and cfg.KD_method!="naive":
        logging.info("KD train complete for %s scheme (%s) and %d identities.", cfg.cont_setting, cfg.KD_method, cfg.to_sample)
    else:
        logging.info("KD train complete for scheme %s and %d identities.", cfg.KD_method, cfg.to_sample)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch margin penalty loss  training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--resume', type=int, default=0, help="resume training")
    args_ = parser.parse_args()
    main(args_)