from easydict import EasyDict as edict
import numpy as np

config = edict()

# ========================================================================================
# Eduarda: modifications done by me

config.ethnicity_sanity_check="African"

config.ethnicity="All" # Asian | African | Caucasian | Indian | All

if config.ethnicity=="Asian":
    config.best_model="22788"
elif config.ethnicity=="African":
    config.best_model="27852"
elif config.ethnicity=="Caucasian":
    config.best_model="28017"
elif config.ethnicity=="Indian":
    config.best_model="21440"

config.dataset = "synthetic" # training dataset
config.batch_size = 128 # batch size per GPU
config.augmentation = "hf" # data augmentation policy -> this is irrelevant in our case
config.loss = "AdaFace" #  Option : ElasticArcFace, ArcFace, ElasticCosFace, CosFace, MLLoss, ElasticArcFacePlus, ElasticCosFacePlus, AdaFace

# type of network to train [iresnet100 | iresnet50 | iresnet34]
config.network = "iresnet50"
config.SE = False # SEModule; I didn't change this 

if config.ethnicity == "Indian":
    config.num_classes = 6997 # Indians have 6997 identities instead of 7000
else:
    config.num_classes = 7000

config.num_image = 324106 + 324202 + 274554 + 326070
config.num_epoch = 26
config.warmup_epoch = -1
config.val_targets = ["African_test","Caucasian_test","Asian_test","Indian_test", "cplfw", "calfw", "cfp_fp", "agedb_30", "lfw"] 
config.eval_step = 10000
def lr_step_func(epoch):
    return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
        [m for m in [8, 14,20,25] if m - 1 <= epoch])  # [m for m in [8, 14,20,25] if m - 1 <= epoch])

config.lr_func = lr_step_func


if config.dataset == "balanced":    
    config.dataset_type = 'Balanced'
    config.rec = "/nas-ctm01/datasets/public/BIOMETRICS/race_per_7000_aligned" # directory for BalancedFace
    # output for KD
    #config.output = '/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/output/' +config.network+ '/' +config.loss+ '/' +config.dataset_type+'/'
    config.output = '/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/output/' +config.loss+ '/' + config.ethnicity + '/'
    #config.output = '/nas-ctm01/datasets/public/BIOMETRICS/ResNet100-Elastic-Balanced'
    config.to_sample = 28000 - 3
    # output for eval
    config.output = '/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/output/KD/' +config.dataset_type+ '/' +config.network+ '/naive/_' +str(config.to_sample)+ '/'
elif config.dataset == "synthetic":
    config.rec=""
    config.dataset_type = "Synthetic"
    # output for KD
    config.output = '/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/output/' +config.network+ '/' +config.loss+ '/' +config.dataset_type+'/'
    #config.output = '/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/output/' +config.loss+ '/' +config.dataset+config.ethnicity +'/'

    config.from_file = '/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/dataset_sampled/synthetic.txt'
    config.to_sample = 27099
    # output for eval
    #config.output = '/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/output/KD/' +config.dataset_type+ '/' +config.network+ '/naive/_' +str(config.to_sample)+ '/'
elif config.dataset == "syntheticMix":
    config.rec=""
    config.dataset_type = "SyntheticMix"
    # output for KD
    #config.output = '/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/output/' +config.network+ '/' +config.loss+ '/' +config.dataset_type+'/'
    #config.output = '/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/output/' +config.loss+ '/'+config.dataset+config.ethnicity +'/'

    config.from_file = '/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/dataset_sampled/syntheticMix.txt'
    config.to_sample = 22094
    # output for eval
    config.output = '/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/output/KD/' +config.dataset_type+ '/' +config.network+ '/naive/_' +str(config.to_sample)+ '/'
elif config.dataset == "syntheticMixEqualIDs":
    config.rec=""
    config.dataset_type = "SyntheticMixEqualIDs"
    # output for KD
    config.output = '/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/output/' +config.network+ '/' +config.loss+ '/' +config.dataset_type+'/'
    #config.output = '/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/output/' +config.dataset+config.ethnicity +'/'

    config.from_file = '/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/dataset_sampled/mix_equal_ids.txt'
    config.to_sample = 28000
    # output for eval
    #config.output = '/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/output/KD/' +config.dataset_type+ '/' +config.network+ '/naive/_' +str(config.to_sample)+ '/'
elif config.dataset == "syntheticMixEqualImg":
    config.rec=""
    config.dataset_type = "SyntheticMixEqualImg"
    # output for KD
    config.output = '/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/output/' +config.network+ '/' +config.loss+ '/' +config.dataset_type+'/'
    #config.output = '/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/output/' +config.dataset+config.ethnicity +'/'

    config.from_file = '/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/dataset_sampled/mix_equal_img.txt'
    config.to_sample = 23094
    # output for eval
    #config.output = '/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/output/KD/' +config.dataset_type+ '/' +config.network+ '/naive/_' +str(config.to_sample)+ '/'

if config.dataset == "syntheticMixEqualImg" or config.dataset == "syntheticMixEqualIDs":
    #print('Im in' + config.dataset)
    config.out_fts="/nas-ctm01/datasets/public/BIOMETRICS/ResNet100-Elastic-Balanced/embeddings_on_" + config.dataset_type + "/"
else:
    #print('Else:'+config.dataset)
    config.out_fts="/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/models/teacher/ResNet100-AdaFace-Balanced/embeddings_on_" + config.dataset_type + "/"
    #config.out_fts="/nas-ctm01/datasets/public/BIOMETRICS/ResNet100-AdaFace-Balanced/embeddings_on_" + config.dataset_type + "/"

if config.ethnicity!="All":
    config.backbone_pth="output/"+config.ethnicity+"/"+config.best_model+"backbone.pth"
elif config.loss == 'ElasticArcFace':
    config.backbone_pth="models/teacher/ResNet100-Elastic-Balanced/78048backbone.pth"
elif config.loss == 'AdaFace':
    config.backbone_pth="models/teacher/ResNet100-AdaFace-Balanced/68292backbone.pth"

config.cont_factor=4

config.cont_path_PCA="/nas-ctm01/datasets/public/BIOMETRICS/BalancedFace_embeddings/contracted/PCA"
config.cont_path_baseline="/nas-ctm01/datasets/public/BIOMETRICS/BalancedFace_embeddings/contracted/baseline"

#config.to_sample=22094 #synthetic:27099 mix:22094 7000 | 28000 - 3: used for KD train + for training the baseline with 7k samples balanced across the 4 races
config.KD_method="naive" # baseline | PCA | naive | triplet_loss | arc
config.cont_setting="b_vs_a" # b_vs_a | b_vs_c | b_vs_i

config.loss_lambda=10000

config.triplet_margin=0.2 
config.phi=1
config.consider_id=True
config.consider_race=False
config.fc_lbd=1

config.out_fc='/nas-ctm01/datasets/public/BIOMETRICS/KD_FR/output/naive/'+ config.network+ '/'+ config.loss+ '/' +config.dataset_type+'/'
config.num_epoch_fc=10
config.lr_fc=1

config.arc_method="selection" # selection | fusion | positioning | encoding
config.out_arc_adapter="/nas-ctm01/homes/mecaldeira/output/ArcAdapter/lr_"+str(config.lr_fc)+"/"+config.arc_method
config.out_arc_fts="/nas-ctm01/datasets/public/BIOMETRICS/BalancedFace_embeddings/fully_connected/ArcFace/lr_"+str(config.lr_fc)+"/"+config.arc_method
if config.KD_method!="naive":
    a=1
    #if config.KD_method=="triplet_loss":
    #    config.output_KD="/nas-ctm01/homes/mecaldeira/output/KD/no_CEL/triplet_loss/lr_"+str(config.lr_fc)+"/"+config.arc_method+"/"
    #    if config.consider_id:
    #        config.output_KD=config.output_KD+"phi_"+str(config.phi) + "_"
    #    if config.consider_race:
    #        config.output_KD=config.output_KD+"tllbd_"+str(config.fc_lbd) + "_"
    #    config.output_KD=config.output_KD + "/lbd_" + str(config.loss_lambda) + "/" + config.network + "/_" + str(config.to_sample)
    #elif config.KD_method=="arc":
    #     config.output_KD="/nas-ctm01/homes/mecaldeira/output/KD/no_CEL/arc/lr_"+str(config.lr_fc)+"/"+config.arc_method + "/" + config.network + "/_" + str(config.to_sample)
    #else:
    #    config.output_KD='/nas-ctm01/homes/mecaldeira/output/KD/no_CEL/'+ config.dataset_type + "/" + config.network + "/" + config.KD_method + "/" + config.cont_setting + "/_" + str(config.to_sample)
else:
    config.output_KD='/nas-ctm01/datasets/public/BIOMETRICS/KD_FR/output/KD/'+ config.dataset_type + "/" + config.network + "/" +config.loss+ '/' + config.KD_method + "/_" + str(config.to_sample)


if config.dataset == "syntheticMixEqualImg" or config.dataset == "syntheticMixEqualIDs":
    #print('Im in' + config.dataset)
    config.path_teacher_fts="/nas-ctm01/datasets/public/BIOMETRICS/ResNet100-Elastic-Balanced/embeddings_on_" + config.dataset_type + "/"
else:
    #print('Else:'+config.dataset)
    config.path_teacher_fts="/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/models/teacher/ResNet100-AdaFace-Balanced/embeddings_on_" + config.dataset_type + "/"
    #config.path_teacher_fts="/nas-ctm01/datasets/public/BIOMETRICS/ResNet100-AdaFace-Balanced/embeddings_on_" + config.dataset_type + "/"

config.off_african=0
if config.cont_setting=="b_vs_a":
    config.off_asian=180
    config.off_caucasian=90
    config.off_indian=270
elif config.cont_setting=="b_vs_c":
    config.off_asian=90
    config.off_caucasian=180
    config.off_indian=270
else:
    config.off_asian=270
    config.off_caucasian=90
    config.off_indian=180

config.out_fc_fts="/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/models/teacher/fully_connected/"+config.loss+'/'+config.dataset_type+"/"
if config.consider_id:
    if config.consider_race:
        config.out_fc_fts=config.out_fc_fts+"lr_"+str(config.lr_fc)+"/"+config.arc_method+"/id_race/phi_"+str(config.phi)+"/lbd_"+str(config.fc_lbd)
        config.out_fc=config.out_fc+"lr_"+str(config.lr_fc)+"/"+config.arc_method+"/id_race/phi_"+str(config.phi)+"/lbd_"+str(config.fc_lbd)
    else:
        config.out_fc_fts=config.out_fc_fts+"lr_"+str(config.lr_fc)+"/"+config.arc_method+"/id/phi_"+str(config.phi)
        config.out_fc=config.out_fc+"lr_"+str(config.lr_fc)+"/"+config.arc_method+"/id/phi_"+str(config.phi)
elif config.consider_race:
    config.out_fc_fts=config.out_fc_fts+"lr_"+str(config.lr_fc)+"/"+config.arc_method+"/race/lbd_"+str(config.fc_lbd)
    config.out_fc=config.out_fc+"lr_"+str(config.lr_fc)+"/"+config.arc_method+"/race/lbd_"+str(config.fc_lbd)

if config.KD_method=="naive":
    config.path_teacher_fts=config.path_teacher_fts#+"/original"
elif config.KD_method=="triplet_loss":
    config.path_teacher_fts=config.out_fc_fts
elif config.KD_method=="arc":
    config.path_teacher_fts=config.out_arc_fts
else:
    config.path_teacher_fts=config.path_teacher_fts+"/contracted/"+config.KD_method 

config.is_teacher_baseline=False
config.out_teacher_baseline='/nas-ctm01/homes/icolakovic/RaceBalancedFaceRecognition-master/output/teacher_baseline/'+config.ethnicity

if config.is_teacher_baseline:
    config.rec="/nas-ctm01/datasets/public/BIOMETRICS/BalancedFace_embeddings/baseline/teacher_"+config.ethnicity
    config.output=config.out_teacher_baseline

config.out_adaptor_test="/nas-ctm01/homes/mecaldeira/output/adaptor_test/"
if config.is_teacher_baseline:
    if config.ethnicity=="Indian":
        config.to_sample=6998
    else:
        config.to_sample=6999

    config.out_adaptor_test=config.out_adaptor_test+"baseline/"+config.KD_method
else:
    config.out_adaptor_test=config.out_adaptor_test+"debias_model/"+config.KD_method

if config.KD_method=="triplet_loss":
    config.out_adaptor_test=config.out_adaptor_test+"/phi_"+str(config.phi)

# ========================================================================================

config.momentum = 0.9
config.embedding_size = 512 # embedding size of model
config.weight_decay = 5e-4
config.lr = 0.1
# train model output folder
config.global_step=0 # step to resume
config.s=64.0
config.m=0.50
config.std=0.05

if (config.loss=="ElasticArcFacePlus"):
    config.s = 64.0
    config.m = 0.50
    config.std = 0.0175
elif (config.loss=="ElasticArcFace"):
    config.s = 64.0
    config.m = 0.50
    config.std = 0.05
if (config.loss=="ElasticCosFacePlus"):
    config.s = 64.0
    config.m = 0.35
    config.std = 0.02
elif (config.loss=="ElasticCosFace"):
    config.s = 64.0
    config.m = 0.35
    config.std = 0.05
elif (config.loss=="CurricularFace"):
    config.s = 64.0
    config.m = 0.50
elif (config.loss=="RaceFace"):
    config.s = 64.0
    config.m = 0.50
elif (config.loss=="AdaFace"):
    config.s = 64.0
    config.m = 0.40
