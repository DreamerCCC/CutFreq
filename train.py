import os
from config import Config 
opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np

import utils
from dataloaders.data_rgb import get_training_data, get_validation_data
from pdb import set_trace as stx
from networks.model import LWISP
from losses import CharbonnierLoss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from msssim import MSSSIM, SSIM, LPIPS
from vgg_loss import VGG19
from tensorboardX import SummaryWriter
from pytorch_wavelets import DWTForward, DWTInverse
import dataset_utils


######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
######### Set Seeds ###########

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir   = opt.TRAINING.VAL_DIR
save_images = opt.TRAINING.SAVE_IMAGES

# Training process visualization
writer = SummaryWriter('./training_visual/LoL-CutFreq/')

def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)
    
def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3], keepdim=keepdim)

def MultiPercep(outs0, outs1):
    feats0, feats1, diffs = {}, {}, {}
    for kk in range(5):
        feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
        diffs[kk] = (feats0[kk]-feats1[kk])**2
    res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(5)]
    val = res[0]
    for l in range(1, 5):
        val += res[l]
    loss_content = val/5
    return loss_content.mean()

def frequency_distance(im1, im2, xfm):
    
    Dll = 0
    Dlh = 0
    Dhl = 0
    Dhh = 0
    
    im1_l, im1_h = xfm(im1)
    im2_l, im2_h = xfm(im2)
    
    for index in range(len(im1_h)):
        im1_h[index] = normalize_tensor(im1_h[index])
        im2_h[index] = normalize_tensor(im2_h[index])
        Dhl += torch.nn.functional.l1_loss(im2_h[index][:,:,0,:,:], im1_h[index][:,:,0,:,:], reduction = 'mean')
        Dlh += torch.nn.functional.l1_loss(im2_h[index][:,:,1,:,:], im1_h[index][:,:,1,:,:], reduction = 'mean')
        Dhh += torch.nn.functional.l1_loss(im2_h[index][:,:,2,:,:], im1_h[index][:,:,2,:,:], reduction = 'mean')
    
    Dll += torch.nn.functional.l1_loss(normalize_tensor(im2_l), normalize_tensor(im1_l))
    
    return Dll, Dhl, Dlh, Dhh

######### Model ###########
model_restoration = LWISP(instance_norm=True, instance_norm_level_1=True)
model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  print("\nLet's use", torch.cuda.device_count(), "GPUs!\n")

new_lr = opt.OPTIM.LR_INITIAL

optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)

######### Resume ###########
if opt.TRAINING.RESUME:
    path_chk_rest    = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    lr = utils.load_optim(optimizer, path_chk_rest)

    for p in optimizer.param_groups: p['lr'] = lr
    warmup = False
    new_lr = lr
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:",new_lr)
    print('------------------------------------------------------------------------------')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-start_epoch+1, eta_min=1e-6)
else:
    warmup = True

######### Scheduler ###########
if warmup:
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

if len(device_ids)>1:
    model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)

######### Loss ###########
criterion = CharbonnierLoss().cuda()

######### DataLoaders ###########
img_options_train = {'patch_size':opt.TRAINING.TRAIN_PS}

train_dataset = get_training_data(train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False)

val_dataset = get_validation_data(val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

mixup = utils.MixUp_AUG()
best_psnr = 0
best_ssim = 0
best_lpips = 0
best_ll = 0
best_hl = 0
best_lh = 0
best_hh = 0
best_epoch = 0
best_iter = 0

eval_now = len(train_loader)//4 - 1
print(f"\nEvaluation after every {eval_now} Iterations !!!\n")

vggp = VGG19(requires_grad=False).cuda()
MS_SSIM = MSSSIM()
SSIM_ = SSIM()
LPIPS_ = LPIPS()
xfm = DWTForward(J=1, mode='zero', wave='haar').cuda()
xfm1 = DWTForward(J=3, mode='zero', wave='haar').cuda()
ifm1 = DWTInverse(wave='haar', mode='zero').cuda()

for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
    #pbar = tqdm(train_loader)
     
    print("LWISP with CutFreq augmentation method in low-light enhancement - Epoch %d begins to train ..." %(epoch))
    
    #for i, data in enumerate(tqdm(train_loader), 0):    
    for i, data in enumerate(train_loader):  
        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target = data[0].cuda()
        input_ = data[1].cuda()
        
        if epoch <= 60:
            target, input_aug, mask, aug = dataset_utils.apply_augment(target, input_, xfm1, ifm1, augs=["CutFreq_StageI", "none"], mix_p=[0.5,0.5], component=-1)
            if aug == "CutFreq_StageI":
                input_ = 0.2 * input_aug + (1 - 0.2) * input_
            else:
                input_ = input_aug
        else:
            target, input_aug, mask, aug = dataset_utils.apply_augment(target, input_, xfm1, ifm1, augs=["CutFreq_StageII", "CutFreq_StageI", "none"], mix_p=[0.4,0.4,0.2], component=-1)
            if aug == "CutFreq_StageI":
                input_ = 0.2 * input_aug + (1 - 0.2) * input_
            else:
                input_ = input_aug

        restored = model_restoration(input_)
        restored = torch.clamp(restored,0,1)  
        
        chard_loss = criterion(restored, target)
        enhanced_vgg = vggp(restored)
        target_vgg = vggp(target)
        loss_content = MultiPercep(enhanced_vgg, target_vgg)
        loss_ssim = MS_SSIM(restored, target)
        
        loss = chard_loss + loss_content * opt.TRAINING.ALPHA + (1 - loss_ssim) * 0.4 * opt.TRAINING.BETA
        loss.backward()
        
        optimizer.step()
        epoch_loss +=loss.item()
        
        #### Evaluation ####
        if i%eval_now==0 and i>0:
            if save_images:
                utils.mkdir(result_dir + '%d/%d'%(epoch,i))
            model_restoration.eval()
            Dis_LL = 0
            Dis_HL = 0
            Dis_LH = 0
            Dis_HH = 0
            with torch.no_grad():
                psnr_val_rgb = []
                ssim_eval = []
                lpips_eval = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    filenames = data_val[2]

                    restored = model_restoration(input_)
                    restored = torch.clamp(restored,0,1)
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, 1.))
                    ssim_eval.append(SSIM_(target, restored))
                    lpips_eval.append(LPIPS_(target, restored))
                    Dll_J3, Dhl_J3, Dlh_J3, Dhh_J3 = frequency_distance(target, restored, xfm)
                    Dis_LL += Dll_J3
                    Dis_HL += Dhl_J3
                    Dis_LH += Dlh_J3
                    Dis_HH += Dhh_J3
                    
                    if save_images:
                        target = target.permute(0, 2, 3, 1).cpu().detach().numpy()
                        input_ = input_.permute(0, 2, 3, 1).cpu().detach().numpy()
                        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
                        
                        for batch in range(input_.shape[0]):
                            temp = np.concatenate((input_[batch]*255, restored[batch]*255, target[batch]*255),axis=1)
                            utils.save_img(os.path.join(result_dir, str(epoch), str(i), filenames[batch][:-4] +'.jpg'),temp.astype(np.uint8))

                psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
                ssim_eval = sum(ssim_eval) / len(ssim_eval)
                lpips_eval = sum(lpips_eval) / len(lpips_eval)
                Dis_LL = Dis_LL / len(val_loader)
                Dis_HL = Dis_HL / len(val_loader)
                Dis_LH = Dis_LH / len(val_loader)
                Dis_HH = Dis_HH / len(val_loader)
                
                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_ssim = ssim_eval
                    best_lpips = lpips_eval
                    best_ll = Dis_LL
                    best_hl = Dis_HL
                    best_lh = Dis_LH
                    best_hh = Dis_HH
                    best_epoch = epoch
                    best_iter = i 
                    torch.save({'epoch': epoch, 
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))

                print("[Ep %d it %d\t PSNR: %.4f SSIM: %.4f LPIPS: %.4f\t] ----  [best_SIDD %.4f best_LPIPS %.4f best_PSNR %.4f, best_LL %.4f, best_HL %.4f, best_LH %.4f, best_HH %.4f] " % (epoch, i, psnr_val_rgb, ssim_eval, lpips_eval, best_ssim, best_lpips, best_psnr, best_ll, best_hl, best_lh, best_hh))
            
            model_restoration.train()

    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    writer.add_scalar('PSNR', psnr_val_rgb, epoch)
    writer.add_scalar('SSIM', ssim_eval, epoch)
    writer.add_scalar('LPIPS', lpips_eval, epoch)
    writer.add_scalar('Dis_LL', Dis_LL, epoch)
    writer.add_scalar('Dis_HL', Dis_HL, epoch)
    writer.add_scalar('Dis_LH', Dis_LH, epoch)
    writer.add_scalar('Dis_HH', Dis_HH, epoch)
    writer.add_scalar('Learning_rate', scheduler.get_lr()[0], epoch)
    writer.add_scalar('Loss', epoch_loss, epoch)
    
    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,f"model_epoch_{epoch}.pth")) 

