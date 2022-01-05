import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data.distributed import DistributedSampler

import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage 

import random
import numpy as np
from PIL import Image
import inception_v3 
from slic_loader import McDataset

def main():
    netT = inception_v3.inception_v3(pretrained=False)
    netT.load_state_dict(torch.load('/mnt/blob/ex_git/sup_pxiel/PRETRAIN/inception_v3_google-1a9a5a14.pth'))
    netT.eval()
    netT.cuda()
    
    mean_arr = (0.5,0.5,0.5)
    stddev_arr = (0.5,0.5,0.5)
    
    slic_dir = '/mnt/blob/ex_git/sup_pxiel/DATASET/slic/4000_20'
    print (slic_dir)
    test_dataset = McDataset(
        '/mnt/blob/testset/resize_val',
        slic_dir,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_arr, stddev_arr)
        ]) )
    
    test_loader = DataLoader(test_dataset, batch_size=16,shuffle=False)
    root = os.path.join('./OUTPUT','Test')
    L2 = torch.nn.MSELoss(reduce=False).cuda()

    Iter = 10
    print ("Iter {0}".format(Iter))
    max_epsilon = 4

    eps = max_epsilon*np.sqrt(299*299*3)
    temp_eps = eps  / Iter

    Baccu = []
    for i in range(1):
        temp_accu = AverageMeter()
        Baccu.append(temp_accu)
    Zaccu = []
    for i in range(1):
        temp_accu = AverageMeter()
        Zaccu.append(temp_accu)
    lossX = AverageMeter()
    
    slic_size = 4
    print ("Slic {0}".format(slic_size))
    
    for idx, data in enumerate(test_loader):
        iter_start_time = time.time()
        input_A = data['A']
        input_A = input_A.cuda()
        real_A = Variable(input_A,requires_grad = False)
        
        image_names = data['name']
        image_index = data['index']
        
        #########ATTNEION
        temp_img = 0
        with torch.no_grad():
                feature_B,_,logist_B = netT(real_A)
                _,pre_label = torch.max(logist_B,1)
                real_feature_maps = feature_B
                params = list(netT.parameters())
                weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
                new_weight = []
                for num in range(real_A.size(0)):
                    new_weight.append(weight_softmax[int(pre_label[num]),:])
                new_weight = Variable(torch.from_numpy(np.asarray(new_weight).reshape(feature_B.size(0),feature_B.size(1),1,1)).cuda())
                temp_hot_map = new_weight * real_feature_maps
                
                temp_hot_map = torch.sum(temp_hot_map,1)
                reshape_temp = temp_hot_map.view(temp_hot_map.size(0), -1)
                reshape_temp = reshape_temp - torch.min(reshape_temp,1)[0].view(reshape_temp.size(0),-1).repeat(1, reshape_temp.size(1))
                reshape_temp = reshape_temp / torch.max(reshape_temp,1)[0].view(reshape_temp.size(0),-1).repeat(1, reshape_temp.size(1))
                
                temp_hot_map = reshape_temp.view(temp_hot_map.size(0), temp_hot_map.size(1), temp_hot_map.size(2))
                temp_hot_map = temp_hot_map.unsqueeze(1)
                Up = torch.nn.Upsample(size = (299))
                im_H = Up(temp_hot_map)
                temp_img += im_H
            
        reshape_temp = temp_img.view(temp_img.size(0), -1)
        reshape_temp = reshape_temp - torch.min(reshape_temp,1)[0].view(reshape_temp.size(0),-1).repeat(1, reshape_temp.size(1))
        reshape_temp = reshape_temp / torch.max(reshape_temp,1)[0].view(reshape_temp.size(0),-1).repeat(1, reshape_temp.size(1))
        temp_img = reshape_temp.view(temp_img.size(0), temp_img.size(1), temp_img.size(2), temp_img.size(3))
        hot_map = temp_img.repeat(1,3,1,1)
        
        SLIC = 12
        hot_map = torch.ceil(hot_map * SLIC) / SLIC
        hot_map = torch.ceil(hot_map - Variable(torch.ones(hot_map.size()).cuda()) / SLIC * slic_size) 
        
        #########SUPER-PIXEL

        claster_num = int(torch.max(image_index).item() + 1)
        rand_noise = (torch.round(torch.rand(real_A.size(0), real_A.size(1), claster_num))*2 -1)  * 2.0 * max_epsilon / 255.0 / 2
        noise = Variable(rand_noise.cuda(),requires_grad = True)
    
        image_index = Variable(image_index.cuda()).long()
    
        adv_noise = torch.gather(noise, 2, image_index).view(real_A.size(0), real_A.size(1),real_A.size(2), real_A.size(3))
        adv = torch.clamp(adv_noise + real_A, -1, 1)
    
        loss_adv = CWLoss
        
        _,_,logist_B = netT(real_A)
        _,target=torch.max(logist_B,1)
        momentum = 0
        
        for i in range(Iter):
            _,_,logist_B = netT(adv)
            _,pre=torch.max(logist_B,1)
            Loss_adv = loss_adv(logist_B, target,-100,False) / real_A.size(0)
            Loss = Loss_adv
            netT.zero_grad()
            if noise.grad is not None:
                noise.grad.data.fill_(0)
            
            Loss.backward()
            
            temp_grad = noise.grad
            std = Variable(torch.FloatTensor(stddev_arr).cuda().view(1, 3, 1, 1).expand_as(real_A))
            temp_noise = torch.gather(temp_grad, 2, image_index).view(real_A.size(0), real_A.size(1),real_A.size(2), real_A.size(3))
            temp_noise = temp_noise.mul( hot_map)
            l2_temp_noise = torch.sqrt(torch.sum( torch.sum( torch.sum(temp_noise.pow(2)* std* std* 255* 255,3),2),1))
            factor = temp_eps / l2_temp_noise
            factor = factor.view(real_A.size(0), 1, 1).expand_as(temp_grad)
            scaled_grad = temp_grad.mul(factor)
            noise = noise - scaled_grad
            noise = Variable(noise.data, requires_grad=True)
        
            adv_noise = torch.gather(noise, 2, image_index).view(real_A.size(0), real_A.size(1),real_A.size(2), real_A.size(3))
            adv_noise = adv_noise.mul( hot_map)
            temp_A = scale(adv_noise + real_A ,real_A ,eps ,stddev_arr ,'l_2')
            adv = torch.clamp(temp_A, -1, 1)
            
        L_X = L2(adv,real_A)
        L_X_show = torch.sqrt(torch.sum(L_X * std * std *255 * 255)/len(real_A))
        reduced_lossX = (L_X_show ).data.clone()
        lossX.update(reduced_lossX.item(), input_A.size(0))
        
        _,_,logist_B = netT(adv)
        _,pre=torch.max(logist_B,1)
        top1 = torch.sum(torch.eq(target.cpu().data.float(),pre.cpu().data.float()).float()) / input_A.size(0)
        
        top1 = torch.from_numpy(np.asarray( [(1 - top1)*100 ])).float().cuda()
        Baccu[0].update(top1[0], input_A.size(0))
        
        dec = 2

        zoom_A = F.interpolate(adv, size = int(299/dec), mode='bilinear')
        zoom_A = F.interpolate(zoom_A, size = 299, mode='bilinear')
        
        _,_,logist_B = netT(zoom_A)
        _,pre=torch.max(logist_B,1)
        
        Ztop1 = torch.sum(torch.eq(target.cpu().data.float(),pre.cpu().data.float()).float()) / input_A.size(0)
        Ztop1 = torch.from_numpy(np.asarray( [(1 - Ztop1)*100 ])).float().cuda()
        Zaccu[0].update(Ztop1[0], input_A.size(0))
            
        print('[{iter:.2f}]\t'
                      'Clean TOP1: {BTOP1.avg:.2f}\t'
                      'Resize TOP1: {ZTOP1.avg:.2f}\t'
                      'lossX: {lossX.avg:.3f}\t'.format(
                          iter = float(idx*100)/len(test_loader), BTOP1 = Baccu[0], ZTOP1 = Zaccu[0], lossX = lossX))

        if not os.path.exists(root):
            os.makedirs(root)

        for i in range(input_A.size(0)):
            clip_img = ToPILImage()((adv[i].data.cpu()+ 1) / 2) 
            adv_img = ToPILImage()(((adv_noise[i].data.cpu()+ 1) / 2) * 100)
            im_AB =Image.fromarray( np.concatenate([clip_img,adv_img], 1))
            save_path = os.path.join(root, image_names[i] + '.png')
            im_AB.save(save_path)
    
def CWLoss(logits, target, kappa=0, tar = True):
    # inputs to the softmax function are called logits.
    # https://arxiv.org/pdf/1608.04644.pdf
    target = torch.ones(logits.size(0)).type(torch.cuda.FloatTensor).mul(target.float())
    target_one_hot = Variable(torch.eye(1000).type(torch.cuda.FloatTensor)[target.long()].cuda())

    # workaround here.
    # subtract large value from target class to find other max value
    # https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
    real = torch.sum(target_one_hot*logits, 1)
    other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)
    
    if tar:
        return torch.sum(torch.max(other-real, kappa))
    else :
        return torch.sum(torch.max(real - other, kappa))
        
def scale(adv_A ,real_A ,eps ,std_arr ,p):
    diff = real_A - adv_A
    std = Variable(torch.FloatTensor(std_arr).cuda().view(1,3,1,1).expand_as(real_A))
    if p == 'l_2':
        mse = torch.nn.MSELoss(reduce=False)
        norm = torch.sum(mse(adv_A , real_A) * std * std * 255 *255 , dim=3)
        norm = torch.sum(norm ,dim=2)
        norm = torch.sqrt(torch.sum(norm ,dim=1))
        #print (norm)
        #print (norm/eps)
        factor = norm/eps
        factor = factor.view(len(real_A),1,1,1).expand_as(real_A)
        diff = diff / factor
        return real_A - diff
    if p == 'l_inf':
        l_channel,_ = torch.max(diff,dim=3)
        l_channel,_ = torch.max(l_channel , dim=2)
        l_channel.view(len(real_A),-1,1,1).expand_as(real_A)
        convert_eps = eps/(std * 255)
        factor = Variable(torch.min(Variable(torch.ones(real_A.size()).cuda()) , convert_eps/l_channel).data)
        diff = diff * factor
        return real_A - diff
 
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
