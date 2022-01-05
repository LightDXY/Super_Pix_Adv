from slic import get_slic
import scipy.io as scio
import os
import math
import numpy as np
import random 

K = 4000
M = 20
N = 1
size = 299
dir_A = '/mnt/blob/testset/resize_val'


out_dir = './DATASET/slic/'+str(int(K))+'_'+str(int(M))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
imgs= os.listdir(dir_A)
random.shuffle(imgs)
A_paths = []
A_names = []
finish = os.listdir(out_dir)
for img in imgs :
    if img not in finish:
        if img[-4:]=='JPEG' or img[-3:]=='png' or img[-3:]=='jpg':
            imgpath=os.path.join(dir_A,img)
            A_paths.append(imgpath)
            A_names.append(img.split('.')[0])

for idx in range(len(A_paths)):
    c_map = get_slic(A_paths[idx], K, M, N, size)
    save = os.path.join(out_dir,A_names[idx]+'.mat')
    scio.savemat(save, {'slic':c_map})
    if idx % 10 == 0:
        print (idx)
        

