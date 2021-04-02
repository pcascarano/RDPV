##Code for super-resolution (figures $1$ and $5$ from main paper).. Change factor to $8$ to reproduce images from fig. $9$ from supmat.

#You can play with parameters and see how they affect the result.


# import libs
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib
#matplotlib inline
import time
import argparse
import os
import imageio
from PIL import Image
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
from models import *
import torch
import torch.optim
# import EarlyStopping
from pytorchtools import EarlyStopping
from skimage.measure import compare_psnr, compare_mse
from models.downsampler import Downsampler
from pytorch_msssim import ssim
from utils.sr_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize = -1 
factor = 4 # 8
enforse_div32 = 'CROP' # we usually need the dimensions to be divisible by a power of two (32 in this case)
PLOT = True



# set up parameters and net
input_depth = 3
 
INPUT =     'noise'
pad   =     'reflection'
OPT_OVER =  'net'
KERNEL_TYPE='lanczos2'


NET_TYPE = 'skip' # UNet, ResNet   
patience = 10

LR = 0.001
tv_weight = 0.0#0.0000003 ###########################################################################################################

OPTIMIZER = 'adam'

if factor == 4: 
    num_iter = 500
    num_iter_start_early_stop = 300
    reg_noise_std = 0.03
elif factor == 8:
    num_iter = 4000
    reg_noise_std = 0.05
else:
    assert False, 'We did not experiment with other factors'
list_l = []
rg = range(2,3)


eval_array = np.zeros([len(rg),6])
src_dir = 'reconstruction/'
for num in rg:
    net = get_net(input_depth, 'skip', pad,
              upsample_mode='bilinear').type(dtype)

    net.load_state_dict(torch.load('updating_model/recursive_model.pt'))
    net.cuda()
    path_to_image = 'data/example/frame'+str(num)+'.png'

    # load images and baselines
    # Starts here
    imgs = load_LR_HR_imgs_sr(path_to_image , imsize, factor, enforse_div32)

    imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np'] = get_baselines(imgs['LR_pil'], imgs['HR_pil'])

##    if PLOT:
##        plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np']], 4,12);
##        print ('PSNR bicubic: %.4f   PSNR nearest: %.4f' %  (
##                                            compare_psnr(imgs['HR_np'], imgs['bicubic_np']), 
##                                            compare_psnr(imgs['HR_np'], imgs['nearest_np'])))

    
    net_input = get_noise(input_depth, INPUT, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0])).type(dtype).detach()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    

    # Losses
    mse = torch.nn.MSELoss().type(dtype)

    img_LR_var = np_to_torch(imgs['LR_np']).type(dtype)
    img_HR_var = np_to_torch(imgs['HR_np']).type(dtype)
    downsampler = Downsampler(n_planes=3, factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)

    # define closure and optimize
    running_loss = 0.0
    loss_values = []
    ssim_values = []
    def closure():
        global i, net_input, running_loss, loss_values, early_stopping, num_iter_start_early_stop
        
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out_HR = net(net_input)
        out_LR = downsampler(out_HR)

        total_loss = mse(out_LR, img_LR_var) 
        ssim_fun =  ssim(out_HR, img_HR_var, data_range=1, size_average=True) 
        if tv_weight > 0:
            total_loss += tv_weight * tv1_loss(out_HR) # oppure tv_alpha
            
        total_loss.backward()
        running_loss = total_loss.item()
        loss_values.append(running_loss)
        ssim_values.append(ssim_fun.item())
##        plt.semilogy(np.array(loss_values), 'r')
##        plt.xlabel('Number of Iteration')
##        plt.ylabel('Loss')
##        plt.pause(0.05)
        
        if i>num_iter_start_early_stop:
            early_stopping(running_loss, net)
            flag_early_stopping = early_stopping.early_stop# early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        else:
            flag_early_stopping = 0    
        
        # Log
        psnr_LR = compare_psnr(imgs['LR_np'], torch_to_np(out_LR))
        psnr_HR = compare_psnr(imgs['HR_np'], torch_to_np(out_HR))
        #print ('num_frame %d Iteration %05d  PSNR_HR %.3f' % (num,i, psnr_HR), '\r', end='')
                          
                      
        # History
        psnr_history.append([psnr_LR, psnr_HR])
        
          
        if  i % 100 == 0:
            print ('num_frame %d Iteration %05d  PSNR_HR %.3f' % (num,i, psnr_HR), '\r', end='')
            
        
        i += 1
        
        return total_loss, flag_early_stopping



    psnr_history = []
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    i = 0
    p = get_params(OPT_OVER, net, net_input)
    optimize_early_stop(OPTIMIZER, p, closure, LR, num_iter)
    #plt.show()
    out_HR_np = np.clip(torch_to_np(net(net_input)), 0, 1)
    list_l = [num]
    list_l.append(i)
    list_l.append(psnr_history[-1][0])
    list_l.append(psnr_history[-1][1])
    list_l.append(loss_values[-1])
    list_l.append(ssim_values[-1])
    eval_array[num-rg[0],:] = np.array(list_l) 
    name_image = 'reconstruction/frame'+str(num)+'.png'
    out_new = np.zeros([288,288,3])
    out_new[:,:,0]=out_HR_np[0,:,:]
    out_new[:,:,1]=out_HR_np[1,:,:]
    out_new[:,:,2]=out_HR_np[2,:,:]
    
    #out_new= (255.0 / out_new.max() * (out_new - out_new.min())).astype(np.uint8)
    out_new1 = 255*out_new # 255*imgs['LR_np'].max()*out_new
    out_new2= out_new1.astype(np.uint8)
    im=Image.fromarray(out_new2)
    im.save(name_image)
    torch.save(net.state_dict(),'updating_model/recursive_model.pt')
    del p, noise, net_input, net_input_saved, imgs, mse, downsampler, list_l, psnr_history, loss_values, ssim_values, out_HR_np, out_new, out_new1, out_new2, im, early_stopping
eval_name =  'all_frames_evaluation.csv'
np.savetxt(os.path.join(src_dir,eval_name), eval_array)
del eval_array
