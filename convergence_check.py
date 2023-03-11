import os
import sys

import threadpoolctl
import torch
import numpy as np
import datetime
import wandb
import matplotlib.pyplot as plt
import common_utils
from common_utils.common import AverageValueMeter, load_weights, now, save_weights
from CreateData import setup_problem
from CreateModel import create_model
from extraction import calc_extraction_loss, evaluate_extraction, get_trainable_params
from GetParams import get_args
# from problems.cifar10_noisy_background import get_dataloader
from problems.cifar10_vehicles_animals import get_dataloader

from analysis import *



def setup_args(args):
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    from settings import datasets_dir, models_dir, results_base_dir
    args.results_base_dir = results_base_dir
    args.datasets_dir = datasets_dir
    if args.pretrained_model_path:
        args.pretrained_model_path = os.path.join(models_dir, args.pretrained_model_path)
    args.model_name = f'{args.problem}_d{args.data_per_class_train}'
    if args.proj_name:
        args.model_name += f'_{args.proj_name}'

    torch.manual_seed(args.seed)

    if args.wandb_active:
        wandb.init(project=args.wandb_project_name, entity='dataset_reconsruction')
        wandb.config.update(args)

    if args.wandb_active:
        args.output_dir = wandb.run.dir
    else:
        import dateutil.tz
        timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
        run_name = f'{timestamp}_{np.random.randint(1e5, 1e6)}_{args.model_name}'
        args.output_dir = os.path.join(args.results_base_dir, run_name)
    print('OUTPUT_DIR:', args.output_dir)

    args.wandb_base_path = './'

    return args


args = get_args(sys.argv[1:])
args = setup_args(args)

        
        
train_loader, test_loader, val_loader = get_dataloader(args)    

# train set
Xtrn, Ytrn = next(iter(train_loader))
ds_mean = Xtrn.mean(dim=0, keepdims=True).data
Xtrn = Xtrn.data - ds_mean.data

x_paths = [
    # r'C:\Users\Mahdi\Desktop\University\Image Reconstruction Project\Code\dataset_reconstruction\reconstructions\cifar10_vehicles_animals\k60fvjdy_x.pth',
    r'C:\Users\Mahdi\Desktop\University\Image Reconstruction Project\Code\dataset_reconstruction\reconstructions\cifar10_vehicles_animals\b9dfyspx_x.pth',
    # r'C:\Users\Mahdi\Desktop\University\Image Reconstruction Project\Results\28Feb\cifar_proposed_correct\36000_x.pth'
]

# X_rec = torch.cat([torch.load(x_paths[0], map_location=torch.device('cpu')), torch.load(x_paths[1], map_location=torch.device('cpu'))])
X_rec = torch.cat([torch.load(x_paths[0], map_location=torch.device('cpu'))])




model = create_model(args, extraction=True)
model.eval()
model = load_weights(model, args.pretrained_model_path, device=args.device)


print("Xtrn shape: ", Xtrn.shape)

print("XRec shape: ", X_rec.shape)

# Find Nearest Neighbour
xx1 = find_nearest_neighbour(X_rec.double(), Xtrn.double(), search='ncc', vote='min', use_bb=False, nn_threshold=None)
_, _, _, sort_idxs_temp = sort_by_metric(xx1, Xtrn, sort='ssim')
# Scale to Images
xx_scaled, yy_scaled = scale(xx1, Xtrn, ds_mean)

xx, yy, ssims, sort_idxs = sort_by_metric(xx_scaled, yy_scaled, sort='ssim')


# print('xx shape: ', xx.shape)
# xx_scaled = xx_scaled.to(torch.float32)
# X = X.to(torch.float32)

# outputs_scaled = model(xx_scaled)
# outputs = model(X)




def eval_kkt(X):
    # grad_norms = []
    X = X.to(torch.float32)
    # # Calculate dummy gradients
    # for i in range (X.shape[0]):
    #     x = X[i:i+1]
    #     model(x).mean().backward()
    #     grads = []
    #     for param in model.parameters():
    #         grads.append(param.grad.view(-1))

    #     grads = torch.cat(grads)
    #     grad_norms.append(torch.linalg.norm(grads) / int(grads.shape[0]))

    # print('max grad norms: ', max(grad_norms))    
    # print('---------------------------------------------')
    # print('---------------------------------------------')
    outputs = model(X)
    print(outputs[sort_idxs[0:11]])
    print('MIN OUTPUTS: ', torch.min(torch.abs(outputs)))
    print('MAX OUTPUTS: ', torch.max(torch.abs(outputs)))
    print('Average OUTPUTS: ', torch.mean(torch.abs(outputs)))




eval_kkt(xx1)    
# print(xx1.shape)

# print('sort_idxs shape: ', sort_idxs.shape)
# print(torch.max(sort_idxs))

# print(yy[0].shape)

# plt.imshow(xx1[0].permute(2,1,0))
# plt.show()



# lambdas_dir = r'C:\Users\Mahdi\Desktop\University\Image Reconstruction Project\Results\28Feb\cifar_noisy_1\750000_epoch_train\45000_l.pth'

# lambdas = torch.load(lambdas_dir, map_location=torch.device('cpu'))

# print(lambdas)










# # Find Nearest Neighbour
# xx1 = find_nearest_neighbour(X.double(), Xtrn.double(), search='ncc', vote='min', use_bb=False, nn_threshold=None)
# # Scale to Images
# xx_scaled, yy_scaled = scale(xx1, Xtrn, ds_mean)
# # # Sort
# xx, yy, ssims, sort_idxs = sort_by_metric(xx1, Xtrn, sort='ssim')

# # Plot
# # color_by_labels = Ytrn[sort_idxs]
# color_by_labels = None
# figpath=None
# plot_table(xx, yy, fig_elms_in_line=15, fig_lines_per_page=4, fig_type='one_above_another', color_by_labels=color_by_labels, figpath=figpath, show=True, dpi=100)





# print(sort_idxs_temp[0:20])
# print(sort_idxs[0:20])
# print(ssims[sort_idxs[0:11]])