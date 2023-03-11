import os
import sys

import threadpoolctl
import torch
import numpy as np
import datetime
import wandb

import common_utils
from common_utils.common import AverageValueMeter, load_weights, now, save_weights
from CreateData import setup_problem
from CreateModel import create_model
from extraction import calc_extraction_loss, evaluate_extraction, get_trainable_params
from GetParams import get_args
from problems.mnist_odd_even_noisy_background import get_dataloader
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
    r'C:\Users\Mahdi\Desktop\result_noisy/49000_x.pth',
#     './reconstructions/mnist_odd_even/rbijxft7_x.pth'
]

X = torch.cat([torch.load(x_paths[0], map_location=torch.device('cpu'))])

# Find Nearest Neighbour
xx1 = find_nearest_neighbour(X.double(), Xtrn.double(), search='ncc', vote='min', use_bb=False, nn_threshold=None)
# Scale to Images
xx_scaled, yy_scaled = scale(xx1, Xtrn, ds_mean)
# # Sort
xx, yy, ssims, sort_idxs = sort_by_metric(xx_scaled, yy_scaled, sort='ssim')

# Plot
# color_by_labels = Ytrn[sort_idxs]
color_by_labels = None
figpath=None
plot_table(xx, yy, fig_elms_in_line=15, fig_lines_per_page=4, fig_type='one_above_another', color_by_labels=color_by_labels, figpath=figpath, show=True, dpi=100)