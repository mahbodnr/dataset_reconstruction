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


        
        


pretrained_model_path = r"C:\Users\Mahdi\Desktop\University\Image Reconstruction Project\Results\28Feb\cifar_noisy_1\750000_epoch_train\weights-750000-cifar10_noisy_background_d20.pth"

model = create_model(args, extraction=True)
model.eval()
model = load_weights(model, pretrained_model_path, device=args.device)

for param in model.parameters():
    print(param)
    break

