import argparse
import ast


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(v):
    return ast.literal_eval(v)


def get_args(*args):
    parser = argparse.ArgumentParser(description='')

    # general parameters
    parser.add_argument('--cuda', default='true', type=str2bool, help='')
    parser.add_argument('--seed', default=1, type=int, help='')
    parser.add_argument('--proj_name', default='', help='description of run, for saving stuff')
    parser.add_argument('--precision', default='double', help='')
    parser.add_argument('--run_mode', default='train', help='options: train, reconstruct', choices=['train', 'reconstruct'])

    # files
    # parser.add_argument('--datasets_dir', default='datasets', help='this is loaded from settings.py')
    # parser.add_argument('--results_base_dir', default='./', help='this is loaded from settings.py')
    parser.add_argument('--pretrained_model_path', default='', help='path to pretrained model (ignored if empty)')
    parser.add_argument('--train_save_model', default='true', type=str2bool, help='')
    parser.add_argument('--train_save_model_every', default=100000, type=int, help='only saves if >0')
    parser.add_argument('--extract_save_results', default='true', type=str2bool, help='')
    parser.add_argument('--extract_save_results_every', default=5000, type=int, help='only saves if >0')
    parser.add_argument('--save_args_files', default='true', type=str2bool, help='')

    # wandb
    parser.add_argument('--wandb_active', default='false', type=str2bool, help='actviate wandb logging')
    parser.add_argument('--wandb_project_name', default='Dataset_Extraction', help='')
    parser.add_argument('--wandb_entity', default='dataset_extraction', help='')

    # data creation
    parser.add_argument('--problem', default='cifar10_noisy_background', help='')
    parser.add_argument('--data_per_class_train', default=50, type=int, help='')
    parser.add_argument('--data_per_class_val', default=0,type=int, help='')
    parser.add_argument('--data_per_class_test', default=1000, type=int, help='')
    parser.add_argument('--data_reduce_mean', default='true', type=str2bool, help='')
    parser.add_argument('--noise_perc', default=1.0, type=float, help='percentage of noisy images in the dataset')

    # model_train
    parser.add_argument('--model_type', default='mlp', help='options: mlp')
    parser.add_argument('--model_hidden_list', default='[1000, 1000]', type=str2list, help='should be a list, even for one item. ')
    parser.add_argument('--model_use_bias', default='false', type=str2bool, help='')

    # train
    parser.add_argument('--use_init_scale', default='true', type=str2bool, help='')
    parser.add_argument('--use_init_scale_only_first', default='true', type=str2bool, help='')
    parser.add_argument('--model_init_list', default='[0.0001, 0.0001]', type=str2list, help='should be a list of the same size as model_hidden_list')
    parser.add_argument('--model_train_activation', default='relu', help='options: relu, sigmoid, leakyrelu, tanh')
    parser.add_argument('--train_epochs', default=5000000, type=int, help='')
    parser.add_argument('--train_lr', default=0.01, type=float, help='')
    parser.add_argument('--train_evaluate_rate', default=5000, type=int, help='')
    parser.add_argument('--train_threshold', default=1e-15, type=float, help='stop training below this loss')
    parser.add_argument('--train_SGD', default='false', type=str2bool, help='')
    parser.add_argument('--train_SGD_batch_size', default=64, type=int, help='')

    # extraction
    parser.add_argument('--extraction_epochs', default=50000, type=int, help='')
    parser.add_argument('--margin', default=12.0, type=float, help='training data margin')
    parser.add_argument('--extraction_data_amount_per_class', default=100, type=int, help='0 = same as data_amount')
    parser.add_argument('--extraction_model_activation', default='modifiedrelu', help='options: same as model_train_activation')
    parser.add_argument('--extraction_model_relu_alpha', default=149.86555429083975, type=float, help='')
    parser.add_argument('--extraction_init_scale', default=0.03497673778414215, type=float, help='')
    parser.add_argument('--extraction_lr', default=0.03052419903283405, type=float, help='')
    parser.add_argument('--extraction_lambda_lr', default=1e-4, type=float, help='')
    parser.add_argument('--extraction_lr_x', default=1e-4, type=float, help='')
    parser.add_argument('--extraction_lr_l', default=1e-4, type=float, help='')
    parser.add_argument('--extraction_evaluate_rate', default=500, type=int, help='')
    parser.add_argument('--extraction_min_lambda', default=0.4470505589528116, type=float, help='minimum lambda in the extraction loss')
    parser.add_argument('--extraction_loss_type', default='kkt', help='options: kkt, naive')
    parser.add_argument('--extraction_stop_threshold', default=5000, type=int)
    if not isinstance(args, list):
        args = args[0]
    args = parser.parse_args(args)

    return args


