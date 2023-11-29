import argparse
import os
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    # Sampler and Model parameters
    parser.add_argument('-n', type=int, default=1)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--model_path', type=str, default='model2.xml')
    parser.add_argument('--parameter', type=str, default='k')

    # Sampler only parameters
    parser.add_argument('--sampler_mean', type=float, default=np.log(2.5))
    parser.add_argument('--sampler_variance', type=float, default=1)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--dir_result_name', type=str, default='sampling')
    parser.add_argument('--plot_data', type=bool, default=True)

    # Model only parameters
    parser.add_argument('--prior_mean', type=float, default=np.log(2))
    parser.add_argument('--prior_variance', type=float, default=1)
    parser.add_argument('--compartment', type=str, default='[y_gut]')
    parser.add_argument('--n_post', type=int, default=2000)

    args = parser.parse_args()

    setattr(args, 'save_dir', args.data_dir+'/'+args.dir_result_name)

    os.makedirs(args.save_dir, exist_ok=True)

    return args




