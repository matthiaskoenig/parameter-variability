import argparse
import os
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mean', type=float, default=np.log(2.5))
    parser.add_argument('--variance', type=float, default=1)
    parser.add_argument('--parameter', type=str, default='k')
    parser.add_argument('-n', type=int, default=1)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--model_path', type=str, default='model2.xml')

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--dir_result_name', type=str, default='sampling')
    parser.add_argument('--plot_data', type=bool, default=True)

    args = parser.parse_args()

    setattr(args, 'save_dir', args.data_dir+'/'+args.dir_result_name)

    os.makedirs(args.save_dir, exist_ok=True)

    return args




