"""
A main training script.
"""


# Copyright (c) Facebook, Inc. and its affiliates.
import warnings
warnings.filterwarnings('ignore') # never print matching warnings
import logging
import os
from collections import OrderedDict
import torch
from torch import nn
from utils import default_argument_parser, init_distributed_mode, check_dist_portfile, create_param_group, deepspeed_init
from model import ToyModel

def main(args):
    model = ToyModel()
    parameters = create_param_group(model, args)
    model, optimizer = deepspeed_init(model, parameters, args)


    if args.start_step > 0:
        _, client_sd = model.load_checkpoint(args.output_dir, args.start_step, load_optimizer_states=args.load_optimizer_states)


    criterion = nn.MSELoss()

    for step in range(args.start_step, 1000):
        #forward() method
        x = torch.randn(3, 10).to('cuda')
        y = torch.randn(3, 10).to('cuda')
        out = model(x)
        loss = criterion(out, y.to(dtype=out.dtype))

        #runs backpropagation
        model.backward(loss)

        #weight update
        model.step()

        #save checkpoint
        if step > 0 and step % args.save_interval == 0:
            client_sd= { "step": step}
            model.save_checkpoint(args.output_dir, str(step), client_state=client_sd)

    print('---finished----')


def get_args_parser():
    parser = default_argument_parser()

    parser.add_argument('--init_method', default='slurm', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--zero_stage', default=1, type=int)
    parser.add_argument('--start_step', default=0, type=int)
    parser.add_argument('--load_optimizer_states', default=False, type=bool)

    parser.add_argument('--save_interval', default=200, type=int)
    parser.add_argument('--output_dir', default='outputs', type=str)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args_parser()
    print("Command Line Args:", args)
    if args.init_method == 'slurm':
        # slurm init
        check_dist_portfile()
        init_distributed_mode(args)
        main(args)
    elif args.init_method == 'pytorch':
        main(args)
    else:
        raise NotImplementedError