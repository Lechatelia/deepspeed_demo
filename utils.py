import argparse
import torch
import socket
import subprocess
import os
import time
import sys
import deepspeed


def check_dist_portfile():
    if "SLURM_JOB_ID" in os.environ and int(os.environ["SLURM_PROCID"]) == 0:  # rank==0
        hostfile = "dist_url_" + os.environ["SLURM_JOBID"] + ".txt"
        if os.path.exists(hostfile):
            os.remove(hostfile)


def find_free_port():
    s = socket.socket()
    s.bind(('', 0))  # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        if int(os.environ["RANK"]) == 0:
            print('this task is not running on cluster!')
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
        # addr = socket.gethostname()
        if os.environ.get('MASTER_ADDR_OVERRIDE', None):
            os.environ['MASTER_ADDR'] = os.environ['MASTER_ADDR_OVERRIDE']
        if os.environ.get('MASTER_PORT_OVERRIDE', None):
            os.environ['MASTER_PORT'] = os.environ['MASTER_PORT_OVERRIDE']

        addr = os.environ['MASTER_ADDR']
        # addr = socket.gethostbyname(os.environ['MASTER_ADDR'])

    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        if proc_id == 0:
            print('Init dist using slurm!')
            print("Job Id is {} on {} ".format(os.environ["SLURM_JOBID"], os.environ['SLURM_NODELIST']))
        ntasks = int(os.environ['SLURM_NTASKS'])
        # node_list = os.environ['SLURM_NODELIST']
        node_list = os.environ['SLURM_STEP_NODELIST']
        # node_list = os.environ['SLURM_STEP_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
        jobid = os.environ["SLURM_JOBID"]
        hostfile = "dist_url_" + jobid + ".txt"
        if proc_id == 0:
            args.tcp_port = str(find_free_port())
            print('write port {} to file: {} '.format(args.tcp_port, hostfile))
            with open(hostfile, "w") as f:
                f.write(args.tcp_port)
        else:
            print('read port from file: {}'.format(hostfile))
            while not os.path.exists(hostfile):
                time.sleep(1)
            time.sleep(2)
            with open(hostfile, "r") as f:
                args.tcp_port = f.read()

        os.environ['MASTER_PORT'] = str(args.tcp_port)
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('rank: {} addr: {}  port: {}'.format(args.rank, addr, os.environ['MASTER_PORT']))
    # torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    deepspeed.init_distributed(dist_backend=args.dist_backend,  init_method=args.dist_url)
    torch.distributed.barrier()
    if 'SLURM_PROCID' in os.environ and args.rank == 0:
        if os.path.isfile(hostfile):
            os.remove(hostfile)


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)")

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def create_param_group(model, args):

    params = []
    default_group = {
        "params": [],
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "name": 'default'
    }

    for name, param in model.named_parameters():
        if param.ndim <= 1:
            params += [{
                "params": [param],
                "lr": 0.0001,
                "weight_decay": 0.0002,
                "name": name,
            }]
        else:
            default_group['params'].append(param)

    params.append(default_group)

    return params

def deepspeed_init(model, parameters, args):
    print("Creating DeepSpeed engine...")


    ds_config = {
        "train_micro_batch_size_per_gpu": 64,
        "gradient_accumulation_steps":  1,
        "steps_per_print": 10 * 1000,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.0001,
                "betas": (0.9, 0.95),
                "eps": 1e-6,
                "weight_decay": 0.0002,
                "adam_w_mode": True,
                "torch_adam": True
            }
        },
        "gradient_clipping": 0.5,
        "prescale_gradients": False,
        "fp16": {
            "enabled": True,
            "auto_cast": True,
            "fp16_master_weights_and_grads": False,
            "loss_scale": 0,
            "loss_scale_window": 50,
            "hysteresis": 2,
            "min_loss_scale": 32.0,  # for the first 10k iters
            "initial_scale_power": 13
        },
        "amp": {
            "enabled": False,
            "opt_level": "O1",
            "min_loss_scale": 32.0,  # for the first 10k iters
        },
        "bfloat16": {
            "enabled": False
        },
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": 1,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 50000000,
            "reduce_bucket_size": 50000000,
            "overlap_comm": False,
            "contiguous_gradients": True,
            "offload_param": None,
            "offload_optimizer": None,
        },
        "flops_profiler": {
            "enabled": False,
            "profile_step": 10,
            "module_depth": -1,
            "top_modules": 1,
            "detailed": True,
            "output_file": None,
            "end_profile_step": 5,
        },
    }



    model_engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=parameters, config=ds_config)

    return model_engine, optimizer