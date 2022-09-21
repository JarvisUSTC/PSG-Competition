import argparse
import os
import torch
import subprocess as sp
import time
import re


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config_file", help="config file for train and test", type=str, required=True,
    )
    parser.add_argument(
        "--dataset_names",
        help="used dataset names, split with ','",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset_unzip",
        help="whether using prepare_dataset_compressed",
        action='store_true',
    )
    parser.add_argument(
        "--blob_root", help="path to blob root", type=str, required=True,
    )
    parser.add_argument(
        "--zip_filename", help="zip codebase filename", type=str, required=True,
    )
    parser.add_argument(
        "--output_path", help="output path on blob", type=str, required=True,
    )
    parser.add_argument(
        "--unparsed", help="unparsed", default="", type=str,
    )
    parser.add_argument(
        "--working_dir", required=True, default="", type=str,
    )
    args = parser.parse_args()
    extra_args = args.unparsed
    return args, extra_args


def build_repo(args):
    cmd = f"export PATH={args.conda_prefix}/envs/{args.new_env_name}/bin:{args.conda_prefix}/condabin:$PATH \n"
    cmd += f"pip install -r {args.working_dir}/requirements.txt"
    print(cmd)
    os.system(cmd)

    cmd = f"export PATH={args.conda_prefix}/envs/{args.new_env_name}/bin:{args.conda_prefix}/condabin:$PATH \n"
    cmd += f"cd {args.working_dir} \n"
    cmd += "pip install -v -e . \n"
    print(cmd)
    os.system(cmd)


def prepare_datasets(args):
    dataset_path = os.path.join(args.working_dir, "data")
    dataset_names = args.dataset_names.split(",")
    for dataset_name in dataset_names:
        cmd = f"ln -s {args.blob_root}/data/{dataset_name} {dataset_path}"
        print(cmd)
        os.system(cmd)
        cmd = f"ls {dataset_path}"
        print(cmd)
        os.system(cmd)

    cmd = f"sudo ln -s {args.blob_root}/data/pretrained_models/ {args.working_dir}/checkpoints \n"
    cmd += f"ls {args.working_dir}/checkpoints \n"
    print(cmd)
    sp.run(cmd, shell=True, check=True, cwd=args.working_dir)

def prepare_datasets_compressed(args):
    dataset_path = os.path.join(args.working_dir, "data")
    dataset_names = args.dataset_names.split(",")
    for dataset_name in dataset_names:
        # if dataset_name == 'OpenPSG':
        #     dataset_path_tmp = os.path.join(dataset_path, "psg")
        # else:
        #     dataset_path_tmp = os.path.join(dataset_path, dataset_name)
        dataset_blob_path = f"{args.blob_root}/data/{dataset_name}.zip"
        cmd = f"unzip {dataset_blob_path} -d {dataset_path} >/dev/null\n"
        print(cmd)
        os.system(cmd)
        cmd = f"ls {dataset_path}"
        print(cmd)
        os.system(cmd)
    
    cmd = f"sudo ln -s {args.blob_root}/data/pretrained_models/ {args.working_dir}/checkpoints \n"
    cmd += f"ls {args.working_dir}/checkpoints \n"
    print(cmd)
    sp.run(cmd, shell=True, check=True, cwd=args.working_dir)

def unzip_codebase(args):
    codebase_filepath = os.path.join(args.output_path, args.zip_filename)
    cmd = f"unzip {codebase_filepath} -d {args.working_dir} >/dev/null\n"
    print(cmd)
    os.system(cmd)
    cmd = f"ls {args.working_dir}"
    print(cmd)
    os.system(cmd)


def install_apps():


    cmd = ["which", "sudo"]

    print(cmd)

    proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE)

    stdout, stderr = proc.communicate()

    output = stdout.decode('utf-8').split('\n')[0]

    print(output)

    have_sudo = "sudo" in output

    if have_sudo:

        cmd = "sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \n"

        cmd += "sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \n"

        print(cmd)

        sp.run(cmd, shell=True, check=True)

    else:

        cmd = "apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \n"

        cmd += "apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \n"

        # don't have sudo, maybe run on OCRA or OCR2, have root right, install sudo firstly

        cmd += "apt-get update && apt-get install -y --no-install-recommends sudo >/dev/null\n"

        print(cmd)

        sp.run(cmd, shell=True, check=True)

    cmd = "apt-get update && apt-get install -y apt-transport-https \n"
    cmd += "apt-get install -y --no-install-recommends sudo >/dev/null\n"
    cmd += "sudo apt -y --fix-broken install && sudo apt-get update && sudo apt-get install -y --no-install-recommends zip unzip expect vim-gtk libssl-dev pigz time python3-dev >/dev/null"
    print(cmd)
    sp.run(cmd, shell=True, check=True)

    os.environ["PATH"] += ":{}/.local/bin".format(os.environ["HOME"])
    cmd = "export PATH=$HOME/.local/bin:$PATH"
    print(cmd)
    os.system(cmd)

def install_conda_and_create_new_env(args):
    args.conda_prefix = "$HOME/miniconda"
    INSTALLER_PATH = "$HOME/Miniconda3-latest-Linux-x86_64.sh"
    args.new_env_name = "openpsg"
    # find cuda version
    cmds = ['nvcc', '-V']
    proc = sp.Popen(cmds, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = proc.communicate()
    output = stdout.decode('utf-8').split('\n')[-2]
    CUDA_VERSION = re.findall(r'\d+\.\d+', output)[-1]
    print(CUDA_VERSION)

    # install conda
    cmd = f"wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O {INSTALLER_PATH} >/dev/null \n"
    cmd += f"bash {INSTALLER_PATH} -b -p {args.conda_prefix} -f >/dev/null \n"
    print(cmd)
    sp.run(cmd, shell=True, check=True)

    # create new env
    cmd = f"export PATH={args.conda_prefix}/envs/{args.new_env_name}/bin:{args.conda_prefix}/condabin:$PATH \n"
    cmd += f"conda env create -f {args.working_dir}/environment.yml \n"
    print(cmd)
    sp.run(cmd, shell=True, check=True)

    cmd = f"export PATH={args.conda_prefix}/envs/{args.new_env_name}/bin:{args.conda_prefix}/condabin:$PATH \n"
    cmd += "pip install mmcv-full==1.4.3 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.7.0/index.html"
    print(cmd)
    os.system(cmd)

    cmd = f"export PATH={args.conda_prefix}/envs/{args.new_env_name}/bin:{args.conda_prefix}/condabin:$PATH \n"
    cmd += "pip install openmim"
    print(cmd)
    os.system(cmd)

    cmd = f"export PATH={args.conda_prefix}/envs/{args.new_env_name}/bin:{args.conda_prefix}/condabin:$PATH \n"
    cmd += "pip install mmdet==2.20.0"
    print(cmd)
    os.system(cmd)

    cmd = f"export PATH={args.conda_prefix}/envs/{args.new_env_name}/bin:{args.conda_prefix}/condabin:$PATH \n"
    cmd += "pip install git+https://github.com/cocodataset/panopticapi.git"
    print(cmd)
    os.system(cmd)

    cmd = f"export PATH={args.conda_prefix}/envs/{args.new_env_name}/bin:{args.conda_prefix}/condabin:$PATH \n"
    cmd += "conda install -c conda-forge pycocotools \n"
    cmd += "pip install detectron2==0.5 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.7/index.html"
    print(cmd)
    os.system(cmd)

    cmd = f"export PATH={args.conda_prefix}/envs/{args.new_env_name}/bin:{args.conda_prefix}/condabin:$PATH \n"
    cmd += "pip install wandb \n"
    cmd += "wandb login d8d48d5c16ca9b51769d812605aed1929aca30e1\n"
    print(cmd)
    os.system(cmd)

    cmd = f"export PATH={args.conda_prefix}/envs/{args.new_env_name}/bin:{args.conda_prefix}/condabin:$PATH \n"
    cmd += f"cd {args.working_dir}/openpsg/models/utils/ops \n"
    cmd += f"sh make.sh \n"
    print(cmd)
    os.system(cmd)


def barrier(args, machine_rank, num_machines, dist_url):
    # sync
    print(f"Node {machine_rank} enters barrier.")

    cmd = f"export LC_ALL=C.UTF-8"
    print(cmd)
    os.system(cmd)

    cmd = f"export LANG=C.UTF-8"
    print(cmd)
    os.system(cmd)
    os.environ["LANG"] = "C.UTF-8"
    os.environ["LC_ALL"] = "C.UTF-8"

    os.environ["OPENBLAS_NUM_THREADS"] = "12"

    install_apps()
    unzip_codebase(args)

    build_repo(args)
    if args.dataset_unzip:
        prepare_datasets_compressed(args)
    else:
        prepare_datasets(args)

    cmd = "export GLOO_SOCKET_IFNAME=eth0\n"
    cmd += f"python run_on_aml/barrier.py --dist-url {dist_url} --machine-rank {machine_rank} --num-machines {num_machines}"
    print(cmd)
    sp.run(cmd, shell=True, cwd=args.working_dir, check=True)
    print(f"Node {machine_rank} exits barrier.")


def main():
    install_apps()
    args, extra_args = parse_args()
    args.working_dir = "/OpenPSG"
    cmd = f"sudo mkdir -p {args.working_dir} \n"
    cmd += f"sudo chmod -R 777 {args.working_dir} \n"
    print(cmd)
    sp.run(cmd, shell=True, check=True)

    args.log_dir = os.path.join(args.working_dir, "temp_log")
    work_dir = os.path.join(args.working_dir, 'work_dirs')
    cmd = f"sudo mkdir -p {args.log_dir} \n"
    cmd += f"sudo chmod -R 777 {args.log_dir} \n"
    cmd += f"sudo mkdir -p {work_dir} \n"
    cmd += f"sudo chmod -R 777 {work_dir} \n"
    print(cmd)
    sp.run(cmd, shell=True, check=True)

    args.output_path = os.path.join("/blob", args.output_path)
    cmd = f"sudo ln -s {args.blob_root} /blob \n"
    print(cmd)
    sp.run(cmd, shell=True, check=True, cwd=args.working_dir)

    num_machines = int(os.getenv("OMPI_COMM_WORLD_SIZE", default="1"))
    machine_rank = int(os.getenv("OMPI_COMM_WORLD_RANK", default="0"))
    # cmd = "env | grep NODE\n env | grep OMPI \n env | grep MASTER\n"
    # print(cmd)
    # sp.run(cmd, shell=True, check=True, cwd=args.working_dir)
    print(f"num_machines={num_machines}, machine_rank={machine_rank}")
    gpus = torch.cuda.device_count()

    if num_machines > 1:
        if "MASTER_IP" not in os.environ.keys():
            # OCRA or OCR2
            AZ_BATCH_MASTER_NODE = os.environ["AZ_BATCH_MASTER_NODE"]
            dist_url = f"tcp://{AZ_BATCH_MASTER_NODE}"
        else:
            # k8s compute target
            master_ip = os.environ["MASTER_IP"]
            master_port = os.environ["MASTER_PORT"]
            dist_url = f"tcp://{master_ip}:{master_port}"
        # sync before training
        barrier(args, machine_rank, num_machines, dist_url)
        # start training
        cmd = "export GLOO_SOCKET_IFNAME=eth0\n"

        cmd += (
            f"python train_net.py --resume "
            f"--dist-url {dist_url} "
            f"--machine-rank {machine_rank} "
            f"--num-machines {num_machines} "
            f"--config-file {args.config_file} "
            f"--num-gpus {gpus} "
            f"LOG_TEMP_OUTPUT {args.log_dir} "
            f"OUTPUT_DIR {args.output_path} {extra_args} "
        )
        print(cmd)
        sp.run(cmd, shell=True, check=True, cwd=args.working_dir)
    else:
        cmd = f"export LC_ALL=C.UTF-8"
        print(cmd)
        os.system(cmd)

        cmd = f"export LANG=C.UTF-8"
        print(cmd)
        os.system(cmd)
        os.environ["LANG"] = "C.UTF-8"
        os.environ["LC_ALL"] = "C.UTF-8"
        os.environ["OPENBLAS_NUM_THREADS"] = "12"
        # install_apps()
        unzip_codebase(args)
        install_conda_and_create_new_env(args)

        build_repo(args)
        if args.dataset_unzip:
            prepare_datasets_compressed(args)
        else:
            prepare_datasets(args)

        cmd = f"sudo chmod -R 777 ./ \n" # prevent permission denied
        print(cmd)
        sp.run(cmd, shell=True, check=True)

        """cmd = (
            f"GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8  "
            f"{args.config_file} "
            f"--output_dir {args.output_path} "
            f"--resume {args.output_path}/checkpoint.pth 2>&1 |tee {args.output_path}/azure_log.txt"
        )"""
        dataset_names = args.dataset_names.split(",")

        cmd = f"export PATH={args.conda_prefix}/envs/{args.new_env_name}/bin:{args.conda_prefix}/condabin:$PATH \n"
        cmd += (
            f"python -m torch.distributed.launch "
            f"--nproc_per_node=8 --master_port=29500 "
            f"tools/train.py "
            f"{args.config_file} "
            f"--gpus 8 "
            f"--launcher pytorch "
            # f"--resume_from {args.output_path}/checkpoint.pth "
            f"--work-dir {args.output_path} 2>&1 |tee {args.output_path}/azure_log.txt"
        )
        print(cmd)
        sp.run(cmd, shell=True, check=True, cwd=args.working_dir)

    # copy the log frm working_dir to blob
    """cmd = f"cp -r {args.log_dir} {args.output_path}"
    print(cmd)
    sp.run(cmd, shell=True, check=True, cwd=args.working_dir)"""


if __name__ == "__main__":
    main()
