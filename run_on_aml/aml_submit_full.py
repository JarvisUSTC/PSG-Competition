#
# This Python script submits the specific job to AML.
#
# Author: Philly Beijing Team <PhillyBJ@microsoft.com>

#
# We have the following assumptions:
#     1. A workspace has been created in advance.
#     2. An existing compute (either a GPU- or a CPU-cluster) has been created in the workspace.
#     3. An existing external datastore has been registered to the workspace.
#
import os
import time
import json
import argparse
import subprocess as sp
from azureml.core import Workspace, Datastore, Experiment, ScriptRunConfig, Environment
from azureml.core.runconfig import MpiConfiguration
from azureml.core.compute import ComputeTarget
from azureml.core.container_registry import ContainerRegistry
from azureml.contrib.core.k8srunconfig import K8sComputeConfiguration
from azureml.train.estimator import Estimator, Mpi
from azureml.core.authentication import AzureCliAuthentication
from azureml.contrib.aisc.aiscrunconfig import AISuperComputerConfiguration


def make_container_registry(address, username, password):
    cr = ContainerRegistry()
    cr.address = address
    cr.username = username
    cr.password = password
    return cr


def get_ws_ct(target):
    workspace_dict = {
        # intern can't access these clusters
        # OCRA targets
        # https://ml.azure.com/compute/list/training?wsid=/subscriptions/b8da985a-830d-4d20-b9e5-8d4c0d798c7f/resourcegroups/Vision_GPU/workspaces/OCRA&tid=72f988bf-86f1-41af-91ab-2d7cd011db47
        "NC24rsv3-prod": "run_on_aml/.azureml/OCRA.json", # 4x 16G V100, total 74 nodes
        "NC24rsv3-prod2": "run_on_aml/.azureml/OCRA.json", # 4x 16G V100, total 58 nodes
        # OCR2 targets
        # https://ml.azure.com/compute/list/training?wsid=/subscriptions/b8da985a-830d-4d20-b9e5-8d4c0d798c7f/resourceGroups/FY20Vision/workspaces/OCR2&tid=72f988bf-86f1-41af-91ab-2d7cd011db47
        "ND40rsv2": "run_on_aml/.azureml/OCR2.json", # 8x 32G V100, total 6 nodes
        "ND40rsv2-prod1": "run_on_aml/.azureml/OCR2.json", # 8x 32G V100, total 6 nodes
        "ND40rsv2-prod2": "run_on_aml/.azureml/OCR2.json", # 8x 32G V100, total 8 nodes
        "ND40rsv2-prod3": "run_on_aml/.azureml/OCR2.json", # 8x 32G V100, total 8 nodes
        # OCR-ITP targets
        "f4sv2-8gb-wus2": "run_on_aml/.azureml/vision-itp-ocr-res-ws01-westus2.json", # CPU only target, total 8 nodes
    }
    k8s_workspace_dict = {
        # k8s all targets list
        # https://ml.azure.com/clusters?flight=itpmerge
        # OCR-ITP targets
        "v100-32gb-wus2": "run_on_aml/.azureml/vision-itp-ocr-res-ws01-westus2.json", # 8x 32G V100, quota 8 GPUs
        "v100-8x-scus": "run_on_aml/.azureml/vision-itp-ocr-res-ws01-scus.json", # 8x 32G V100, quota 8 GPUs
        # researchvc targets
        # https://dev.azure.com/msresearch/GCR/_wiki/wikis/GCR.wiki/3438/AML-Kubernetes-(aka-AML-K8s)(aka-ITP)-Overview
        "itplabrr1cl1": "run_on_aml/.azureml/resrchvc.json", # 8x 32G V100 PCIe, quota 24 GPUs
        "itpeusp100cl": "run_on_aml/.azureml/resrchvc-eus.json", # 4x 16G P100 PCIe, quota 32 GPUs
        "itpeusp40cl": "run_on_aml/.azureml/resrchvc-eus.json", # 4x 24G P40 PCIe, quota 64 GPUs
    }
    sing_dict = {
        # Singularity targets
        # https://ml.azure.com/virtualClusters?flight=itpmerge
        "msrresrchws": "run_on_aml/.azureml/msrresrchws.json",  # 16G V100 & 80G A100, quota unknow, maybe is 64 V100 (32 for ) and 24 A100
        "msroctows": "run_on_aml/.azureml/msroctows.json",  # 16G V100, quota 64 GPUs
        "scus-sing": "run_on_aml/.azureml/vision-itp-ocr-res-ws01-scus-sing.json", # 32G V100
    }
    if target in workspace_dict.keys():
        workspace_config = workspace_dict[target]
        isk8s_target = False
        is_singularity_target = False
    elif target in k8s_workspace_dict.keys():
        workspace_config = k8s_workspace_dict[target]
        isk8s_target = True
        is_singularity_target = False
    else:
        assert target in sing_dict.keys()
        workspace_config = sing_dict[target]
        isk8s_target = False
        is_singularity_target = True
    
    cli_auth = AzureCliAuthentication()
    ws = Workspace.from_config(workspace_config, auth=cli_auth)
    # ws = Workspace.from_config(workspace_config)

    if is_singularity_target:
        ct = None
    else:
        ct = ComputeTarget(workspace=ws, name=target)
    print("workspace: {}, target: {}".format(workspace_config, target))
    return ws, ct, isk8s_target, is_singularity_target

def azupload(src, dest, SAS):
    cmd = 'sudo azcopy copy "{}" "https://zhuzho.blob.core.windows.net/v-jiaweiwang/{}{}" '.format(
        src, dest, SAS
    )
    print(cmd)
    sp.run(cmd, shell=True, check=True)


def check_config_and_upload_codes(args):

    if not os.path.exists(args.config_file):
        raise ValueError(f"{args.config_file} not found")

    config_file = args.config_file
    output_path = os.path.splitext(config_file)[0].replace("configs/", "")
    zip_filename = f"PSG@{output_path.replace('/', '_')}@{args.target}@{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.zip" # @{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.zip"
    zip_dirname = "zip_codebase"
    zip_filepath = f"{zip_dirname}/{zip_filename}"
    cmd = f"mkdir -p {zip_dirname} \n"
    cmd += f'zip -ry {zip_filepath} . -x "run_on_aml/*" -x "zip_codebase/*" -x "work_dirs/*" -x "data/*" -x "wandb/*" -x "submission/*" -x "openpsg.egg-info/*" -x "checkpoints/*" >/dev/null \n'
    print(cmd)
    sp.run(cmd, shell=True, check=True)
    output_dir = os.path.join(args.blob_output_root, output_path)
    azupload(zip_filepath, os.path.join(output_dir, zip_filename), args.SAS)
    return zip_filename, output_dir

def main():
    parser = argparse.ArgumentParser(description='AzureML Job Submission')
    # The following are AML job submission related arguments.
    parser.add_argument(
        '--target', 
        required=False, 
        default='v100-32gb-wus2',
        help='which target'
    )
    parser.add_argument(
        '--node_nums', 
        required=False, 
        default=1, 
        type=int,
        help='num of nodes used'
    )
    parser.add_argument(
        '--gpu_nums',   # only useful when use k8s targets and use 1 node
        required=False, 
        default=8, 
        type=int,
        help='num of gpus used'
    )
    parser.add_argument(
        "--config_file", required=False, type=str, help="config file for code"
    )
    parser.add_argument(
        '--azureml_config_file', 
        required=False, 
        type=str,
        help="azureml config file",
        default='v-kahu1.json',
    )
    parser.add_argument(
        '--preemption_allowed',  # only useful when use k8s targets
        action="store_true",
        help="preemption allowed",
    )
    parser.add_argument(
        '--instance_type',   # only useful when use singularity
        required=False, 
        type=str,
        help="singularity instance type",
        default='ND40s_v2g1',
    )
    parser.add_argument(
        '--sla_tier',   # only useful when use singularity
        required=False, 
        type=str,
        help="singularity sla_tier",
        default='Basic',
    )
    parser.add_argument(
        '--singularity_image_version',   # only useful when use singularity
        required=False, 
        type=str,
        help="singularity image_version",
        default='pytorch-1.9.0-a100',
    )
    parser.add_argument(
        "--dataset_names",
        help="used dataset names, split with ','",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--dataset_unzip",
        help="whether using prepare_dataset_compressed",
        action='store_true',
    )
    parser.add_argument(
        "--sleep_for_debug",
        help="sleep hours for debug",
        type=int,
        required=False,
        default=-1,
    )
    parser.add_argument("--experiment_name", type=str, required=True)

    args, unparsed = parser.parse_known_args()
    with open(args.azureml_config_file, "r") as fp:
        data = json.load(fp)
    args.datastore = data['datastore_name']
    args.exp_name = data['experiment_group_name']
    args.blob_output_root = data["blob_output_root"]
    args.SAS = data['SAS']
    container_registry_address = data['container_registry_address']
    container_registry_username = data['container_registry_username']
    container_registry_password = data['container_registry_password']
    docker_image = data['docker_image']
    unparsed = " ".join(unparsed)
    ws, ct, isk8s, is_singularity_target = get_ws_ct(args.target)
    ds = Datastore(workspace=ws, name=args.datastore)
    ds_ref = ds.path("./").as_mount()
    if args.sleep_for_debug > 0:
        entry_script = "sleep.py"
        script_params = [
            "--blob_root",
            str(ds_ref),
            "--sleep_for_debug",
            args.sleep_for_debug,
        ]
    else:
        if "ND40rsv2" in args.target:
            entry_script = "run_on_aml_ocr2.py"
        else:
            entry_script = "run_on_aml.py"
        zip_filename, output_path = check_config_and_upload_codes(args,)
        if args.dataset_unzip:
            script_params = [
                "--config_file",
                args.config_file,
                "--blob_root",
                str(ds_ref),
                "--dataset_names",
                args.dataset_names,
                "--dataset_unzip",
                "--zip_filename",
                zip_filename,
                "--output_path",
                output_path,
                "--working_dir",
                args.experiment_name,
            ]
        else:
            script_params = [
                "--config_file",
                args.config_file,
                "--blob_root",
                str(ds_ref),
                "--dataset_names",
                args.dataset_names,
                "--zip_filename",
                zip_filename,
                "--output_path",
                output_path,
                "--working_dir",
                args.experiment_name,
            ]
        if unparsed != "":
            script_params["--unparsed"] = unparsed
    env_name = args.experiment_name
    est = ScriptRunConfig(
        source_directory="run_on_aml",
        script=entry_script,
        arguments=script_params,
        compute_target=ct,
        environment=Environment.from_docker_image(
            env_name,
            docker_image,
            container_registry=make_container_registry(
                address=container_registry_address,
                username=container_registry_username,
                password=container_registry_password,
            ),
        ),
        distributed_job_config=MpiConfiguration(
            process_count_per_node=1, node_count=args.node_nums
        ),
    )
    est.run_config.data_references = {ds_ref.data_reference_name: ds_ref.to_config()}
    est.run_config.docker.shm_size = "512g"
    est.run_config.docker.use_docker = True
    est.run_config.communicator = "OpenMPI"

    if isk8s:
        itpconfig = K8sComputeConfiguration()
        itp = dict()
        itp['gpu_count'] = args.gpu_nums
        itp['job_priority'] = 200
        itp['preemption_allowed'] = args.preemption_allowed
        itpconfig.configuration = itp
        est.run_config.cmk8scompute = itpconfig

    if is_singularity_target:
        if args.target == 'msrresrchws':
            virtual_cluster_arm_id = "/subscriptions/22da88f6-1210-4de2-a5a3-da4c7c2a1213/resourceGroups/gcr-singularity-resrch/providers/Microsoft.MachineLearningServices/virtualclusters/msrresrchvc"
        elif args.target == 'msroctows':
            virtual_cluster_arm_id = "/subscriptions/d4404794-ab5b-48de-b7c7-ec1fefb0a04e/resourceGroups/gcr-singularity-octo/providers/Microsoft.MachineLearningServices/virtualclusters/msroctovc"
        elif args.target == 'scus-sing':
            virtual_cluster_arm_id = "/subscriptions/48b6cd5e-3ffe-4c2e-9e99-5760a42cd093/resourceGroups/vision-sing-ocr/providers/Microsoft.MachineLearningServices/workspaces/vision-sing-ocr-res-ws01-scus"
        est.run_config.target = "aisupercomputer"
        est.run_config.aisupercomputer = AISuperComputerConfiguration()
        est.run_config.aisupercomputer.instance_type = args.instance_type
        est.run_config.aisupercomputer.sla_tier = args.sla_tier
        est.run_config.aisupercomputer.image_version = args.singularity_image_version
        est.run_config.node_count = args.node_nums
        est.run_config.aisupercomputer.priority = "High"
        est.run_config.aisupercomputer.scale_policy.auto_scale_interval_in_sec = 36000
        est.run_config.aisupercomputer.scale_policy.max_instance_type_count = args.node_nums
        est.run_config.aisupercomputer.scale_policy.min_instance_type_count = args.node_nums
        est.run_config.environment.environment_variables['OMPI_COMM_WORLD_SIZE'] = str(args.node_nums)
        est.run_config.aisupercomputer.virtual_cluster_arm_id = virtual_cluster_arm_id

    tags = {'id': args.config_file}

    run = Experiment(workspace=ws, name=args.exp_name).submit(est, tags=tags)
    print("URL: ", run.get_portal_url())
    print("Success!")


if __name__ == "__main__":
    main()
