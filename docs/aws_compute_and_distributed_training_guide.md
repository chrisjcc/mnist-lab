# AWS Compute Architectures & Distributed PyTorch Training Guide

This guide explains how interactive cloud environments (like Amazon SageMaker Studio) operate, how traditional High-Performance Computing (HPC) paradigms like **Slurm** and **HTCondor** map to Amazon Web Services (AWS), and how to transition this PyTorch training pipeline from single-node execution to multi-GPU, multi-node distributed training.

---

## 📑 Table of Contents

1. [Demystifying Interactive Compute: SageMaker Studio vs. Edge Compute](#1-demystifying-interactive-compute-sagemaker-studio-vs-edge-compute)
2. [Mapping Traditional HPC (Slurm / Condor) to AWS Services](#2-mapping-traditional-hpc-slurm--condor-to-aws-services)
3. [The Industry Standard: SageMaker Managed Training Jobs](#3-the-industry-standard-sagemaker-managed-training-jobs)
4. [Scaling the Pipeline: Adapting PyTorch for Distributed Training](#4-scaling-the-pipeline-adapting-pytorch-for-distributed-training)
5. [Architecture Comparison & Decision Matrix](#5-architecture-comparison--decision-matrix)

---

## 1. Demystifying Interactive Compute: SageMaker Studio vs. Edge Compute

When you open a terminal in Amazon SageMaker Studio and run `make data`, `make train`, or `make test`, it is important to understand what hardware you are actually touching:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Interactive Cloud Workspace                           │
│                                                                             │
│  [Developer Laptop] ──(Web Browser / SSH)──► [SageMaker Studio Workspace]   │
│                                                       │                     │
│                                           Running on Single EC2 VM          │
│                                           (e.g., ml.m5.large, ml.g4dn)      │
│                                           • Ephemeral EBS storage           │
│                                           • Shared EFS home directory       │
│                                           • Interactive bash shell          │
└─────────────────────────────────────────────────────────────────────────────┘
```

* **Not "Edge Compute"**: In engineering and IoT contexts, *Edge Compute* refers to physical hardware deployed outside data centers on-premises (e.g., factory inspection cameras, autonomous vehicle computers, NVIDIA Jetson boards, or AWS Snowball/Greengrass appliances).
* **Interactive Cloud Instance**: The SageMaker terminal runs inside a container hosted on an **EC2 virtual machine** inside an AWS data center. It functions as an **interactive head node** or **bastion workspace**.
* **Workload Scope**: For small-to-medium benchmarks like MNIST, this interactive instance has plenty of CPU, RAM, and GPU capacity to run data preprocessing and training in seconds. However, for massive models, large datasets (e.g., ImageNet), or high-throughput hyperparameter optimization, running long jobs inside the interactive terminal is an anti-pattern because the instance can be terminated, lacks elasticity, and cannot scale across multiple machines.

---

## 2. Mapping Traditional HPC (Slurm / Condor) to AWS Services

If you come from an academic, research, or scientific computing environment, you are likely familiar with job schedulers such as **Slurm Workload Manager** or **HTCondor**. AWS provides direct cloud-native equivalents for both paradigms:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HPC Schedulers vs. AWS                             │
├──────────────────────────┬──────────────────────────────────────────────────┤
│ Traditional On-Premises  │ Equivalent AWS Service                           │
├──────────────────────────┼──────────────────────────────────────────────────┤
│ Slurm (sbatch, srun)     │ AWS ParallelCluster (Native Slurm on EC2)        │
│ HTCondor / LSF / Grid    │ AWS Batch (Dockerized High-Throughput Queue)     │
│ Bare-Metal GPU Cluster   │ Amazon EC2 Cluster Placement Groups + EFA        │
│ Shared NFS Storage       │ Amazon FSx for Lustre / Amazon EFS               │
└──────────────────────────┴──────────────────────────────────────────────────┘
```

### A. AWS ParallelCluster *(Native Slurm in the Cloud)*
* **How it works**: AWS ParallelCluster is an AWS-supported open-source cluster management framework. It provisions a complete, authentic **Slurm cluster** on top of EC2 instances.
* **The Developer Experience**: 100% identical to an on-premises HPC supercomputer. You log into a head node via SSH, manage modules, inspect the queue with `squeue`, and submit distributed jobs using familiar `#SBATCH` scripts:
  ```bash
  #!/bin/bash
  #SBATCH --job-name=mnist-dist
  #SBATCH --nodes=4
  #SBATCH --ntasks-per-node=8
  #SBATCH --gres=gpu:8
  #SBATCH --partition=gpu-partition

  srun torchrun --nproc_per_node=8 scripts/train.py
  ```
* **Elastic Auto-Scaling**: Unlike static on-premises clusters, AWS ParallelCluster automatically spins up EC2 worker instances when jobs enter the Slurm queue and terminates them when the queue empties.
* **Storage Interconnect**: Nodes are connected to an **Amazon FSx for Lustre** filesystem delivering hundreds of gigabytes per second of shared throughput.

### B. AWS Batch *(High-Throughput / Condor Equivalent)*
* **How it works**: AWS Batch is a managed service that plans, schedules, and executes containerized batch workloads across dynamically provisioned EC2 or AWS Fargate instances.
* **How it feels**: Very similar to HTCondor. You define a container image (`Dockerfile`), package your command (e.g., `make train ARGS="..."`), and submit 100s or 1,000s of independent jobs to a queue. AWS Batch provisions the instances, pulls the Docker container, runs the job, and destroys the VM.
* **Best use case**: Embarrassingly parallel workloads, such as running a 100-trial hyperparameter sweep where each trial runs independently on its own single GPU.

---

## 3. The Industry Standard: SageMaker Managed Training Jobs

While AWS ParallelCluster is excellent for organizations with existing Slurm tooling, modern industry MLOps teams predominantly use **Amazon SageMaker Managed Training Jobs**.

Instead of managing a standing cluster, training compute is treated as **ephemeral, serverless infrastructure**:

```
                       1. Submit Job via Python SDK or CLI
                                      │
  [Developer Terminal] ───────────────┼────────────────────────┐
  (SageMaker Studio / Laptop)         │                        │
                                      ▼                        ▼
                       ┌─────────────────────────┐  ┌─────────────────────┐
                       │   Amazon S3 (Data)      │  │  ECR (Docker Image) │
                       │ s3://bucket/data/       │  │ mnist-pipeline:v1   │
                       └──────────┬──────────────┘  └──────────┬──────────┘
                                  │ FastFile Streaming         │
                                  ▼                            ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                   Ephemeral Multi-Node Cluster                         │
  │                                                                        │
  │   Node 0 (Rank 0) [8x A100]  ◄─── Elastic Fabric Adapter ───►  Node 1  │
  │            ▲                                (EFA)               ▲      │
  │            │                                                    │      │
  │            ▼                                                    ▼      │
  │   Node 2 (Rank 2) [8x A100]  ◄──────────────────────────────►  Node 3  │
  └────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ 2. Real-time telemetry to W&B
                                      │ 3. Checkpoints uploaded to S3
                                      ▼
             [Cluster Shuts Down Automatically Upon Completion]
```

### Key Advantages of Managed Training Jobs

1. **Zero Idle Cost**: You never pay for idle GPU instances. If an 8-node cluster trains for 42 minutes, you are billed for exactly 42 minutes. Once the training script exits (or crashes), AWS automatically tears down the instances.
2. **Managed Network Fabric (EFA)**: AWS **Elastic Fabric Adapter (EFA)** provides customized OS-bypass hardware interfaces on high-end instances (`p4de`, `p5`, `g5`), delivering up to 3,200 Gbps network bandwidth for ultra-low latency all-reduce gradient synchronization.
3. **Managed Spot Training**: By setting `use_spot_instances=True`, you can run training on surplus AWS capacity at up to a **70–90% cost reduction**. SageMaker handles spot interruption notices and resumes from your latest checkpoint in S3.
4. **Data Streaming (`FastFile` Mode)**: Instead of downloading hundreds of gigabytes of data before training starts, SageMaker mounts the S3 bucket directly into the container as POSIX-like files, streaming data on demand.

### Submitting a Distributed Job via Python SDK

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="scripts/train.py",
    source_dir=".",
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    framework_version="2.2.0",
    py_version="py310",
    instance_count=2,                # 2 machines
    instance_type="ml.g5.12xlarge",  # 4x NVIDIA A10G GPUs per machine (Total 8 GPUs)
    distribution={
        "torch_distributed": {"enabled": True}  # Configures torchrun automatically
    },
    hyperparameters={
        "train.epochs": 20,
        "train.lr": 0.001,
        "model": "transformer",
    },
)

# Launch the ephemeral cluster and stream logs
estimator.fit({"training": "s3://my-mnist-bucket/data/processed"})
```

---

## 4. Scaling the Pipeline: Adapting PyTorch for Distributed Training

To adapt this codebase for distributed training (whether running on AWS ParallelCluster, SageMaker, or raw EC2), PyTorch's native **DistributedDataParallel (DDP)** is used.

### Core Architectural Changes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Distributed Data Parallel (DDP)                        │
│                                                                             │
│                   Full Dataset (e.g. 60,000 images)                         │
│                                   │                                         │
│         ┌─────────────────────────┴─────────────────────────┐               │
│         ▼                                                   ▼               │
│   Rank 0: Batch A                                     Rank 1: Batch B       │
│         │                                                   │               │
│   Forward Pass (Local Model)                          Forward Pass          │
│   Loss & Backward (Local Gradients)                   Loss & Backward       │
│         │                                                   │               │
│         └──────────► All-Reduce Ring Gradient Sync ◄────────┘               │
│                                   │                                         │
│                     Synchronized Average Gradients                          │
│                                   │                                         │
│                     Optimizer Step on All Ranks                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 1. Initializing Process Groups in `train.py`
```python
import os
import torch
import torch.distributed as dist

def init_distributed() -> tuple[int, int, int]:
    """Initialize process group using environment variables provided by torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, local_rank, world_size
    return 0, 0, 1  # Single-process fallback
```

#### 2. Wrapping the Model in `DistributedDataParallel`
```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = build_model(cfg.model).to(local_rank)
if world_size > 1:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
```

#### 3. Sharding Data via `DistributedSampler`
In `src/mnist_pipeline/data.py`, ensure data batches are split without overlap:
```python
from torch.utils.data.distributed import DistributedSampler

train_sampler = DistributedSampler(
    train_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
) if world_size > 1 else None

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.data.batch_size,
    sampler=train_sampler,
    shuffle=(train_sampler is None),
)

# Inside the training loop:
for epoch in range(cfg.train.epochs):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)  # Ensures deterministic shuffle per epoch
```

#### 4. Rank-0 Logging & Checkpointing
In distributed runs, multiple processes run simultaneously. Only **Rank 0** should create directories, log to Weights & Biases, or write checkpoints to disk:
```python
is_main_process = (rank == 0)

if is_main_process:
    logger.info("Epoch %d: Loss = %.4f", epoch, loss.item())
    save_checkpoint(model.module, checkpoint_path)  # Note .module unwraps DDP
```

#### 5. Launching via `torchrun`
```bash
# Multi-GPU on single node (e.g. 4 GPUs):
torchrun --nproc_per_node=4 scripts/train.py

# Multi-Node (2 nodes, 8 GPUs total):
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --rdzv_id=mnist_job_101 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=node-0.internal:29500 \
  scripts/train.py
```

---

## 5. Architecture Comparison & Decision Matrix

| Dimension | Interactive Terminal (Current) | AWS ParallelCluster | AWS Batch | SageMaker Training Jobs |
| :--- | :--- | :--- | :--- | :--- |
| **Scheduler** | Bash / Manual terminal | **Slurm** (`sbatch`, `srun`) | AWS Batch Job Queue | Managed Ephemeral API |
| **User Familiarity** | Linux shell / IDE | High (Exact Slurm commands) | Moderate (Docker/JSON specs) | High for Python / MLOps |
| **Infrastructure Nature** | Persistent VM (until stopped) | Elastic Slurm cluster | Containerized fleet | 100% Ephemeral (serverless) |
| **Multi-Node Networking** | N/A (single node) | AWS EFA over Slurm | Standard VPC / Limited EFA | Native AWS EFA + NCCL |
| **Idle Cost Risk** | High if instance is left on | Medium (scales to 0 worker nodes) | Zero (fleets terminate) | **Zero (billed by the second)** |
| **Spot Instance Handling** | Manual | Slurm spot configuration | Managed Spot queues | Automated Spot Checkpointing |
| **Best Fit For** | Prototyping, debugging, smoke tests | Academic / HPC research labs | Embarrassingly parallel sweeps | Production pipelines & Foundation Models |

### Recommendation

* **For Current Development (`mnist-lab`)**: Stick with the **SageMaker Interactive Terminal / Local Compute**. MNIST is a lightweight benchmark (training completes in $< 30$ seconds on CPU/GPU), so distributed multi-node clusters add unnecessary orchestration overhead.
* **If Transitioning to Slurm Workflows**: Use **AWS ParallelCluster** if your lab or engineering team already maintains `.sbatch` scripts, shared user accounts, and Slurm monitoring tools.
* **If Building Production Large-Scale Pipelines**: Use **SageMaker Managed Training Jobs** with `torchrun` and `PyTorch Estimator`. It eliminates cluster administration, offers native EFA interconnect, and guarantees zero idle server costs.
