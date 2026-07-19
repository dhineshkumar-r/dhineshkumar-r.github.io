## Overview

This guide helps configure model parallelism dimensions for Megatron-based training jobs and manage GPU memory methodically. It covers constraints for validating configurations, tuning guidelines for common scenarios. As there can be different implementations of a given model parallelism technique, this document specifically focuses on implementations relevant to Megatron.

## Background

In essence, model parallelism techniques enable large-scale training and inference on constrained infrastructure(GPU memory).

  - Pipeline Parallelism (PP)

    Partitions the model parameters and optimizer states in layers. Layers of the model are grouped into stages to form a pipeline where the data samples flow sequentially between stages. For example, an LLM with 48 layers and PP = 4 partitions the model across 4 GPUs with each containing one stage (12 layers) of the pipeline.

  - Tensor Parallelism (TP)

    Partitions parameters and activations of modules within a layer. Megatron's implementation of TP partitions Attention and MLP modules of a transformer block. The remaining layers, LayerNorm and Dropout, demand a considerable amount of 
    activation memory when input sequences are long. To further partition activations of these layers, Sequence Parallelism (SP) is employed. Note that in the context of Megatron, SP does not partition the input.

  - Context Parallelism (CP)
    
    Especially useful for long context training, CP partitions input sequence across CP ranks, thus effectively reducing the activation memory per GPU. 

  - Data Parallelism (DP)

    While the previously discussed strategies aim to divide parameters and activations, DP aims to multiply model copies to scale training. DP also influences the way samples per batch are processed during training. For example, with num_gpus=4 and DP=2, data samples of a batch are divided by 2 (model copies) and processed separately. Note that each data-parallel GPU will have a copy of the whole model. When a model does not fit in 1 GPU, other model parallelization techniques discussed above come into play to divide the model parameters within a DP group.


  - Expert Parallelism (EP)

    This is relevant to Mixture of Experts (MoE) LLMs. Contrary to dense LLMs, each transformer layer contains multiple MLP modules. EP, specifically expert model parallelism, determines the way expert units are partitioned across data-parallel GPUs. Tensors of each expert can further be partitioned through expert tensor parallelism.


## Constraints

The following are some important constraints to keep in mind when tuning model parallelism parameters:

- `world_size = num_nodes * num_gpus_per_node = TP x PP x CP x DP`
- **DP** — Not explicitly set in the config, but inferred: world_size / (TP x PP x CP).
- **PP** — must evenly divide the number of model layers.
- **EP** — must be <= # data-parallel GPUs.
- **Global batch size** — must be divisible by DP degree.

## Tuning Guidelines

The following are some practical guidelines for tuning model parallelism degrees.

* For MoEs, tuning EP is more useful than TP, as expert MLP blocks consume the most memory and TP is limited to attention modules.
* For longer sequence training, tune CP over SP. 
* To accelerate training, increase DP. This can be done either by adding more GPUs (increasing world_size) or by reducing other parallelism dimensions.
* Increase PP consciously, as it introduces a pipeline bubble that impacts GPU utilization. PP does not help reduce the memory required for activations, and idle time between micro-batches affects GPU utilization.

Complementary to model parallelism strategies, the following techniques further help manage GPU memory and address OOMs: 
* Activation Checkpointing / Recomputation
  
  The idea here is to cache activations of specific layers (as opposed to all layers) during the forward pass and reuse them to compute activations of remaining layers during the backward pass [6].
* Distributed Optimizer
  
  This feature allows the optimizer state of a model to be partitioned across GPUs of a given DP group [7]. Without this, the optimizer state (the most memory-intensive component) of the model is stored in each GPU.

## Additional Resources

1. [Megatron-LM](https://arxiv.org/pdf/1909.08053)
2. [Megatron FSDP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/megatron_fsdp.html)
3. [Megatron Sequence Parallelism](https://arxiv.org/pdf/2205.05198)
4. [Parallelisms (NeMo docs)](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html)
5. [Megatron Context Parallel](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
6. [Activation Recomputation](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/activation-recomputation.html)
7. [Distributed Optimizer](https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/features/dist_optimizer.html)
8. [ST-MoE](https://arxiv.org/pdf/2202.08906)
