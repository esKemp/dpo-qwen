# DPO on Qwen

A from-scratch implementation of Direct Preference Optimization, applied to Qwen2.5-0.5B-Instruct.

SF Phase build — rep 1. Part of a portfolio building toward RL / post-training depth
(DPO → PPO/GRPO → financial time-series VAE).

## Status

- Environment + model forward pass
- [] Load preference dataset
- [] DPO training loop
- [] Verify the policy learns the preference signal

## Stack

PyTorch · transformers · trl · datasets — Apple Silicon (MPS).