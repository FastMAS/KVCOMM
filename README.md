[![arXiv](https://img.shields.io/badge/arXiv-2510.12872-b31b1b.svg)](https://arxiv.org/abs/2510.12872)
[![GitHub issues](https://img.shields.io/github/issues/HankYe/KVCOMM)](https://github.com/HankYe/KVCOMM/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/HankYe/KVCOMM/pulls)

<div align="center">
<h1 style="text-align: center; font-size: 2.5rem; font-weight: bolders">
KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems
</h1>
</div>

<p align="center">
  <h3 align="center">🖥️ Demo of TTFT speedup under a two-agent setting</h3>
  <video src="https://github-production-user-asset-6210df.s3.amazonaws.com/39550579/501226767-9e209eaf-8c58-42c9-ac0f-2cc155e806e8.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20251015%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251015T010601Z&X-Amz-Expires=300&X-Amz-Signature=ce9f372145bff4c622536050a1871f555bf3cda35bf20ed8f8e4e5386bc617a6&X-Amz-SignedHeaders=host"
         controls
         muted
         playsinline>
    <a href="https://github-production-user-asset-6210df.s3.amazonaws.com/39550579/501226767-9e209eaf-8c58-42c9-ac0f-2cc155e806e8.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20251015%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251015T010601Z&X-Amz-Expires=300&X-Amz-Signature=ce9f372145bff4c622536050a1871f555bf3cda35bf20ed8f8e4e5386bc617a6&X-Amz-SignedHeaders=host">Demo of TTFT speedup under a two-agent setting.</a>
  </video>
</p>

<p align="center">
    <br>
    <img src="assets/comparison.svg"/>
    <br>
<p>

## 🔖 Introduction
This is the up-to-date official implementation of KVCOMM in the paper, [**KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems**](https://arxiv.org/abs/2510.12872). KVCOMM is a novel adaptive serving paradigm for LLM-based Multi-agent Systems, which highlights the **online cross-context KV-cache resuse** for the shared text among dependent LLM agents. KVCOMM estimates and adjusts KV-caches for shared content by referencing a pool of cached examples—termed ***anchors***—that store observed cache deviations under varying prefixes. The anchor pool is maintained and updated online, allowing dynamic adaptation to distinct user requests and context structures. KVCOMM achieves over $70$\% reuse rate across diverse multi-agent workloads, including retrieval-augmented generation, math reasoning, and collaborative coding tasks, all without quality degradation. Particularly, when each fully-connected agent receives 1K input tokens with 512 prefix tokens and 512 output tokens under a five-agent setting, KVCOMM achieves up to $7.8\times$ speedup compared to the standard prefill pipeline, reducing TTFT from $\sim430$ ms to $\sim55$ ms on a single H100 GPU.

🌟 KVCOMM offers the following insights:
| | |
|---|---|
| <img src="assets/kv_proximity.jpg" width="100%" alt="KV proximity"/> | <img src="assets/offset_proximity.jpg" width="100%" alt="Offset proximity"/> |

> 1. **KV Proximity**: Tokens that stay closer in embedding space have closer KV vectors across layers (left).
> 2. **Offset Proximity**: Under the same prefix context change, the offsets for similar tokens still stay close (right).

<p align="center">
    <br>
    <img src="assets/framework.jpg"/>
    <br>
<p>

🌟 And further proposes an anchor-based KV-cache Communication Module, which contains:
> - **Anchor Matching**: locates the nearest anchors for the requested segment (See [kvcomm_engine.py:KVCOMMEngine._select_anchor_indices](https://github.com/HankYe/KVCOMM/blob/main/KVCOMM/llm/kvcomm_engine.py#L806)).
> - **Offset Approximation**: predicts the offset by weighting the anchors' stored deviations (See [kvcomm_engine.py:KVCOMMEngine.offset_kv_cache_pair](https://github.com/HankYe/KVCOMM/blob/main/KVCOMM/llm/kvcomm_engine.py#L868)).
> - **Anchor Prediction**: determines whether newly-generated KV-caches, e.g., responses, user inputs, etc., could be shared or treated as new anchor (See [kvcomm_engine.py:KVCOMMEngine.predict_as_anchor](https://github.com/HankYe/KVCOMM/blob/main/KVCOMM/llm/kvcomm_engine.py#L991)).

## ⚙️ Installation
```bash
$ git clone https://github.com/HankYe/KVCOMM
$ cd KVCOMM
$ conda create -n kvcomm python==3.10
$ conda activate kvcomm
$ pip install -r requirements.txt
```

## 🔬 Experiments
Several examples are provided to run the main experiments in the paper, including three benchmarks, i.e., MMLU, GSM8K, HumanEval. The instructions are as follows, where 
- `--mode` is the connection mode between agents, 
- `--agent_names` assign agent class to each agent as defined in `KVCOMM/agents`, 
- `--agent_nums` defines the number of each unique agent class, which should be in the same size of `--agent_names`, 
- `--llm_name` gives the specific model to serve all agents, 
- `--execution_mode` determine whether the experiment runs in `default` (dense prefill) or `allow_kv_reuse` (KVCOMM) mode, 
- `--kv-threshold` is the entropy threshold $\gamma$ to estimate the KV-cache sharability (defaut = 0.3), 
- `--kv-max-anchor-num` is the maximum number of stored anchors $\mathcal{V}$ for each placeholder id (defaut = 20), 
- `--kv-window-size` is the anchor number that will be regarded as dropping candidates for the update of the anchor pool (defaut = 5). 

The results are saved in `--output_dir`. In our experiments reported in the paper, we tuned the agent number and assigned the same agent class for all agents in each benchmark.

<details on>
<summary>MMLU</summary>

To run the baseline on MMLU under the 3-agent setting with Llama-3.1-8B-Instruct, the script is:
```bash
python experiments/run_mmlu.py --mode FullConnected --agent_names AnalyzeAgent --agent_nums 3 --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/mmlu_default --execution_mode default
```
To run the KVCOMM method on MMLU under the 3-agent setting with Llama-3.1-8B-Instruct, the script is:
```bash
python experiments/run_mmlu.py --mode FullConnected --agent_names AnalyzeAgent --agent_nums 3 --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/mmlu_kvcomm --execution_mode allow_kv_reuse --kv-threshold 0.3 --kv-max-anchor-num 20 --kv-window-size 5
```
</details>

<details on>
<summary>GSM8K</summary>

To run the baseline on GSM8K under the 3-agent setting with Llama-3.1-8B-Instruct, the script is:
```bash
python experiments/run_gsm8k.py --mode FullConnected --agent_names MathSolver --agent_nums 3 --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/gsm8k_default --execution_mode default
```
To run the KVCOMM method on GSM8K under the 3-agent setting with Llama-3.1-8B-Instruct, the script is:
```bash
python experiments/run_gsm8k.py --mode FullConnected --agent_names MathSolver --agent_nums 3 --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/gsm8k_kvcomm --execution_mode allow_kv_reuse --kv-threshold 0.3 --kv-max-anchor-num 20 --kv-window-size 5
```
</details>
<details on>
<summary>HumanEval</summary>

To run the baseline on HumanEval under the 3-agent setting with Qwen2.5-Coder-7B-Instruct, the script is:
```bash
python experiments/run_humaneval.py --mode FullConnected --agent_names CodeWriting --agent_nums 3 --llm_name Qwen/Qwen2.5-Coder-7B-Instruct --output_dir ./runs/humaneval_default --execution_mode default
```
To run the KVCOMM method on HumanEval under the 3-agent setting with Qwen2.5-Coder-7B-Instruct, the script is:
```bash
python experiments/run_humaneval.py --mode FullConnected --agent_names CodeWriting --agent_nums 3 --llm_name Qwen/Qwen2.5-Coder-7B-Instruct --output_dir ./runs/humaneval_kvcomm --execution_mode allow_kv_reuse --kv-threshold 0.3 --kv-max-anchor-num 20 --kv-window-size 5
```
</details>

## 🪜 Next Step
Integration with the existing KV management frameworks, e.g., [LMCache](https://github.com/LMCache/LMCache) to support modern LLM-based serving systems.

## 📖 Citation

```bibtex
@misc{kvcomm25ye,
  author = {Hancheng Ye and Zhengqi Gao and Mingyuan Ma and Qinsi Wang and Yuzhe Fu and Ming-Yu Chung and Yueqian Lin and Zhijian Liu and Jianyi Zhang and Danyang Zhuo and Yiran Chen},
  title = {KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems},
  year = {2025},
  booktitle = {The Thirty-Ninth Annual Conference on Neural Information Processing Systems}
}
```



## 🍀 Acknowledgements
We greatly acknowledge the authors of *AgentPrune* and *GPTSwarm* for their open-source codes. Visit the following links to access their more contributions.

- [AgentPrune](https://github.com/yanweiyue/AgentPrune)
- [GPTSwarm](https://github.com/metauto-ai/GPTSwarm)
