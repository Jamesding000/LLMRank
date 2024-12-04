# LLMRank

**LLMRank** aims to investigate the capacity of LLMs that act as the ranking model for recommender systems. [[paper]](https://arxiv.org/abs/2305.08845)

> Yupeng Hou†, Junjie Zhang†, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian McAuley, Wayne Xin Zhao. Large Language Models are Zero-Shot Rankers for Recommender Systems. ECIR 2024.

## 🛍️ LLMs as Zero-Shot Rankers

![](assets/model.png)

We use LLMs as ranking models in an instruction-following paradigm. For each user, we first construct two natural language patterns that contain **sequential interaction histories** and **retrieved candidate items**, respectively. Then these patterns are filled into a natural language template as the final instruction. In this way, LLMs are expected to understand the instructions and output the ranking results as the instruction suggests.

## 🚀 Quick Start

1. Write your own OpenAI API keys into [`llmrank/openai_api.yaml`](https://github.com/RUCAIBox/LLMRank/blob/master/llmrank/openai_api.yaml).
2. Unzip dataset files.
    ```bash
    cd llmrank/dataset/ml-1m/; unzip ml-1m.inter.zip
    cd llmrank/dataset/Games/; unzip Games.inter.zip
    ```
    For data preparation details, please refer to [[data-preparation]](llmrank/dataset/data-preparation.md).
3. Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```
4. Evaluate ChatGPT's zero-shot ranking abilities on ML-1M dataset.
    ```bash
    cd llmrank/
    python evaluate.py -m Rank
    ```

## 🔍 Key Findings

> Please click the links below each "Observation" to find the code and scripts to reproduce the results.

### Observation 1. LLMs struggle to perceive order of user historie, but can be triggered to perceive the orders

LLMs can utilize historical behaviors for personalized ranking, but *struggle to perceive the order* of the given sequential interaction histories.

By employing specifically designed promptings, such as recency-focused prompting and in-context learning, *LLMs can be triggered to perceive the order* of historical user behaviors, leading to improved ranking performance.

<div align="center"> 
<img src='assets/tab-2.png' width="75%">
</div>


**Code is here ->** [[reproduction scripts]](scripts/ob1-struggle-to-perceive-order-but-can-be-triggered.md)

### Observation 2. Biases exist in using LLMs to rank

LLMs suffer from position bias and popularity bias while ranking, which can be alleviated by specially designed prompting or bootstrapping strategies.

<div align="center"> 
<img src='assets/2-biases-exist-in-using-llms-to-rank.png' width="75%">
</div>

**Code is here ->** [[reproduction scripts]](scripts/ob2-llms-suffer-from-position-bias-and-popularity-bias.md)


### Observation 3. Promising zero-shot ranking abilities

LLMs have promising zero-shot ranking abilities, ...

<div align="center"> 
<img src='assets/tab-3.png' width="75%">
</div>

..., especially on candidates retrieved by multiple candidate generation models with different practical strategies.

<div align="center"> 
<img src='assets/tab-4.png' width="70%">
</div>

**Code is here ->** [[reproduction scripts]](scripts/ob3-zero-shot-abilities.md)



## 🌟 Acknowledgement

Please cite the following paper if you find our code helpful.

```bibtex
@inproceedings{hou2024llmrank,
  title={Large Language Models are Zero-Shot Rankers for Recommender Systems},
  author={Yupeng Hou and Junjie Zhang and Zihan Lin and Hongyu Lu and Ruobing Xie and Julian McAuley and Wayne Xin Zhao},
  booktitle={{ECIR}},
  year={2024}
}
```

The experiments are conducted using the open-source recommendation library [RecBole](https://github.com/RUCAIBox/RecBole).

We use the released pre-trained models of [UniSRec](https://github.com/RUCAIBox/UniSRec) and [VQ-Rec](https://github.com/RUCAIBox/VQ-Rec) in our zero-shot recommendation benchmarks.

Thanks [@neubig](https://github.com/neubig) for the amazing implementation of asynchronous dispatching OpenAI APIs. [[code]](https://gist.github.com/neubig/80de662fb3e225c18172ec218be4917a)


## Below is for 11667 Project 6

# LLMRank: Fine-tuning and Evaluation Framework

This repository provides a framework for fine-tuning and evaluating LLM models on custom datasets. 

## Table of Contents
1. [Setup](#setup)
2. [Downloading Pretrained Models](#downloading-pretrained-models)
3. [Fine-tuning](#fine-tuning)
4. [Evaluation](#evaluation)
5. [Configuring Models and Datasets](#configuring-models-and-datasets)

---

## Setup
Navigate to the project llmrank directory:
`cd llmrank`

Install the required dependencies:
`pip install -r requirements.txt`

---

## Downloading Pretrained Models
To download pretrained models, run:
`python get_pretrained_models.py`

---

## Fine-tuning
To fine-tune a pretrained model, use the following command:
`python finetune.py -m MODEL_NAME -d DATASET -g GPU_ID`

### Example:
`python finetune.py -m llama3 -d ml-1m-full -g 0`

### Supported Model Names:
- llama3
- gpt-neo

Modify the fine-tuning configuration files in the `configs` directory to customize the training process.

---

## Evaluation
To evaluate a fine-tuned model, use the following command:
`python evaluate_llm.py -m MODEL_NAME -d DATASET -p FINETUNED_MODEL_PATH`

### Example:
`python evaluate_llm.py -m llama3 -d ml-1m -p models/Llama-3.2-1B-sft`

---

## Configuring Models and Datasets
Fine-tuning and evaluation configurations can be modified in the `configs` directory. 
Adjust parameters like learning rate, batch size, and model settings to suit your requirements.

---