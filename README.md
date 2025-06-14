# MM-TSFlib
MM-TSFlib is an open-source library for multimodal time-series forecasting based on [Time-MMD](https://github.com/AdityaLab/Time-MMD/) dataset. We achieve multimodal time series forecasting tasks, including both short-term and long-term, by integrating time series models and language models. Our framework is illustrated in the figure. Please note that this Lib is only a first-step attempt toward the multimodal extension of TSF and does not represent the optimal solution. Many of the designs are naive, such as pre-stored matching and independent multimodal modeling approaches, which can be improved through better solutions, such as carefully designed retrieval matching and multimodal fusion.

<div align="center">
    <img src="https://github.com/AdityaLab/MM-TSFlib/blob/main/lib_overview_00.png" width="500">
</div>


 
## Usage

1. Install environment, execute the following command.

```
pip install -r environment.txt
```

2. Prepare Data. Our dataset is [Time-MMD](https://github.com/AdityaLab/Time-MMD/) dataset.
We provide preprocessed data in the ./data folder to accelerate training, particularly simplifying the text matching process.

2. Prepare for ClosedSource LLM. Our framework is already capable of integrating closed-source LLMs. To save costs, you should first use closed-source LLMs, such as GPT-3.5, to generate text-based predictions. We have provided specific preprocessing methods in the [[document/file](https://github.com/AdityaLab/MM-TSFlib/tree/main/data/DataPre_ClosedSourceLLM)]. We have also provided preprocessed data that can be directly used in `./data/` You can use any other closedsource llm to replace it.

3. Prepare for open-source LLMs. Our framework currently supports models such as LLAMA2, LLAMA3, GPT2, BERT, GPT2M, GPT2L, and GPT2XL, all available on Hugging Face. Please ensure you have your own Hugging Face token ready.

4. Train and evaluate model. We provide the example experiment script under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
#Conduct experiments on the health dataset using GPU 0, and utilize the 0th to 1st models.
bash ./scripts/week_health.sh.sh 0 1 0
```
- For electronic health record tasks (48h IHM prediction and 24h phenotype classification),
  use `scripts/run_ehr.py` with `--ehr_task ihm` or `--ehr_task pheno`. The script combines PatchTST with a Llama text encoder and reports AUROC/AUPRC/F1 metrics. Patch sizes (`--patch_len`, `--stride`) and tokenizer max length (`--max_seq_len`) can be configured via command line.
- You can set a list of model names, prediction lengths, and random seeds in the script for batch experiments. We recommend specifying `--save_name` to better organize and save the results.
- `--llm_model` can set as LLAMA2, LLAMA3, GPT2, BERT, GPT2M, GPT2L, GPT2XL, Doc2Vec, ClosedLLM. When using ClosedLLM, you need to do Step 3 at first.
- `--pool_type` can set as avg min max attention for different pooling ways of token. When `--pool_type` is set to attention, we use the output of the time series model to calculate attention scores for each token in the LLM output and perform weighted aggregation.


## Citation

If you find this repo useful, please cite our paper.

```
@misc{liu2024timemmd,
      title={Time-MMD: A New Multi-Domain Multimodal Dataset for Time Series Analysis}, 
      author={Haoxin Liu and Shangqing Xu and Zhiyuan Zhao and Lingkai Kong and Harshavardhan Kamarthi and Aditya B. Sasanur and Megha Sharma and Jiaming Cui and Qingsong Wen and Chao Zhang and B. Aditya Prakash},
      year={2024},
      eprint={2406.08627},
      archivePrefix={arXiv},
      primaryClass={id='cs.LG' full_name='Machine Learning' is_active=True alt_name=None in_archive='cs' is_general=False description='Papers on all aspects of machine learning research (supervised, unsupervised, reinforcement learning, bandit problems, and so on) including also robustness, explanation, fairness, and methodology. cs.LG is also an appropriate primary category for applications of machine learning methods.'}
}
```

## Contact
If you have any questions or suggestions, feel free to contact:
hliu763@gatech.edu
## Acknowledgement

This library is constructed based on the following repos:

https://github.com/thuml/Time-Series-Library/
