---
license: apache-2.0
---

<div align="center">

# *TranX-Adapter*:<br> Bridging Artifacts and Semantics within MLLMs for Robust AI-generated Image Detection

<a href="https://arxiv.org/abs/2602.21716" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2602.21716-red?logo=arxiv" height="25" />
</a>
<a href="https://huggingface.co/collections/DreamMr/tranxadapter" target="_blank">
    <img alt="HF Models: TranX-Adapter" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Models-TranX--Adapter-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://huggingface.co/datasets/DreamMr/TranXAdapter-Dataset" target="_blank">
    <img alt="HF Data: TranXAdapter-Data" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Data-TranX--Adapter Data-3cb371?color=3cb371&logoColor=white" height="25" />
</a>

<div style="font-family: charter;">
    Wenbin Wang<sup>1</sup>,
    Yuge Huang<sup>2</sup>,
    Jianqing Xu<sup>2</sup>,
    Yue Yu<sup>2</sup>,
    Jiangtao Yan<sup>2</sup>,
    <br>
    Shouhong Ding<sup>2</sup>,
    Pan Zhou<sup>3</sup>,
    Yong Luo<sup>1</sup>
</div>

<div style="font-family: charter;">
    <sup>1</sup>Wuhan University&nbsp;&nbsp;
    <sup>2</sup>Tencent YouTu Lab&nbsp;&nbsp;
    <sup>3</sup>Singapore Management University
</div>


</div>

## News

[2026.06.03] TranX-Adapter code is available! Additionally, we have also open-sourced the [trained models](https://huggingface.co/collections/DreamMr/tranxadapter) along with the corresponding [training and evaluation data](https://huggingface.co/datasets/DreamMr/TranXAdapter-Dataset).

[2026.05.01] Our paper was accepted to ICML 2026! 🎉

[2026.02.25] We released the ArXiv paper. 🚀

## TL;DR

While prior work improves AIGI detection by combining artifact and semantic features in MLLMs, we find that artifact features often suffer from high intra-feature similarity, causing uniform attention and ineffective fusion. To address this attention dilution problem, we propose **TranX-Adapter**, a lightweight fusion module that combines task-aware optimal-transport fusion and cross-attention-based X-Fusion to enable bidirectional interaction between artifact and semantic features.


## 🔧 Installation

1. Clone this repository and navigate into the codebase
    ```bash
    git clone https://github.com/DreamMr/TranX-Adapter.git
    cd TranX-Adapter
    ```

2. Install Packages
    ```bash
    bash install.sh
    ```

    When the environment is created successfully, you will see:

    `Conda environment name tranxadapter has been created🎉. Now you can run "conda activate tranxadapter"`

## 📦 Preparation

1. Download the datasets

    - Training Data: [GenImage](https://genimage-dataset.github.io/), [RRDataset](https://arxiv.org/abs/2509.09172), [BFree](https://github.com/grip-unina/B-Free/tree/main/training_data) (SD2.1_selfconditioned_origBG.zip (ai) and COCO_real_512.zip (real)).

    - Evaluation Data: [GenImage](https://genimage-dataset.github.io/), [Chameleon](https://github.com/shilinyan99/AIDE/issues/7), [RRDataset](https://arxiv.org/abs/2509.09172) 

    - [Our constructed VQA dataset](https://huggingface.co/datasets/DreamMr/TranXAdapter-Dataset/tree/main)

    Download the above data to `./Dataset`, with the structure as follows:

    ```text
    Dataset/
    ├── TranXAdapter-Dataset/
    │   ├── training/
    │   │   └── GenImage_Sdv1d4.jsonl
    │   │   └── GenImageAll.jsonl
    │   │   └── RRDataset.jsonl
    │   │   └── BFP.jsonl
    │   ├── evaluation/
    │   │   └── Chamelon.tsv
    │   │   └── GenImage.tsv
    │   │   └── RR.tsv
    ├── GenImage/
    │   ├── ADM
    │   ├── test
    │   ├── ...
    ├── Chameleon/test/
    │   ├── 0_real
    │   ├── 1_fake
    ├── RRDataset_final/
    │   ├── original
    │   ├── redigital
    │   ├── ...
    ├── RRDataset_original_train_val
    │   ├── train
    │   ├── val
    ├── BFP
    │   ├── ai
    │   ├── real
    ```
    
2. Process Data

    Run `python preprocess_data.py` to replace the image paths in JSONL/CSV files with absolute paths.

    **Note: You need to copy the MD5 values corresponding to the CSV files into  DATASET_MD5 in ./VLMEvalKit/vlmeval/dataset/aigc_detection.py**


## 🏋️ Training

1. Merge TranX-Adapter into the MLLM

    First, TranX-Adapter needs to be merged into the MLLM so that it can be directly loaded with `from_pretrained()`. We provide merge scripts (`./llavanpr/merge_model.py` and `./qwen3vlnpr/merge_model.py`) as well as the merged models: [DreamMr/TranXAdapter-LLaVA-next-mistral7B-v0](https://huggingface.co/DreamMr/TranXAdapter-LLaVA-next-mistral7B-v0), [DreamMr/TranXAdapter-Qwen3VL2B-v0](https://huggingface.co/DreamMr/TranXAdapter-Qwen3VL2B-v0), [DreamMr/TranXAdapter-Qwen3VL4B-v0](https://huggingface.co/DreamMr/TranXAdapter-Qwen3VL4B-v0)


2. Start training
    take training Qwen3Vl-2B on GenImage Sdv1.4 as an example:

    ```bash
    cd ms-swift/scripts/training
    bash train_qwen3vl_Chameleon.sh
    ```

    

    i.  If you want to train on RRDataset, you need to set the input image resolution to 512x512 (`./ms-swift/swift/llm/template/template/qwen.py line 637` and `./ms-swift/swift/llm/template/templatellava.py line192`).

    ii. We found that if the model is trained directly on GenImage Sdv1.4, the MLLM tends to overfit to the input image resolution. Therefore, we recommend training with real and fake images that have the same resolution. We use the [BFree](https://github.com/grip-unina/B-Free/tree/main/training_data) (SD2.1_selfconditioned_origBG.zip and COCO_real_512.zip) to prevent the model from overfitting to image resolution. We recommend downloading the data from the [official link](https://github.com/grip-unina/B-Free/tree/main/training_data).

    iii. We found that MLLM training converges quickly and also overfits rapidly. Therefore, we recommend using a checkpoint from the middle of training.

## 📈 Evaluation

1. Modify the LMUData in `./VLMEvalKit/scripts/run_task.sh`

    You need to modify `LMUData` to the absolute path of `Dataset`.

2. Modify `DATASET_URL` and `DATASET_MD5` in `./VLMEvalKit/vlmeval/dataset/aigc_detection.py`.

    Replace `DATASET_URL` with the absolute path of the CSV file, and fill in `DATASET_MD5` with the MD5 value [computed earlier](#📦preparation).

3. Run code
    ```bash
    cd VLMEvalKit/scripts
    bash run_task.sh
    ```



## 📧 Contact
- Wenbin Wang: [wangwenbin97@whu.edu.cn](wangwenbin97@whu.edu.cn)

## 🖊️ Citation

If you use TranX-Adapter in your research, please cite our work:
```
@inproceedings{wang2026tranx,
  title={TranX-Adapter: Bridging Artifacts and Semantics within MLLMs for Robust AI-generated Image Detection},
  author={Wang, Wenbin and Huang, Yuge and Xu, Jianqing and Yu, Yue and Yan, Jiangtao and Ding, Shouhong and Zhou, Pan and Luo, Yong},
  booktitle={Forty-third International Conference on Machine Learning},
  url={https://arxiv.org/abs/2602.21716}
}
```

## 🙏 Acknowledgement

- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
- [ms-swift](https://github.com/modelscope/ms-swift)
