# CHBias: Bias Evaluation and Mitigation of Chinese Conversational Language Models

Jiaxu Zhao, Meng Fang, Zijing Shi, Yitong Li, Ling Chen, Mykola Pechenizkiy  

[Paper Link](https://aclanthology.org/2023.acl-long.757.pdf)

Abstract

Warning: This paper contains content that may be offensive or upsetting. Pretrained conversational agents have been exposed to safety issues, exhibiting a range of stereotypical human biases such as gender bias. However, there are still limited bias categories in current research, and most of them only focus on English. In this paper, we introduce a new Chinese dataset, CHBias, for bias evaluation and mitigation of Chinese conversational language models. Apart from those previous well-explored bias categories, CHBias includes under-explored bias categories, such as ageism and appearance biases, which received less attention. We evaluate two popular pretrained Chinese conversational models, CDial-GPT and EVA2.0, using CHBias. Furthermore, to mitigate different biases, we apply several debiasing methods to the Chinese pretrained models. Experimental results show that these Chinese pretrained models are potentially risky for generating texts that contain social biases, and debiasing methods using the proposed dataset can make response generation less biased while preserving the models’ conversational capabilities.

## Requirement

The library requires 

Python 3.7

PyTorch v1.10.2

Transformers 3.3.0

## Scripts

The script is in the src/run_all.sh

## Dataset

We provide Four bias datasets (gender, orientation, age, and appearance):

data/age/

data/gender/

data/appearance/

data/orientation/

## Acknowledgement
The implementation is heavily based on Soumya Barikeri' implementation for experiments on the [RedditBias](https://github.com/SoumyaBarikeri/debias_transformers).

## Citation

If you use this library in a research paper, please cite this repository.

```
@inproceedings{zhao-etal-2023-chbias,
    title = "{CHB}ias: Bias Evaluation and Mitigation of {C}hinese Conversational Language Models",
    author = "Zhao, Jiaxu  and
      Fang, Meng  and
      Shi, Zijing  and
      Li, Yitong  and
      Chen, Ling  and
      Pechenizkiy, Mykola",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.757",
    doi = "10.18653/v1/2023.acl-long.757",
    pages = "13538--13556",
   }
```

