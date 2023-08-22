# CHBias: Bias Evaluation and Mitigation of Chinese Conversational Language Models

Jiaxu Zhao, Meng Fang, Zijing Shi, Yitong Li, Ling Chen, Mykola Pechenizkiy  

[Paper](https://aclanthology.org/2023.acl-long.757.pdf)

Abstract

Warning: This paper contains content that may be offensive or upsetting. Pretrained conversational agents have been exposed to safety issues, exhibiting a range of stereotypical human biases such as gender bias. However, there are still limited bias categories in current research, and most of them only focus on English. In this paper, we introduce a new Chinese dataset, CHBias, for bias evaluation and mitigation of Chinese conversational language models. Apart from those previous well-explored bias categories, CHBias includes under-explored bias categories, such as ageism and appearance biases, which received less attention. We evaluate two popular pretrained Chinese conversational models, CDial-GPT and EVA2.0, using CHBias. Furthermore, to mitigate different biases, we apply several debiasing methods to the Chinese pretrained models. Experimental results show that these Chinese pretrained models are potentially risky for generating texts that contain social biases, and debiasing methods using the proposed dataset can make response generation less biased while preserving the models’ conversational capabilities.
