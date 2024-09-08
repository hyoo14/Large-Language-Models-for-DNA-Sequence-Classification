# Large-Language-Models-for-DNA-Sequence-Classification

## Introduction

The Large-Language-Models-for-DNA-Sequence-Classification project aims to explore the flexibility of using large language models for analyzing DNA sequence.



### Model

You can find the fine-tuned model for antimicrobial drug classification on Hugging Face at the following link:

[Quantized Fine-Tuned LLama 3.1 8B Model for AMR Drug Classification](https://huggingface.co/hyoo14/Meta-Llama-3.1-8B-Instruct-bnb-4bit_DNA_AMR)


The fine-tuned model for mapping text responses to drug classification labels is available on Hugging Face at the following link:

[Quantized Fine-Tuned LLama 3.1 8B Model for Converting Answer Text to Labels](https://huggingface.co/hyoo14/Meta-Llama-3.1-8B-Instruct-bnb-4bit_AnswerToLabel)


<br/>

### Colab Notebook

You can find the Colab notebook at the following link:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1w4BXSYMP0uyIVepgZI2OTzA9hfqewia6?usp=sharing]







<br/>



### Usage

* Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```


* Fine-tune the model:

    ```sh
    python train.py
    ```

* Inference with the model:

    ```sh
    python inference.py
    ```





<br/>
<br/>


### Foundation Model: Quantized LLama 3.1 

This project utilized the [Unsloth](https://github.com/unslothai/unsloth) repository's implementation of **LLama 3.1 Quantization** for this project. Their model, designed for efficient inference with reduced memory usage, was crucial in fine-tuning the tasks related to drug classification and other NLP applications.

You can find more details and access the quantized models through their official repository: [Unsloth on GitHub](https://github.com/unslothai/unsloth).

