# Large-Language-Models-for-DNA-Sequence-Classification

## Introduction

The Large-Language-Models-for-DNA-Sequence-Classification project aims to explore the flexibility of using large language models for analyzing DNA sequence.



### Model

You can find the fine-tuned model for antimicrobial drug classification on Hugging Face at the following link:

[Quantized Fine-Tuned LLama 3.1 8B Model for AMR Drug Classification](https://huggingface.co/hyoo14/Meta-Llama-3.1-8B-Instruct-bnb-4bit_DNA_AMR)


The fine-tuned model for mapping text responses to drug classification labels is available on Hugging Face at the following link:

[Quantized Fine-Tuned LLama 3.1 8B Model for Converting Answer Text to Labels](https://huggingface.co/hyoo14/Meta-Llama-3.1-8B-Instruct-bnb-4bit_AnswerToLabel)


<br/>


### Usage

* Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```





<br/>
<br/>


### Foundation Model: Quantized LLama 3.1 

This project utilized the [Unsloth](https://github.com/unslothai/unsloth) repository's implementation of **LLama 3.1 Quantization** for this project. Their model, designed for efficient inference with reduced memory usage, was crucial in fine-tuning the tasks related to drug classification and other NLP applications.

You can find more details and access the quantized models through their official repository: [Unsloth on GitHub](https://github.com/unslothai/unsloth).

