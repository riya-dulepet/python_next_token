# Python Next Token

## Requirements installation
1) cd python_next_token
2) pip install -r requirements.txt

## Data Preparation
1) python convert.py --segment_len 256 --stride 10 --dev_size 0.1
This will create a directory dataset/source_code/json where the train and val data will be stored.
2) For the purposes of this assignment I used 50% of the data found at: https://huggingface.co/datasets/ArtifactAI/arxiv_python_research_code

## Finetune Model
python train.py [all the arguments have default, take a look at train.py to see which arguments you want to change]
The model checkpoint will be stored at stored_model/0_GPTSingleHead

![alt text](train_loss.png)
![alt text](val_loss.png)
![alt text](val_perplexity.png)

## Test the Model
python predict.py --model_path stored_model/0_GPTSingleHead (there are other optional arguments for prompt length)

## Inspiration
https://github.com/wangcongcong123/auto_coding 

## Approach 
Initially I tried training from scratch with GPT2 tokenizer however was running into issues with compute so then I turned to finetuning. During that time, I discovered the repository I linked above. The repository enables finetuning of GPT2 models (distill, medium, large). For the purpose of compute and time given I had invested tons of time trying to figure out the training from scratch of GPT2, I decided to finetune the distill model. I changed the convert.py file they had to fit my purposes of loading from a Huggingface dataset and enabling parallel data processing. In terms of the model architecture there is a GPTSingleHead Class and EmptyHeads Class:
1) GPTSingleHead Class: Enhances the base GPT-2 model (GPT2LMHeadModel) by wrapping it together with the GPT2Tokenizer, providing a unified interface for model operations such as tokenization and special token handling. Accepts parameters for the model path, maximum sequence length, case sensitivity, and special tokens to provide configuration flexibility right from instantiation. This class manages tokenization and integrates special tokens into the tokenizer, adapts the model embedding layer to accommodate new tokens, implements the forward pass, and includes utility methods for saving and reloading configurations.
2) EmptyHeads Classs: Acts as a placeholder or a basic structure to potentially host additional neural network layers or mechanisms, which can be used to further process the outputs of the GPTSingleHead. Contains minimal functionality and can be seen as a template for extending the model with task-specific heads (e.g., feedforward layers, classification heads).

I ran on 50% of the data because of compute and time constraints. I also only ran 1 epoch and had time permit I would have run 2-3 more depending on loss. We can see that the perplexity decreases consistently on the validation which is a good sign. Both the val and training loss also is decreasing and the model could benefit from a bit more training as the loss has not completely stabilized but is close to it. 

## Training from Scratch Attempt:
https://drive.google.com/drive/folders/1RQK14zfXFEzd9p14rYjFDNmyhBRdxiOx?usp=sharing 
