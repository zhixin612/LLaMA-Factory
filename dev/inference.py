import os
import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM

llama2_origin = r'/root/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots' \
                r'/8a0442e81540efaeb1a0fe3e95477b5e0edfd423'
llama2_finetune = r'/root/llm/finetune/dev/saves/LLaMA2-7B/llama2-7b_alpaca_zh_lora'
llama2_predict = None


def finetune_compare():
    inputs = [
        'Hello, introduce yourself please',
        'How to accelerate the LLM training process?',
        '你好，做个自我介绍吧',
        '当前有哪些LLM推理优化技术？'
    ]

    tokenizer = LlamaTokenizer.from_pretrained(llama2_origin)
    model = LlamaForCausalLM.from_pretrained(llama2_origin)

    tokenizer.save_pretrained(llama2_finetune)
    model.save_pretrained(llama2_finetune)

