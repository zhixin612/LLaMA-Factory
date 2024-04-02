import os
import re
import tqdm
import json
import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer.json, tokenizer.model, tokenizer_config.json,
# special_tokens_map.json, config.json, generation_config.json,
# model-00001-of-00002.safetensors, model-00002-of-00002.safetensors, model.safetensors.index.json
llama2_origin = r'/root/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots' \
                r'/8a0442e81540efaeb1a0fe3e95477b5e0edfd423'
llama2_finetune = r'/root/llm/finetune/dev/saves/LLaMA2-7B/llama2-7b_alpaca_zh_lora'

# output length predict
llama2_added = r'/root/llm/finetune/dev/saves/LLaMA2-7B/llama2-7b_added'
llama2_predict = r'/root/llm/finetune/dev/saves/LLaMA2-7B/llama2-7b_predict'


def convert(
        output: str,
        mod_tokenizer: transformers.PreTrainedTokenizer,
        special_token: str = '<e>',
        punctuation_pattern: str = r'[。？！，]'
):
    matches = re.finditer(punctuation_pattern, output)
    positions = [match.start() for match in matches]
    for p in reversed(positions[:-1]):
        sub_len = len(mod_tokenizer(output[p + 1:])['input_ids'])
        output = output[:p + 1] + special_token + str(sub_len) + special_token + output[p + 1:]
    output = special_token + str(len(mod_tokenizer(output)['input_ids'])) + special_token + output
    return output


tokenizer = LlamaTokenizer.from_pretrained(llama2_origin, trust_remote_code=True)
tokenizer.add_tokens(['<e>'])
model = LlamaForCausalLM.from_pretrained(llama2_origin, torch_dtype=torch.bfloat16)
model.resize_token_embeddings(len(tokenizer))

data = json.loads(open('data/alpaca_data_zh_51k.json').read())
for i in tqdm.tqdm(range(len(data))):
    try:
        data[i]['output'] = convert(data[i]['output'], tokenizer)
    except KeyError as e:
        print(e, i, data[i].keys(), data[i])
        continue

data = json.dumps(data, ensure_ascii=False, indent=4)
with open('data/alpaca_predict_zh_51k.json', 'w', encoding='utf-8') as f:
    f.write(data)

# tokenizer.save_pretrained(llama2_added)
# model.save_pretrained(llama2_added)




