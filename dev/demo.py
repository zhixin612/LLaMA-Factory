import os
import re
import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM

# tokenizer.json, tokenizer.model, tokenizer_config.json,
# special_tokens_map.json, config.json, generation_config.json,
# model-00001-of-00002.safetensors, model-00002-of-00002.safetensors, model.safetensors.index.json
llama2_root = r'/root/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots' \
              r'/8a0442e81540efaeb1a0fe3e95477b5e0edfd423'
llama2_root = r'../saves/LLaMA2-7B/llama2-7b_alpaca_zh_lora'


def test_tokenizer():
    # load & encode & decode
    tokenizer = LlamaTokenizer.from_pretrained(llama2_root)
    print(tokenizer.encode('hello world'), tokenizer.decode([1, 22172, 3186]))
    print(tokenizer.max_model_input_sizes, tokenizer.vocab_size, tokenizer.model_max_length)

    # padding & truncation
    # note: tokenizer() == tokenizer.encode()
    tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer(['hello', 'how are you today'], return_tensors='pt', padding=True, truncation=True))

    # add special tokens
    tokenizer.add_tokens('')


def test_model():
    tokenizer = LlamaTokenizer.from_pretrained(llama2_root)
    inputs = tokenizer('hello!', return_tensors='pt')  # {'input_ids'; 'attention_mask'}

    # load & forward
    model = LlamaForCausalLM.from_pretrained(llama2_root)
    # get_input_embeddings: Embedding(32000, 4096) -> input[bsize, seq_len, 4096]
    # get_output_embedding: Linear(in_features=4096, out_features=32000, bias=False) -> output[bsize, seq_len, 32000]
    outputs = model(**inputs)
    print(type(outputs), dict(outputs).keys())
    print(outputs.logits.shape, outputs.logits[0, -1, :].argmax())
    print(outputs.logits.argmax(-1))
    print(tokenizer.decode(outputs.logits.argmax(-1)[0], skip_special_tokens=False))
    # print('Inference: ', model(**inputs))

    # embedding layer
    # embedding_layer = model.get_input_embeddings()
    # embeddings1 = embedding_layer(inputs)
    # embeddings2 = embedding_layer.forward(inputs)
    # print('Embedding Shape = ', embeddings1.shape)
    # print(embeddings1)
    # print(embeddings2)


def test_inference():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch.cuda.set_device('cuda:0')

    tokenizer: LlamaTokenizer = AutoTokenizer.from_pretrained(llama2_root, trust_remote_code=True)
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        llama2_root, trust_remote_code=True, torch_dtype=torch.bfloat16)

    model.requires_grad_(False)
    model.eval()
    model.to('cuda:0')

    model.resize_token_embeddings(len(tokenizer))

    input_text = "你好！"
    inputs = tokenizer(input_text, return_tensors='pt').to('cuda:0')

    outputs = model.generate(**inputs, max_length=256, do_sample=True, temperature=0.95, top_p=0.7, top_k=50)
    # output: tensor(bsize, seq_len): batched output_ids
    print(tokenizer.decode(outputs[0]))


def test_add_tokens():
    tokenizer = LlamaTokenizer.from_pretrained(llama2_root)

    print(tokenizer('hello<e>1<e>'))  # [1, 22172, 29966, 29872, 29958, 29896, 29966, 29872, 29958]
    tokenizer.add_tokens(['<e>'])
    print(tokenizer('hello<e>1<e>'))  # [1, 22172, 32000, 29896, 32000]
    # tokenizer.save_pretrained('llama2_tokenizer_mod')


def test_mod_data():
    tokenizer = LlamaTokenizer.from_pretrained(llama2_root)
    tokenizer.add_tokens(['<e>'])

    import json
    data = json.loads(open('alpaca_data_zh_51k.json').read())
    print(len(data), data[0].keys())
    print('instruction: ', data[0]['instruction'])
    print('input: ', data[0]['input'])
    print('output: ', data[0]['output'])


def test_str():
    def convert(
            output: str,
            mod_tokenizer: transformers.PreTrainedTokenizer,
            special_token: str = '<e>',
            punctuation_pattern: str = r'[。？！]'
    ):
        matches = re.finditer(punctuation_pattern, output)
        positions = [match.start() for match in matches]
        for i in reversed(positions[:-1]):
            sub_len = len(mod_tokenizer(output[i+1:])['input_ids'])
            output = output[:i+1] + special_token + str(sub_len) + special_token + output[i+1:]
        output = special_token + str(len(mod_tokenizer(output)['input_ids'])) + special_token + output
        return output

    tokenizer = LlamaTokenizer.from_pretrained(llama2_root)
    tokenizer.add_tokens(['<e>'])

    print(len(tokenizer))

    # model.set_embedding
    prompt = '这是一句示例。这是示例？示例示例！输出是什么。输出输出输出。这是一句示例。这是示例？示例示例！输出是什么。输出输出输出。'
    print(convert(prompt, tokenizer))


# test_model()
# test_inference()
# test_add_tokens()
# test_mod_data()
test_str()