import argparse
import sys
import os
import torch
import json
import bitsandbytes as bnb
import time 

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_bnb_config, get_prompt

## params setting
llama_path = sys.argv[1]
print('*'*5, "Taiwan llama path:", llama_path, '*'*5)
lora_path = sys.argv[2]
print('*'*5, "pefy config path:", lora_path, '*'*5)
test_json_path = sys.argv[3]
print('*'*5, "input json path:", test_json_path, '*'*5)
output_path = sys.argv[4]
print('*'*5, "output json path:", output_path, '*'*5)

# llama_path = '/home/avlab/桌面/ADL-HW3/Taiwan-LLM-7B-v2.0-chat'
# lora_path = '/home/avlab/桌面/ADL-HW3/tmp/test_ckpt'
# test_json_path = '/home/avlab/桌面/ADL-HW3/data/private_test.json'
# output_path = 'prediction.json'
device = "cuda:0" if torch.cuda.is_available() else 'cpu'

## function 
def load_model(model_name, lora_path, bnb_config):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name,  use_auth_token=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

## main
# load pre-trained model and LoRA
bnb_config = get_bnb_config()
model, tokenizer = load_model(llama_path, lora_path, bnb_config)

# load test data
with open(test_json_path, "r") as f:
    test_data = json.load(f)
instructions = [get_prompt(x["instruction"]) for x in test_data]
id = [x["id"] for x in test_data]

# generate 
count = 0
output_list = []

start_time = time.time()
for prompt, id in zip(instructions, id):
    input = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**input, max_new_tokens=512)
    ans = tokenizer.decode(output[0], skip_special_tokens=True)

    output = ans.split('ASSISTANT:')[-1].strip()
    # print(f'id:{id}')
    # print(f'output:{output}\n')
    output_list.append({'id': str(id),'output': output})

    # clean gpu usage
    torch.cuda.empty_cache()
    inputs = None
    outputs = None

    count += 1
    print(count)
print('-'*40, '\n', f'Time cost: {time.time() - start_time:.0f} sec')

# save as json
json_file = open(output_path, mode='w', encoding="utf8")
json.dump(output_list, json_file, ensure_ascii=False, indent=2)
print('-'*40, '\n', f'save at {output_path}')  

