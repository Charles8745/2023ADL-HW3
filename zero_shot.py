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
llama_path = '/home/avlab/桌面/ADL-HW3/Taiwan-LLM-7B-v2.0-chat'
test_json_path = '/home/avlab/桌面/ADL-HW3/data/public_test.json'
output_path = '/home/avlab/桌面/zero_shot.json'
device = "cuda:0" if torch.cuda.is_available() else 'cpu'
end = 2

## function 
def load_model(model_name, bnb_config):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name,  use_auth_token=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def create_prompt_formats(instruction):
    prompt = f"文言文主要指以秦漢書面語為標準，脫離口語而寫成的文字。 語言學上，先秦及西漢時使用的語言被稱為上古漢語。 此時傳世的一些文獻，如《詩經》《論語》《左傳》《韓非子》《史記》 ，被視為文言文的範本。USER: {instruction} ASSISTANT: "
    return prompt

## main
# load pre-trained model and LoRA
bnb_config = get_bnb_config()
model, tokenizer = load_model(llama_path, bnb_config)

# load test data
with open(test_json_path, "r") as f:
    test_data = json.load(f)
Q = [x["instruction"] for x in test_data[:end]]
instructions = [create_prompt_formats(x["instruction"]) for x in test_data[:end]]
output = [x["output"] for x in test_data[:end]]
id = [x["id"] for x in test_data[:end]]

# generate 
count = 0
output_list = []

start_time = time.time()
for prompt, gt, id, q in zip(instructions, output, id, Q):
    input = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**input, max_new_tokens=512)
    ans = tokenizer.decode(output[0], skip_special_tokens=True)

    output = ans.split('ASSISTANT:')[-1].strip()
    output_list.append({'id': str(id), 'commad': q, 'output': output, 'gt': gt})

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