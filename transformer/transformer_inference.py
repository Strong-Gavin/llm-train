from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
import torch
import pandas as pd
from datasets import Dataset

model_path = "/data/code/llm/models/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_path)

quantization_config = BitsAndBytesConfig()
model = AutoModelForCausalLM.from_pretrained(model_path,quantization_config=quantization_config)

#推理测试

prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
<think>{}
"""

question = "一个患有急性阑尾炎的病人已经发病5天，腹痛稍有减轻但仍然发热，在体检时发现右下腹有压痛的包块，此时应如何处理？"
model.generation_config = GenerationConfig.from_pretrained(model_path)

message = prompt_style.format(question,"")
model_inputs = tokenizer([message],return_tensors="pt").to("cuda")
# attention_mask = model_inputs.input_ids.ne(tokenizer.pad_token_id).long().to("cuda")

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=2048,
    pad_token_id=tokenizer.eos_token_id,
    use_cache=True
)

response = tokenizer.batch_decode(generated_ids,skip_special_tokens=True)
print(response[0].split("### Response:")[1])

