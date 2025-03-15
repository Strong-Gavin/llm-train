import gc
import re

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq

from peft import LoraConfig

## 数据集制作
# train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
# Write a response that appropriately completes the request.
# Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.
#
# ### Instruction:
# You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
# Please answer the following medical question.
#
# ### Question:
# {}
#
# ### Response:
# <think>
# {}
# </think>
# {}"""


train_prompt_style = """以下是一项任务说明，并附带更详细的背景信息。
请撰写一个满足完成请求的回复。
在回答之前，请仔细考虑问题，并创建一个逐步的思考链，以确保逻辑和准确的回答。

### Instruction:
你是一个资深的小红书文案专家
请你根据以下问题完成写作
### Question:
{}
### Response:
<think>
{}
</think>
{}"""



model_path = "/data/code/llm/models/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True
)



EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

# 迭代训练集数据，处理prompt
# def formatting_prompts_func(examples):
#     inputs = examples["Question"]
#     cots = examples["Complex_CoT"]
#     outputs = examples["Response"]
#     texts = []
#     for input, cot, output in zip(inputs, cots, outputs):
#         text = train_prompt_style.format(input, cot, output) + EOS_TOKEN
#         texts.append(text)
#     return {
#         "text": texts,
#     }

# 使用正则提取思考链和最终响应
def formatting_prompts_func(examples):
    instructions  = examples["instruction"]
    outputs = examples["output"]
    texts = []

    for instruct_text, output_text in zip(instructions,outputs):
        #使用正则表达式提取<think>和</think>之间的内容作为cots
        match = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL)
        cot = match.group(1).strip() if match else ""

        #提取</think>之后的内容作为outputs
        outputs = output_text.replace(match.group(0),"").strip() if match else output_text.strip()

        text = train_prompt_style.format(instruct_text, cot, outputs) + EOS_TOKEN

        texts.append(text)

    return {"text": texts}

from datasets import load_dataset
dataset = load_dataset("json",data_files="xhs_data.json", split="train")
#使用map函数处理
dataset = dataset.map(formatting_prompts_func,batched=True)


def process_func(examples, tokenizer, max_seq_length):
    """
    将数据集进行预处理
    """
    input_ids, attention_mask, labels = [], [], []
    inputs = examples["text"]
    input_text =tokenizer.bos_token + inputs + tokenizer.eos_token

    input_tokenizer = tokenizer(
        input_text,
        add_special_tokens=False,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    input_ids += input_tokenizer['input_ids']
    attention_mask += input_tokenizer['attention_mask']

    if len(input_ids) > max_seq_length:  # 做一个截断
        input_ids = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids
    }



train_dataset = dataset.map(process_func,fn_kwargs={"tokenizer": tokenizer, "max_seq_length": tokenizer.model_max_length},remove_columns=dataset.column_names)


from swanlab.integration.transformers import SwanLabCallback

swanlab_config = {
        "peft":"lora"
    }
swanlab_callback = SwanLabCallback(
    project="deepseek-finetune-xhs",
    experiment_name="first-test",
    description="小红书",
    workspace=None,
    config=swanlab_config,
)


lora_config = LoraConfig(
        r=8,  # 设置Lora低秩矩阵的秩（维度），建议是8，16，32，64，128。值越大，模型越准备，但是需要的显存越大，过拟合的风险越大。值越低，越省显存，但是容易减低准确率
        target_modules=[  # 微调的模块，建议微调全模块。
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_alpha=16,  # 缩放因子，参数值的更新幅度，建议是1或者2倍r的值， 值越大可能过拟合，值越小可能更泛化，专业能力不够
        lora_dropout=0,  # 默认0，值越高，更规则化，训练越慢。值越低，更快地训练，对过拟合的影响最小
        bias="none",  # 默认none，控制偏差项更新。none可以或者优化的更快的训练
        use_rslora=False,  # 开始固定rank的LoRA,
        inference_mode=False  # 训练模式
    )


from transformers import TrainingArguments


# 输出地址
output_dir="outputs"
# 配置训练参数
train_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1, #每个设备批次大小
    gradient_accumulation_steps=1,  # 梯度累积步数，批次大小建议调整这个（变相模拟更大batch_size）  总批次大小=per_device_train_batch_size*gradient_accumulation_steps = 8
    warmup_steps=5,  # 学习率预热步数。
    max_steps=200,  # 总训练步数,可以设置总训练部署，也可以设置总epoch
    num_train_epochs=3,
    learning_rate=2e-4,  # 学习率 范围建议1e-4(0.0001) 到5e-5(0.00005)
    logging_steps=10,   #每10步打印一次log
    optim="adamw_8bit", #优化器类型
    weight_decay=0.01,  # 权重衰减系数
    lr_scheduler_type="linear",  # 学习率调度器
    seed=6666,  # 选一个自己喜欢的随机数种子
    remove_unused_columns=False,
    label_names=["labels"]
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config

)




from peft import get_peft_model, LoraConfig, TaskType

from transformers import Trainer

# 用于确保模型的词嵌入层参与训练
model.enable_input_require_grads()
# 应用 PEFT 配置到模型
model = get_peft_model(model,lora_config)
model.print_trainable_parameters()


# 配置训练器
trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        # 添加数据整理函数
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[swanlab_callback]
        )
# 启动训练
trainer.train()

gc.collect()
torch.cuda.empty_cache()
