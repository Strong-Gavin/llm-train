
"""
1、处理数据集
"""
from datasets import load_dataset
import json

from transformers import TrainingArguments


# 加载huggingface数据集  使用 streaming=True 避免一次性下载全部数据
# dataset = load_dateset("Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT",streaming=True)


dataset = load_dataset("json",data_files="/data/code/dataset/distill_r1_110k_sft.json")

# 过滤数据 使用lambda匿名函数 过滤小红书数据
filtered_dateset = dataset.filter(lambda example: example["repo_name"]== 'xhs/xhs')

# 定义保存数据集函数
def save_to_json(dataset, filename):
    with open(filename, 'w',encoding='utf-8') as f:
        for example in dataset:
            # dataset是一个可迭代对象，需要逐条example提取
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')  # 每条数据之间添加换行符


save_to_json(filtered_dateset['train'],'xhs_data.json')

print("数据已保存到 xhs_data.json")

"""
1、处理数据集结束
"""


"""
2、配置wandb记录跟踪训练过程
"""

import wandb
wb_token="925e23f1738a026ef3f4dea9ddd6791c231fa4de"  #官网免费申请
wandb.login(key=wb_token)
run = wandb.init(
    project='HOME fine_tune_DeepSeek-R1-Distill-Qwen-7B on xhs sheet',
    job_type="training",
    anonymous="allow"
)

"""
2、配置wandb记录跟踪训练过程
"""

"""
3、加载模型和分词器
"""
from unsloth import FastLanguageModel, is_bfloat16_supported

#  设置最大上下数，建议测试时2048
max_seq_length = 2048

# 设置数据类型
dtype = None

# 设置4位量化加载模型
load_in_4bit = True

# 加载模型和对应的tokenizer
model,tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/data/code/llm/models /DeepSeek-R1-Distill-Qwen-7B",
    max_seq_length = max_seq_length,  #控制上下文长度，建议测试时2048
    dtype = dtype, #在新一代的GPU建议使用torch.float16或者torch.bfloat16
    load_in_4bit = load_in_4bit  #开启4位量化，降低4倍显存微调可以运行在16G显存，在GPU可以设置none，比如H100提升准确率
)

"""
3、加载模型和分词器结束
"""

"""
4、定义训练时prompt模版
"""
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
EOS_TOKEN = tokenizer.eos_token

"""
4、定义训练时prompt模版结束
"""

"""
5、训练数据处理与标准化
"""
import re

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

"""
5、训练数据处理与标准化结束
"""

"""
6、加载数据集，并使用函数处理
"""
from datasets import load_dataset
dataset = load_dataset("json",data_files="xhs_data.json", split="train")
#使用map函数处理
dataset = dataset.map(formatting_prompts_func,batched=True)

"""
6、加载数据集，并使用函数处理结束
"""

"""
7、设置模型Lora超参
"""
model = FastLanguageModel.get_peft_model(
    model,
    r = 32, #设置Lora低秩矩阵的秩（维度），建议是8，16，32，64，128。值越大，模型越准备，但是需要的显存越大，过拟合的风险越大。值越低，越省显存，但是容易减低准确率
    target_modules=[ #微调的模块，建议微调全模块。
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_alpha=32, #缩放因子，参数值的更新幅度，建议是1或者2倍r的值， 值越大可能过拟合，值越小可能更泛化，专业能力不够
    lora_dropout=0, #默认0，值越高，更规则化，训练越慢。值越低，更快地训练，对过拟合的影响最小
    bias="none", # 默认none，控制偏差项更新。none可以或者优化的更快的训练
    use_gradient_checkpointing="unsloth", # 降低显存使用对于长上下文，使用unsloth可以通过unsloth的梯度检查点算法降低显存
    random_state = 66666, #复现实验的种子，建议设置一个固定值
    use_rslora=False, # 开始固定rank的LoRA,
    loftq_config=None, # 应用量化和高级Lora初始化，默认None，设置为SET。使用奇异向量初始化lora可以提高准确率但是增加显存占用
)

"""
7、设置模型Lora超参结束
"""

"""
8、配置训练器超参
"""

args = TrainingArguments(
    per_device_train_batch_size=2, #每个设备批次大小
    gradient_accumulation_steps=4,  # 梯度累积步数，批次大小建议调整这个（变相模拟更大batch_size）  总批次大小=per_device_train_batch_size*gradient_accumulation_steps = 8
    warmup_steps=5,  # 学习率预热步数。
    max_steps=200,  # 总训练步数,可以设置总训练部署，也可以设置总epoch
    num_train_epochs=3,
    learning_rate=2e-4,  # 学习率 范围建议1e-4(0.0001) 到5e-5(0.00005)
    fp16=not is_bfloat16_supported(),  #精度类型
    bf16=is_bfloat16_supported(), #精度类型
    logging_steps=10,   #每10步打印一次log
    optim="adamw_8bit", #优化器类型
    weight_decay=0.01,  # 权重衰减系数
    lr_scheduler_type="linear",  # 学习率调度器
    seed=6666,  # 选一个自己喜欢的随机数种子
    output_dir="outputs",
    report_to="wandb",  # 使用 wandb 进行报告,实验指标可视化
)

"""
8、配置训练器超参结束
"""

"""
9、设置训练器
"""
from trl import SFTTrainer

trainer = SFTTrainer(
    model = model, #设置模型
    tokenizer = tokenizer, #设置解析器
    train_dataset = dataset, #设置训练数据集
    dataset_text_field = "text", #设置标签
    max_seq_length = max_seq_length, #设置最大上下文
    dataset_num_proc = 2, #预处理数据集的进程数,
    args=args #训练超参
)

"""
9、设置训练器结束
"""

"""
10、启动训练
"""
trainer_stats = trainer.train()

"""-
10、训练结束
"""

"""
***  过拟合(太特殊了，通用能力缺失)
模型记住了训练数据，在未见过的数据上没法生成或者生成不好。
解决方案有以下几种：
1、降低学习率
2、降低训练次数epoch
3、构建自己的数据集和通用数据集
4、提高dropout防止过拟合


*** 欠拟合（太通用，专业能力不足）
模型回答和没训练类似
解决方案有以下几种
1、提高学习率
2、训练更多的次数epoch
3、使用领域相关的数据集



微调没有单一的“最佳”方法，只有最佳实践。实验是找到适合您需求的方法的关键。
"""

"""
模型调试
"""


# prompt_style = """以下是一项任务说明，并附带了更详细的背景信息。
# 请撰写一个满足完成请求的回复。
# 在回答之前，请仔细考虑问题，并创建一个逐步的思考链，以确保逻辑和准确的回答。
#
# ### Instruction:
# 你是一个资深的小红书文案专家
# 请你根据以下问题完成写作
# ### Question:
# {}
# ### Response:
# <think>{}
# """
# question = "写一篇小红书风格的帖子，标题是男生变帅只需三步丨分享逆袭大干货  "
#
#
# FastLanguageModel.for_inference(model)
# inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")
# outputs = model.generate(
#     input_ids=inputs.input_ids,
#     attention_mask=inputs.attention_mask,
#     max_new_tokens=1200,
#     use_cache=True,
# )
# response = tokenizer.batch_decode(outputs)
# print(response[0].split("### Response:")[1])
"""
模型调试结束
"""
"""
模型导出
"""
# new_model_local = "/data/llm/models/unsloth/finetune"
# #保存预训练lora适配器
# model.save_pretrained(new_model_local)
# tokenizer.save_pretrained(new_model_local)
#
#
# #合并原模型和lora为统一模型vllm运行
# model.save_pretrained_merged(new_model_local, tokenizer, save_method="merged_16bit")
#
# #合并原模型和lora并量化为qk_k_m
# model.save_pretrained_gguf(new_model_local, tokenizer, save_method="q4_k_m")


"""
模型导出
"""


