from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer


import pandas as pd
import numpy as np

# 1. 载入模型和分词器
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. 配置 LoRA
lora_config = LoraConfig(
    r=8,            # rank
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # 只在attention投影层加LoRA
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)

# 3. 数据集
dataset = load_dataset("json", data_files="my_sft_data.json")

# 4. 训练参数
args = TrainingArguments(
    output_dir="./lora-sft-model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    fp16=True,
    optim="adamw_torch",
)

# 5. Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    dataset_text_field="output",  
    tokenizer=tokenizer,
    args=args,
)

trainer.train()




