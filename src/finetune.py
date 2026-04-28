import json
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# ===================== 1. 配置 =====================
model_path = "../model/Qwen-1_8B-Chat"
device = "cpu"

# LoRA 配置（无冲突）
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# ===================== 2. 加载模型 =====================
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # 结束占位符填充缺失的边，eos-><eos>(end of sequence)
tokenizer.padding_side = "right" # 结束占位符放在右边

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    trust_remote_code=True
).to(device)

# 绑定 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===================== 3. 运维数据集（你的业务数据） =====================
data = [
    {
        "text": "<|im_start|>user\nCPU使用率100%怎么排查？<|im_end|>\n<|im_start|>assistant\n【故障现象】CPU占用100%\n【排查】top -c\n【修复】kill PID<|im_end|>"
    },
    {
        "text": "<|im_start|>user\n如何查看内存？<|im_end|>\n<|im_start|>assistant\n【命令】free -h\n【说明】used已使用，available可用<|im_end|>"
    }
]
dataset = Dataset.from_list(data)

# ===================== 4. 分词（极简无冲突） =====================
def tokenize(sample):
    out = tokenizer(
        sample["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )
    out["labels"] = out["input_ids"].copy()
    return out

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# ===================== 5. 训练参数（CPU 专用） =====================
args = TrainingArguments(
    output_dir="./lora-ops-model",
    per_device_train_batch_size=1,
    num_train_epochs=2,
    learning_rate=1e-4,
    fp16=False,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
    use_cpu=True
)

# ===================== 6. 原生 Trainer（零冲突！） =====================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# ===================== 7. 训练 + 保存 =====================
trainer.train()
model.save_pretrained("./lora-ops-model")
tokenizer.save_pretrained("./lora-ops-model")

print("✅ 训练完成！无任何依赖冲突！")