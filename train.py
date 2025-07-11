# train.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os

# --- 1. 参数配置 ---
class ModelConfig:
    # 模型和文件路径配置
    MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
    TRAIN_FILE = "train.csv"
    VALID_FILE = "validation.csv"
    
    # 输出目录
    OUTPUT_DIR = "results/gender_classifier"
    LOGGING_DIR = "logs/gender_classifier_logs"

class TrainingConfig:
    # 训练过程超参数配置
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-5
    TRAIN_BATCH_SIZE = 16  # 根据你的显存大小调整
    EVAL_BATCH_SIZE = 32
    MAX_TEXT_LENGTH = 128  # 文本最大长度

# --- 2. 自定义数据集类 ---
class GenderDataset(Dataset):
    """自定义PyTorch数据集，用于加载和预处理性别分类数据"""
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        # 确保 'clean_msg' 列是字符串类型
        self.texts = self.data['clean_msg'].astype(str).tolist()
        # 将 "男" -> 0, "女" -> 1
        self.labels = self.data['性别'].apply(lambda x: 1 if x == '女' else 0).tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        # 使用分词器处理文本
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_tensors='pt',  # 返回PyTorch张量
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- 3. 性能评估函数 ---
def compute_metrics(p):
    """在评估过程中计算并返回性能指标"""
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        p.label_ids, preds, average='binary' # 使用二分类模式
    )
    acc = accuracy_score(p.label_ids, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

# --- 4. 主训练流程 ---
def main():
    print("--- 开始模型训练流程 ---")

    # 检查输入文件是否存在
    for f in [ModelConfig.TRAIN_FILE, ModelConfig.VALID_FILE]:
        if not os.path.exists(f):
            print(f"错误：数据文件 '{f}' 不存在。")
            print("请先运行（简化版）的preprocess.py来生成所需的CSV文件。")
            return
            
    # 加载分词器
    print(f"1. 加载分词器: {ModelConfig.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.MODEL_NAME)

    # 加载和准备数据集
    print("2. 加载和准备数据集...")
    train_df = pd.read_csv(ModelConfig.TRAIN_FILE)
    valid_df = pd.read_csv(ModelConfig.VALID_FILE)
    
    train_dataset = GenderDataset(train_df, tokenizer, TrainingConfig.MAX_TEXT_LENGTH)
    valid_dataset = GenderDataset(valid_df, tokenizer, TrainingConfig.MAX_TEXT_LENGTH)
    
    # 加载预训练模型
    print(f"3. 加载预训练模型: {ModelConfig.MODEL_NAME}")
    # num_labels=2 表示我们的任务是二分类（男/女）
    model = AutoModelForSequenceClassification.from_pretrained(
        ModelConfig.MODEL_NAME, num_labels=2
    )
    
    # 定义训练参数
    print("4. 配置训练参数...")
    training_args = TrainingArguments(
        output_dir=ModelConfig.OUTPUT_DIR,
        logging_dir=ModelConfig.LOGGING_DIR,
        num_train_epochs=TrainingConfig.NUM_EPOCHS,
        learning_rate=TrainingConfig.LEARNING_RATE,
        per_device_train_batch_size=TrainingConfig.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=TrainingConfig.EVAL_BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=10,
        # 移除所有评估相关参数，使用默认设置
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 开始训练
    print("5. 开始训练模型...")
    trainer.train()
    
    # 保存最终的模型和分词器
    print("6. 训练完成，保存最优模型...")
    trainer.save_model(ModelConfig.OUTPUT_DIR)
    tokenizer.save_pretrained(ModelConfig.OUTPUT_DIR)
    
    print("\n--- 模型训练与保存全部完成！---")
    print(f"最优模型已保存至: {ModelConfig.OUTPUT_DIR}")


if __name__ == "__main__":
    main()