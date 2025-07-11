# test.py

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset

# --- 1. 配置 ---
class Config:
    # 训练好的模型所在的路径
    MODEL_PATH = "results/gender_classifier"
    # 需要进行测试的文件路径 (你可以把它改成 'test.csv')
    TEST_FILE = "test.csv"
    # 预测时的批处理大小 (可以根据你的显存调整)
    BATCH_SIZE = 32
    # 文本最大长度，应与训练时保持一致
    MAX_TEXT_LENGTH = 128

# --- 2. 数据集类 (与训练时类似，但不包含标签处理) ---
class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

# --- 3. 主评估流程 ---
def evaluate_model():
    print("--- 开始模型评估流程 ---")

    # 检查路径是否存在
    if not os.path.exists(Config.MODEL_PATH):
        print(f"错误：模型路径 '{Config.MODEL_PATH}' 不存在。请先运行训练脚本。")
        return
    if not os.path.exists(Config.TEST_FILE):
        print(f"错误：测试文件 '{Config.TEST_FILE}' 不存在。")
        return

    # 选择设备 (GPU或CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"1. 使用设备: {device}")

    # 加载已保存的模型和分词器
    print(f"2. 从 '{Config.MODEL_PATH}' 加载模型和分词器...")
    model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)

    # 将模型移动到指定设备
    model.to(device)
    # 设置为评估模式
    model.eval()

    # 加载并准备数据
    print(f"3. 加载测试数据: {Config.TEST_FILE}...")
    df = pd.read_csv(Config.TEST_FILE)
    texts_to_predict = df['clean_msg'].astype(str).tolist()
    
    # 创建数据集和数据加载器
    inference_dataset = InferenceDataset(texts_to_predict, tokenizer, Config.MAX_TEXT_LENGTH)
    inference_loader = DataLoader(inference_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # 进行预测
    print("4. 开始进行预测...")
    all_predictions = []
    with torch.no_grad():  # 在此模式下，不计算梯度，节省资源
        for batch in inference_loader:
            # 将批次数据移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 模型前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 获取预测结果
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())

    # 准备真实标签
    # "男" -> 0, "女" -> 1
    true_labels = df['性别'].apply(lambda x: 1 if x == '女' else 0).tolist()
    label_names = ['男 (class 0)', '女 (class 1)']

    # 生成并打印评估报告
    print("\n--- 评估报告 ---")
    print(f"测试文件: {Config.TEST_FILE}\n")
    # classification_report 会计算所有关键指标
    report = classification_report(true_labels, all_predictions, target_names=label_names, digits=4)
    print(report)

    # 提取并单独打印准确率
    accuracy = np.mean(np.array(all_predictions) == np.array(true_labels))
    print("-----------------------------------------")
    print(f"✅ 总体准确率 (Overall Accuracy): {accuracy:.4f}")
    print("-----------------------------------------")


if __name__ == "__main__":
    evaluate_model()