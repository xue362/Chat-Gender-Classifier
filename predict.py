# predict.py

import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
from tqdm import tqdm
import os
import numpy as np

# --- 1. 配置 ---
class Config:
    # 训练好的模型所在的路径
    MODEL_PATH = "results/gender_classifier"
    # 文本最大长度，应与训练时保持一致
    MAX_TEXT_LENGTH = 128

# --- 2. 核心预测器类 ---
class GenderPredictor:
    """
    封装了模型加载、文本清洗和预测的完整逻辑。
    """
    def __init__(self, model_path):
        print("Initializing predictor...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found at {model_path}. Please make sure you have trained the model.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

        # 从模型配置中读取标签映射 (e.g., {0: '男', 1: '女'})
        self.label_map = {v: k for k, v in self.model.config.label2id.items()}
        print(f"Model loaded. Label mapping: {self.label_map}")

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'^.*\[引用\].*$\n?', '', text, flags=re.MULTILINE)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def predict(self, text: str):
        """
        对单条文本进行预测。
        
        <-- 修改部分：现在返回原始概率，用于后续计算 -->
        Returns:
            tuple: (预测标签, 置信度, [P(男), P(女)])
        """
        cleaned_text = self._clean_text(text)
        if not cleaned_text.strip():
            # 返回一个中性结果，概率均等
            return "无效输入", 0.5, np.array([0.5, 0.5])

        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=Config.MAX_TEXT_LENGTH
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        # 计算Softmax概率
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        predicted_class_id = torch.argmax(logits, dim=-1).item()

        predicted_label = self.label_map[predicted_class_id]
        confidence = probabilities[predicted_class_id].item()

        # 确保返回的概率顺序是 [P(男), P(女)]
        # 假设 label_map 中 0 是 '男', 1 是 '女'
        prob_male = probabilities[0].item()
        prob_female = probabilities[1].item()

        return predicted_label, confidence, np.array([prob_male, prob_female])

# --- 3. 脚本运行逻辑 ---
def run_interactive_mode(predictor: GenderPredictor):
    """启动交互式预测循环。"""
    print("\n--- 交互式预测模式 ---")
    print("输入一段中文文本后按Enter键即可预测。输入 'quit' 或 'exit' 退出。")
    while True:
        text_to_predict = input("请输入文本: ")
        if text_to_predict.lower() in ["quit", "exit"]:
            print("退出程序。")
            break
        
        predicted_gender, confidence, _ = predictor.predict(text_to_predict)
        print(f"  -> 预测结果: {predicted_gender} (置信度: {confidence:.2%})")

def run_file_mode(predictor: GenderPredictor, file_path: str, text_column: str, output_path: str):
    """读取CSV文件，进行批量预测，并使用近似朴素贝叶斯方法输出最终结论。"""
    print(f"\n--- 文件预测模式 ---")
    try:
        df = pd.read_csv(file_path)
        if text_column not in df.columns:
            print(f"错误: 文件 '{file_path}' 中找不到指定的文本列 '{text_column}'。")
            return
            
        print(f"读取文件 '{file_path}'，准备处理 '{text_column}' 列...")
        
        # <-- 修改部分：收集每条消息的预测结果和概率 -->
        per_message_results = []
        all_probabilities = []
        for text in tqdm(df[text_column], desc="正在逐条预测"):
            pred_label, conf, probs = predictor.predict(text)
            per_message_results.append((pred_label, conf))
            all_probabilities.append(probs)
            
        # 将逐条预测的结果添加到DataFrame，用于保存和分析
        predictions_df = pd.DataFrame(per_message_results, columns=['predicted_gender', 'confidence'])
        output_df = pd.concat([df.reset_index(drop=True), predictions_df], axis=1)
        
        # <-- Bug修复：根据正确的标签名进行统计 -->
        # 假设标签是 '男' 和 '女'
        count_male = len(output_df[output_df['predicted_gender'] == 'LABEL_0'])
        count_female = len(output_df[output_df['predicted_gender'] == 'LABEL_1'])
        print(f"\n逐条预测结果统计: 男={count_male}, 女={count_female}")
        
        # 保存包含逐条预测结果的详细文件
        output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"详细的逐条预测结果已保存到 '{output_path}'")

        # <-- 新增部分：近似朴素贝叶斯最终判断 -->
        if not all_probabilities:
            print("没有有效的消息可用于最终判断。")
            return

        # 将概率列表转为numpy数组，并加上一个极小值防止log(0)
        probs_array = np.array(all_probabilities)
        probs_array += 1e-9

        # 计算对数概率并求和
        log_probs = np.log(probs_array)
        total_log_probs = np.sum(log_probs, axis=0) # [sum(log P(男)), sum(log P(女))]
        print(f"对数概率求和结果: {total_log_probs}")

        # 最终判断
        final_prediction_id = np.argmax(total_log_probs)
        final_predicted_label = "女" if final_prediction_id else "男"

        # 计算最终的置信度 (通过Softmax将对数概率转换回来)
        # 为防止数值溢出，先减去最大值
        total_log_probs -= np.max(total_log_probs)
        final_probs = np.exp(total_log_probs) / np.sum(np.exp(total_log_probs))
        final_confidence = np.max(final_probs)

        print("\n--- 最终综合判断 (近似朴素贝叶斯) ---")
        print(f"对文件 '{file_path}' 的整体分析结论为:")
        print(f"✅ 最终预测性别: {final_predicted_label} (综合置信度: {final_confidence:.2%})")
        print("---------------------------------------------")

    except FileNotFoundError:
        print(f"错误: 输入文件 '{file_path}' 未找到。")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用训练好的模型预测文本来源的性别。")
    parser.add_argument("--file", type=str, help="输入CSV文件的路径。如果提供此参数，则进入文件模式。")
    parser.add_argument("--column", type=str, default="message", help="CSV文件中包含文本的列名。")
    parser.add_argument("--output", type=str, default="predictions_details.csv", help="保存逐条预测结果的输出CSV文件名。")

    # python interactive.py --file to_predict.csv --column message --output prediction_details.csv
    
    args = parser.parse_args()

    try:
        predictor = GenderPredictor(Config.MODEL_PATH)
        
        if args.file:
            run_file_mode(predictor, args.file, args.column, args.output)
        else:
            run_interactive_mode(predictor)
    except Exception as e:
        print(f"初始化预测器时发生严重错误: {e}")