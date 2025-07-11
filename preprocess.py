# preprocess.py

import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os

def clean_text_for_bert(text):
    """
    为BERT模型进行文本清洗：只移除噪音，不进行分词或去除停用词。
    """
    if not isinstance(text, str):
        return ""
    
    # 1. 移除引用块 (关键步骤)
    text = re.sub(r'^.*\[引用\].*$\n?', '', text, flags=re.MULTILINE)
    
    # 2. 移除URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. 移除"[图片]"、"[表情]"等系统提示
    text = re.sub(r'\[.*?\]', '', text)
    
    # 4. 移除多余的换行和空格，但保留句子结构
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_data(input_csv='0output.csv'):
    """
    执行数据加载、过滤、基础清洗和划分。
    """
    print("--- 开始简化版数据预处理 (为BERT准备) ---")

    # 检查输入文件
    if not os.path.exists(input_csv):
        print(f"错误：输入文件 '{input_csv}' 不存在。")
        return

    # 1. 加载数据
    df = pd.read_csv(input_csv)
    print(f"1. 成功加载数据，共 {len(df)} 条记录。")

    # 2. 过滤数据
    df = df[df['性别'] != '未知'].copy()
    df.dropna(subset=['消息内容'], inplace=True)
    df = df[df['消息内容'].str.strip() != '']
    print(f"2. 过滤'未知'性别和空消息后，剩余 {len(df)} 条记录。")

    # 3. 清洗文本
    print("3. 正在对文本进行基础清洗（移除引用、URL等）...")
    df['clean_msg'] = df['消息内容'].apply(clean_text_for_bert)
    
    # 再次过滤，丢掉清洗后内容为空的行
    df = df[df['clean_msg'].str.strip() != ''].copy()
    final_count = len(df)
    print(f"   清洗完成。最终有效记录共 {final_count} 条。")
    
    if final_count < 10:
        print("警告：有效数据过少，可能无法成功划分数据集。")
        return

    # 4. 划分数据集
    # stratify=df['性别'] 确保各数据集中男女比例均衡
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['性别']
    )
    validation_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['性别']
    )
    print("4. 数据集划分完成:")
    print(f"   - 训练集: {len(train_df)} 条")
    print(f"   - 验证集: {len(validation_df)} 条")
    print(f"   - 测试集: {len(test_df)} 条")

    # 5. 保存文件
    train_df.to_csv('train.csv', index=False, encoding='utf-8-sig')
    validation_df.to_csv('validation.csv', index=False, encoding='utf-8-sig')
    test_df.to_csv('test.csv', index=False, encoding='utf-8-sig')
    print("5. 所有数据集已成功保存为 train.csv, validation.csv, test.csv。")
    print("\n--- 预处理完成！现在可以运行训练脚本了。---")


if __name__ == "__main__":
    preprocess_data()