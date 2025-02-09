import re
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from train_deocder import self_fix_Model,QADataset  # 确保 self_fix_Model 在 model.py 中

# 载入测试数据集（GSM8K 英文版）
dataset = load_dataset("gsm8k", "main", split="test")

# 载入 tokenizer 和模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_save_path = "../model.pth"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = self_fix_Model(model_name=model_path)
model.load_state_dict(torch.load(model_save_path))  # 加载训练好的模型权重
model.to(device)

# 构造测试数据集
test_dataset = QADataset(dataset, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=4)

# 提取答案中的数值（忽略文字干扰）
def extract_number(text):
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)  # 提取整数或小数
    return numbers[-1] if numbers else None  # 取最后一个数（通常是答案）

# 计算 Jaccard 相似度（适用于文本答案）
def jaccard_similarity(str1, str2):
    set1, set2 = set(str1.split()), set(str2.split())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0

# 评估函数
def evaluate(model, tokenizer, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            answer_ids = batch["answer_ids"].to(device)

            predicted_answers, _ = model(input_ids=input_ids, answer_ids=answer_ids)
            real_answers = [tokenizer.decode(ids, skip_special_tokens=True) for ids in answer_ids]

            # 计算准确率
            for pred, real in zip(predicted_answers, real_answers):
                pred_num = extract_number(pred)
                real_num = extract_number(real)
                print(f"预测: {pred}, 提取数字: {pred_num}")
                print(f"真实: {real}, 提取数字: {real_num}")
                print("------")

                if pred_num and real_num:
                    correct += 1 if pred_num == real_num else 0
                else:
                    similarity = jaccard_similarity(pred, real)
                    correct += 1 if similarity > 0.7 else 0  # 相似度 > 70% 认为正确

                total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"数学能力评估：准确率 = {accuracy:.4f}")

# 运行评估
evaluate(model, tokenizer, test_dataloader, device)