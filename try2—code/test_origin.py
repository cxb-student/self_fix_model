import re
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, max_answer_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_answer_length = max_answer_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        question = sample["question"] + "\nAnswer as short as possible,do not print the way you think,only output answer is best./nnow please write your answer:"
        answer = sample["answer"]
        question_enc = self.tokenizer(
            question,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',  # 添加 padding 参数
            return_tensors="pt"
        )
        answer_enc = self.tokenizer(
            answer,
            max_length=self.max_answer_length,
            truncation=True,
            padding='max_length',  # 添加 padding 参数
            return_tensors="pt"
        )

        return {
            "input_ids": question_enc["input_ids"].squeeze(0),
            "answer_ids": answer_enc["input_ids"].squeeze(0),
        }

def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence([item["input_ids"] for item in batch], batch_first=True,
                                                padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()  # 生成 attention_mask
    answer_ids = torch.nn.utils.rnn.pad_sequence([item["answer_ids"] for item in batch], batch_first=True,
                                                 padding_value=tokenizer.pad_token_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "answer_ids": answer_ids,
    }

dataset = load_dataset("gsm8k", "main", split="test")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
test_dataset = QADataset(dataset, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)
def extract_number(text):
    text = text.strip()
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)  # 兼容整数和小数
    return numbers[-1] if numbers else None
def jaccard_similarity(str1, str2):
    set1, set2 = set(str1.split()), set(str2.split())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 设置 pad_token = eos_token


# 评估函数
def evaluate(model, tokenizer, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                num_beams=5,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id
            )

            predicted_answers = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            real_answers = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["answer_ids"]]

            for pred, real in zip(predicted_answers, real_answers):
                pred_num = extract_number(pred)
                real_num = extract_number(real)

                # 计算正确率
                if pred_num and real_num:
                    if pred_num == real_num:
                        correct += 1
                else:
                    similarity = jaccard_similarity(pred, real)
                    if similarity > 0.7:
                        correct += 1
                    else:
                        print(pred)
                print(f"前{total}轮的正确率")
                print(correct / total)if total > 0 else 0

                total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"原模型数学能力评估：准确率 = {accuracy:.4f}")


# 运行评估
evaluate(model, tokenizer, test_dataloader, device)