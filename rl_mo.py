import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.cuda.amp import autocast, GradScaler
import re
from nltk.translate.bleu_score import sentence_bleu


# 提取数字
def extract_number(text):
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return numbers[-1] if numbers else None


# 计算 Jaccard 相似度
def jaccard_similarity(str1, str2):
    set1, set2 = set(str1.split()), set(str2.split())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0


# 检测句子边界
def detect_sentence_boundaries(text, tokenizer):
    sentence_endings = re.finditer(r'([。！？?!\.])', text)
    sentence_char_boundaries = [match.end() for match in sentence_endings]
    if not sentence_char_boundaries:
        sentence_char_boundaries = [len(text)]
    encoded = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = encoded["offset_mapping"].squeeze(0)
    sentence_boundaries = []
    sentence_start_idx = 0
    for char_boundary in sentence_char_boundaries:
        token_end_idx = 0
        for i, (start, end) in enumerate(offsets):
            if end >= char_boundary:
                token_end_idx = i + 1
                break
        if token_end_idx <= sentence_start_idx:
            continue
        sentence_boundaries.append((sentence_start_idx, token_end_idx))
        sentence_start_idx = token_end_idx
    return sentence_boundaries


# 自定义融合模块
class FusionModule(nn.Module):
    def __init__(self, hidden_size):
        super(FusionModule, self).__init__()
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=-1)
        out = self.fc1(combined)
        out = self.relu(out)
        out = self.fc2(out)
        fused = self.layernorm(x1 + out)
        return fused


# 优化解码器
class OptimizedDecoder(nn.Module):
    def __init__(self, hidden_size, num_heads=12, dim_feedforward=4096, num_layers=6):
        super(OptimizedDecoder, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

    def forward(self, tgt, memory):
        return self.decoder(tgt=tgt, memory=memory)


# 自定义数据集
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
        question = sample["question"]
        answer = sample["answer"]
        question_enc = self.tokenizer(
            question,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        answer_enc = self.tokenizer(
            answer,
            max_length=self.max_answer_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        return {
            "input_ids": question_enc["input_ids"].squeeze(0),
            "attention_mask": question_enc["attention_mask"].squeeze(0),
            "answer_ids": answer_enc["input_ids"].squeeze(0),
            "answer_mask": answer_enc["attention_mask"].squeeze(0)
        }


# 对比损失
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, pred, target):
        cos_sim = F.cosine_similarity(pred, target)
        loss = (1 - cos_sim).mean()
        return loss


# 奖励函数类
class RewardFunction:
    def __init__(self, threshold=0.7, confidence_penalty=0.2):
        self.threshold = threshold
        self.confidence_penalty = confidence_penalty

    def get_reward(self, generated_text, real_answer, logits, labels):
        similarity = self.jaccard_similarity(generated_text, real_answer)
        if similarity > 0.9:
            reward = 1.0
        elif similarity > 0.7:
            reward = 0.5
        else:
            reward = -0.5
        confidence = self.calculate_confidence(logits, labels)
        if confidence < self.confidence_penalty:
            reward *= 0.5
        return reward

    def calculate_confidence(self, logits, labels):
        prob = F.softmax(logits, dim=-1)
        max_prob = prob.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        return max_prob.mean().item()

    def jaccard_similarity(self, str1, str2):
        set1, set2 = set(str1.split()), set(str2.split())
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union else 0


# 强化学习损失
class ReinforceLoss(nn.Module):
    def forward(self, logits, labels, rewards):
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1))
        loss = -selected_log_probs.squeeze(-1) * rewards
        return loss.mean()


# 自修正模型
class SelfFixModel(nn.Module):
    def __init__(self, model_name, decoder_hidden_size=None, lora_rank=8):
        super(SelfFixModel, self).__init__()
        self.transformer = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.pad_token is None else self.tokenizer.pad_token
        self.hidden_size = self.transformer.config.hidden_size
        self.vocab_size = self.transformer.config.vocab_size

        self.logits_to_hidden = nn.Linear(self.vocab_size, self.hidden_size)
        self.decoder = OptimizedDecoder(self.hidden_size)
        self.fusion = FusionModule(self.hidden_size)
        self.reward_function = RewardFunction()

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        self.transformer = get_peft_model(self.transformer, lora_config)

    @autocast()
    def forward(self, input_ids, answer_ids, rewards=None):
        for param in self.transformer.parameters():
            param.requires_grad = False

        batch_size = input_ids.size(0)
        outputs1 = self.transformer(input_ids=input_ids, output_hidden_states=True, return_dict=True)
        hidden_states = outputs1.hidden_states[-1]
        logits = outputs1.logits

        fused_sent_list = []
        for i in range(batch_size):
            fused_list = []
            text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
            bounds = detect_sentence_boundaries(text, self.tokenizer)
            for (start, end) in bounds:
                sent_hidden = hidden_states[i, start:end, :]
                sent_logits = logits[i, start:end, :]
                sent_logits_proj = self.logits_to_hidden(sent_logits.mean(dim=0).unsqueeze(0))
                decoder_out = self.decoder(tgt=sent_logits_proj.unsqueeze(0), memory=sent_hidden.unsqueeze(1))
                fused = self.fusion(sent_hidden.mean(dim=0), decoder_out.squeeze(0))
                fused_list.append(fused.unsqueeze(0))
            fused_sent_tensor = torch.cat(fused_list, dim=0) if fused_list else torch.empty(0, self.hidden_size, device=input_ids.device)
            fused_sent_list.append(fused_sent_tensor)

        new_embeds_list = []
        for i in range(batch_size):
            inp_emb = self.transformer.get_input_embeddings()(input_ids[i])
            fused_sent = fused_sent_list[i]
            new_embeds = torch.cat([inp_emb, fused_sent], dim=0)
            new_embeds_list.append(new_embeds)
        new_embeds_padded = torch.nn.utils.rnn.pad_sequence(new_embeds_list, batch_first=True)
        answer_embeds = self.transformer.get_input_embeddings()(answer_ids)
        inputs_embeds_2 = torch.cat([new_embeds_padded, answer_embeds], dim=1)

        total_len = inputs_embeds_2.size(1)
        new_seq_len = new_embeds_padded.size(1)
        labels = torch.full((batch_size, total_len), -100, device=input_ids.device, dtype=torch.long)
        labels[:, new_seq_len:] = answer_ids

        outputs2 = self.transformer(inputs_embeds=inputs_embeds_2, labels=labels)
        loss = outputs2.loss
        final_logits = outputs2.logits
        final_pred_ids = torch.argmax(final_logits, dim=-1)
        final_answer = self.tokenizer.batch_decode(final_pred_ids, skip_special_tokens=True)

        if rewards is not None:
            reinforce_loss = ReinforceLoss()(final_logits, answer_ids, rewards)
            return final_answer, loss + reinforce_loss
        else:
            return final_answer, loss


# 评估函数
def evaluate(model, tokenizer, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    bleu_scores = []
    jaccard_scores = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            answer_ids = batch["answer_ids"].to(device)
            real_answers = [tokenizer.decode(ids, skip_special_tokens=True) for ids in answer_ids]

            predicted_answers, _ = model(input_ids=input_ids, answer_ids=answer_ids)

            for pred, real in zip(predicted_answers, real_answers):
                pred_num = extract_number(pred)
                real_num = extract_number(real)

                if pred_num and real_num and pred_num == real_num:
                    total_correct += 1
                elif not pred_num or not real_num:
                    similarity = jaccard_similarity(pred, real)
                    if similarity > 0.7:
                        total_correct += 1

                bleu = sentence_bleu([real.split()], pred.split())
                bleu_scores.append(bleu)
                jaccard = jaccard_similarity(pred, real)
                jaccard_scores.append(jaccard)
                total_samples += 1

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0

    print(f"模型评估：准确率 = {accuracy:.4f}, BLEU = {bleu:.4f}, Jaccard = {jaccard:.4f}")
    return accuracy, bleu, jaccard


# 训练函数
def train(model, tokenizer, train_dataloader, test_dataloader, device, num_epochs=10, save_path="model.pth"):
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            answer_ids = batch["answer_ids"].to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(input_ids=input_ids, answer_ids=answer_ids, rewards=None)
                loss = outputs[1]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        evaluate(model, tokenizer, test_dataloader, device)
        print(f"Epoch {epoch + 1} - Loss: {total_loss:.4f}")
        torch.save(model.state_dict(), f"{save_path}_epoch_{epoch + 1}.pth")

    print("训练完成！")


# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "Qwen/Qwen2-VL-7B-Instruct"

    # 加载数据集
    dataset1 = load_dataset("gsm8k", split="train")
    dataset2 = load_dataset("math_qa", split="train")
    dataset3 = load_dataset("grade_school_math_instructions", split="train")

    # 合并数据集
    combined_dataset = []
    combined_dataset.extend(dataset1)
    combined_dataset.extend(dataset2)
    combined_dataset.extend(dataset3)

    # 初始化数据集
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = QADataset(combined_dataset, tokenizer)
    test_dataset = QADataset(dataset1, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4)

    # 初始化模型
    model = SelfFixModel(model_name=model_path)
    model.to(device)

    # 训练
    train(model, tokenizer, train_dataloader, test_dataloader, device)


if __name__ == "__main__":
    main()