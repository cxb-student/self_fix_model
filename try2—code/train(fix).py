import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from datasets import load_dataset
from torch.optim import AdamW


#用来拼接decoder输出和putput1
class FusionModule(nn.Module):
    def __init__(self, hidden_size):
        super(FusionModule, self).__init__()
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=-1)  # (hidden_size*2)
        out = self.fc1(combined)
        out = self.relu(out)
        out = self.fc2(out)
        fused = self.layernorm(x1 + out)
        return fused



#分句函数，没用transformer自带的
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


#数据集，不一样的地方在于句子的开启和末尾也包含了
class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256, max_answer_length=128):
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


#模型本体
class self_fix_Model(nn.Module):
    def __init__(self, model_name, decoder_hidden_size=None,lora_rank=8):
        super(self_fix_Model, self).__init__()
        self.transformer = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.hidden_size = self.transformer.config.hidden_size
        self.vocab_size = self.transformer.config.vocab_size
        if decoder_hidden_size is None:
            decoder_hidden_size = self.hidden_size
        self.logits_to_hidden = nn.Linear(self.vocab_size, self.hidden_size)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_size, decoder_hidden_size)
        )
        self.fusion = FusionModule(self.hidden_size)
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        self.transformer = get_peft_model(self.transformer, lora_config)

    def forward(self, input_ids, answer_ids):
        for param in self.transformer.parameters():
            param.requires_grad = False  # 冻结 transformer 参数

        device = input_ids.device
        batch_size = input_ids.size(0)

        # 使用常规前向传递而不是 generate
        outputs1 = self.transformer(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs1.hidden_states[-1]  # 获取最后一层的 hidden states
        logits = outputs1.logits  # 获取 logits

        bounds_list = []
        generated_texts = []
        for i in range(batch_size):
            output_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
            generated_texts.append(output_text)
            bounds = detect_sentence_boundaries(output_text, self.tokenizer)
            bounds_list.append(bounds)

        embed_tokens = self.transformer.get_input_embeddings()
        input_embeds = embed_tokens(input_ids)

        fused_sent_list = []
        for i in range(batch_size):
            fused_list = []
            for (start, end) in bounds_list[i]:
                if end > hidden_states.size(1):
                    continue

                sent_hidden = hidden_states[i, start:end, :]
                sent_hidden_avg = sent_hidden.mean(dim=0)

                sent_logits = logits[i, start:end, :]
                sent_logits_avg = sent_logits.mean(dim=0)
                sent_logits_proj = self.logits_to_hidden(sent_logits_avg)

                decoder_out = self.decoder(sent_logits_proj)
                fused = self.fusion(sent_hidden_avg, decoder_out)
                fused_list.append(fused.unsqueeze(0))

            if fused_list:
                fused_sent_tensor = torch.cat(fused_list, dim=0)
            else:
                fused_sent_tensor = torch.empty(0, input_embeds.size(-1), device=device)
            fused_sent_list.append(fused_sent_tensor)

        new_embeds_list = []
        for i in range(batch_size):
            inp_emb = input_embeds[i]
            fused_sent = fused_sent_list[i]
            new_embeds = torch.cat([inp_emb, fused_sent], dim=0)
            new_embeds_list.append(new_embeds)

        new_embeds_padded = torch.nn.utils.rnn.pad_sequence(new_embeds_list, batch_first=True)

        answer_embeds = embed_tokens(answer_ids)
        inputs_embeds_2 = torch.cat([new_embeds_padded, answer_embeds], dim=1)
        total_len = inputs_embeds_2.size(1)
        new_seq_len = new_embeds_padded.size(1)

        labels = torch.full((batch_size, total_len), -100, device=device, dtype=torch.long)
        labels[:, new_seq_len:] = answer_ids

        outputs2 = self.transformer(inputs_embeds=inputs_embeds_2, labels=labels)
        loss = outputs2.loss

        final_logits = outputs2.logits
        final_pred_ids = torch.argmax(final_logits, dim=-1)
        final_answer = self.tokenizer.batch_decode(final_pred_ids, skip_special_tokens=True)

        return final_answer, loss


def train():
    device = torch.device("cuda" )


    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    path = "../model.pth"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset("meta-math/GSM8K_zh")
    train_data = ds["train"]
    dataset = QADataset(train_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = self_fix_Model(model_name=model_path)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 20
    model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            answer_ids = batch["answer_ids"].to(device)
            optimizer.zero_grad()
            (ouput_real,loss) = model(input_ids=input_ids, answer_ids=answer_ids)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx} Loss: {loss.item():.4f}")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


if __name__ == '__main__':
    train()