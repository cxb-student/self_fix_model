import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel

# 定义 Transformer Decoder 模型
class ConfidenceEvaluator(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads, dropout=0.1):
        super(ConfidenceEvaluator, self).__init__()
        self.embedding = nn.Linear(hidden_dim, hidden_dim)  # 先降维（如果需要）
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, 3)  # 输出 low/middle/high 三分类
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, logits, past_sentences=None):
        logits = self.embedding(logits)
        memory = None
        if past_sentences is not None:
            memory = past_sentences.permute(1, 0, 2)  # 变一下维度不然进不去

        logits = logits.permute(1, 0, 2)  # (seq_len, batch, hidden_dim)
        output = self.transformer_decoder(logits, memory) if memory is not None else logits
        output = output.permute(1, 0, 2).mean(dim=1)  # 取平均池化，得到 (batch, hidden_dim)
        output = self.fc_out(output)
        return self.softmax(output)  # 返回 softmax 结果

# 参数
hidden_dim = 4096
num_layers = 4
num_heads = 8
model = ConfidenceEvaluator(hidden_dim, num_layers, num_heads)

# 假设数据格式
batch_size = 2
seq_len = 32
logits = torch.randn(batch_size, seq_len, hidden_dim)  # 模拟 logits
past_sentences = torch.randn(batch_size, seq_len, hidden_dim)  # 模拟过去输入

# 计算置信度分类
output = model(logits, past_sentences)  # 输出 (batch, 3)
predicted_class = torch.argmax(output, dim=-1)  # 0=low, 1=middle, 2=high

# 训练逻辑
def train_model(model, dataloader, epochs=5, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for logits, past_sentences, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(logits, past_sentences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss={loss.item()}")

# 示例：创建随机数据集进行训练
logits_fake = torch.randn(100, seq_len, hidden_dim)
past_sentences_fake = torch.randn(100, seq_len, hidden_dim)
labels_fake = torch.randint(0, 3, (100,))  # 0=low, 1=middle, 2=high

dataset = torch.utils.data.TensorDataset(logits_fake, past_sentences_fake, labels_fake)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

train_model(model, dataloader)