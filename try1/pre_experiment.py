import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = "/transformer/model_with_fine_turned/llama3-8b/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()
class ControlDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads=8, num_classes=3):
        super(ControlDecoder, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.linner1 = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, control_input):
        x = self.input_proj(control_input)  # (seq_len, batch, hidden_dim)
        g = self.transformer_decoder(x,x)
        logits_out = self.classifier(g)
        c_out = torch.sum(logits_out, dim=1)
        c = c_out.argmax(dim=-1)
        return c
class whole_model(nn.Module):
    def __init__(self, main_model, control_decoder):
        super(whole_model, self).__init__()
        self.main_model = main_model
        self.control_decoder = control_decoder
    def forward(self, input,logits_out = [],juzi_out="",mistke_juzi=[]):
        input_ids = tokenizer(input, return_tensors="pt", padding=True, truncation=True)
        input_ids_tensor = torch.tensor(input_ids['input_ids'])
        output = self.main_model(input_ids_tensor)
        logits = output.logits

        c = output.logits.argmax(-1)
        logits_out.append(logits)
        for seq in c:
            output_str = tokenizer.decode(seq, skip_special_tokens=True)
            juzi_out += output_str
        score = 0
        if output=="."or"。":
            logits_tensor = torch.cat(logits_out, dim=0)

            score = self.control_decoder(logits_tensor)
            if score==0:
                mistke_juzi.append(juzi_out)
            juzi_out=""
        score+= score
        return juzi_out,score,mistke_juzi
def main():
    vocab_size = 128256
    control_hidden = 256
    control_layers = 4
    batch_size = 4
    seq_len = 50
    control_seq_len = 20
    num_classes = 3
    for param in model.parameters():
        param.requires_grad = False
    control_input_dim =128256
    control_decoder = ControlDecoder(input_dim=control_input_dim, hidden_dim=control_hidden,
                                     num_layers=control_layers, num_classes=num_classes)
    main_model = whole_model(model,control_decoder)
    pre = main_model("你好")
    print(pre)
    optimizer = torch.optim.Adam(control_decoder.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_steps = 10
    upper_answer = ""
    for step in range(num_steps):
        if step == 0:
            Input = "请开始提问吧并准备为我的回答做出评价以及提出下一个问题"
        else:
            Input = upper_answer

        a = com.answer.split("<shut>")
        input_id = a[0]
        if len(a) == 2:
            target = int(a[1])
        else:
            target = 0
        print(input_id, target)
        outputs = ""
        upper_answer = ""
        outputs, re,score,mistke_juzi = main_model(input_id)
        str_out = tokenizer.decode(outputs, skip_special_tokens=True)
        num_retry = 0
        while re == 1&num_retry<5:
            for i in range(len(mistke_juzi)):
                input_id+=f"{mistke_juzi[i]} is not so correct"
            input_ids = tokenizer(input_id, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
            outputs, re,score,mistke_juzi = main_model(input_ids)
            num_retry+=1
        upper_answer = outputs
        loss = criterion(score, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 保存微调后的控制模块
    torch.save(control_decoder.state_dict(), "../control_decoder_finetuned.pth")
    print("Control module fine-tuning finished.")


if __name__ == "__main__":
    main()