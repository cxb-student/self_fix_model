import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和 tokenizer
model_path = "/transformer/model_with_fine_turned/llama3-8b/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

# 检查是否有 GPU
device = torch.device("cpu")
model.to(device)

while True:
    text = input("请输入问题：")
    if text.lower() == "exit":
        break

    # 编码输入文本
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    # 生成文本
    output_ids = model.generate(input_ids, max_length=256, do_sample=True, top_p=0.9, temperature=0.7)

    # 解码输出
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Generated Text:", generated_text)