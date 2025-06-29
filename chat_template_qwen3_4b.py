from transformers import AutoTokenizer
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='True'

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4b")

prompt = "请帮我解读以下这段代码:"
messages = [
    {"role": "system", "content": "你是一个严谨的代码解读专家并且精通业务."},
    {"role": "user", "content": "请帮我解读这段代码: System.out.println(\"你好，大模型\")"}
]
tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

print("qwen3:4b应用了聊天模板后的输入:\n\n"+tokenizer.decode(tokenized_chat[0]))
