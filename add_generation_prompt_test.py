from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("D:\Development\LLM\Models\HuggingFace\Qwen\Qwen2-0.5B")

prompt = "请帮我解读以下这段代码:"
messages = [
    {"role": "system", "content": "你是一个严谨的代码解读专家并且精通业务."},
    {"role": "user", "content": "请帮我解读这段代码: System.out.println(\"你好，大模型\")"}
]
tokenized_chat1 = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

tokenized_chat2 = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt"
)


print("add_generation_prompt 开关测试\n")
print("qwen2:0.5b应用了聊天模板（且add_generation_prompt=True）后的输入:\n\n"+tokenizer.decode(tokenized_chat1[0]))
print('\n')
print("qwen2:0.5b应用了聊天模板（且add_generation_prompt=False）后的输入:\n\n"+tokenizer.decode(tokenized_chat2[0]))
