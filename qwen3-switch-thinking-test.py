from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

prompt = "请帮我解读以下这段代码:"
messages = [
    {"role": "system", "content": "你是一个严谨的代码解读专家并且精通业务."},
    {"role": "user", "content": "请帮我解读这段代码: System.out.println(\"你好，大模型\")"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
streamer = TextStreamer(tokenizer)
model.generate(**model_inputs, max_new_tokens=500, temperature=0.2, do_sample=True, streamer=streamer, pad_token_id=tokenizer.eos_token_id)

