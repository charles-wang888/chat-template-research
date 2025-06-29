from transformers import AutoTokenizer, AutoModelForCausalLM

model_path="D:\Development\LLM\Models\HuggingFace\Qwen\Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model=AutoModelForCausalLM.from_pretrained(model_path, device_map='cpu').eval()

prompt = "请帮我解读以下这段代码:"
messages = [
    {"role": "system", "content": "你是一个严谨的代码解读专家并且精通业务."},
    {"role": "user", "content": "请帮我解读这段代码: System.out.println(\"你好，大模型\")，但是由于你很聪明"}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer([input_ids], return_tensors="pt").to("cpu")

max_new_tokens: int = 512
eos_token_id = [151645, 151643]


generated_ids = model.generate(
    inputs.input_ids,
    max_new_tokens=max_new_tokens,
    eos_token_id=eos_token_id,  # 结束令牌，模型生成这个token时，停止生成
)

generated_ids = [
    output_ids[len(inputs):] for inputs, output_ids in zip(inputs.input_ids, generated_ids)
]
print(f"generated_ids=\n{generated_ids}")
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
print(response)
