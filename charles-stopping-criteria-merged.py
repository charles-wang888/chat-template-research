import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

#定义一个Charles自定义停止条件类CharlesStoppingCriteria, 这个类的_call方法会在模型每生成一个token后被调用
class CharlesStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_sequence, tokenizer, input_length, device):
        # 将停止序列编码为 token ID
        self.stop_sequence_ids = tokenizer.encode(stop_sequence, add_special_tokens=False)
        self.stop_sequence_length = len(self.stop_sequence_ids)
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.device = device

    """
     核心方法 _call__,它将在LLM生成每个token时被调用
       入参1:input_ids 是当前已经生成的完整序列 (包括输入部分)
       入参2:scores 是下一个 token 的概率分数 (通常不用于停止，除非是基于置信度的停止)
       返回：True 表示停止生成，False表示继续生成
    """
    def  __call__(self, input_ids: torch.LongTensor, scores:torch.FloatTensor, **kwargs)-> bool:


        # 提取新生成的令牌
        generated_ids = input_ids[0, self.input_length:]

        # 如果新生成的序列长度小于停止序列长度，就继续生成
        if generated_ids.shape[-1] < self.stop_sequence_length:
            return False

        # 检查新生成序列的末尾（generated_ids 的最后 self.stop_sequence_length 个令牌）是否匹配停止序列
        last_generated_tokens = generated_ids[-self.stop_sequence_length:]

        # 将停止序列 ID转为tensor张量并与generated_ids进行比较
        stop_ids_tensor = torch.tensor(self.stop_sequence_ids, dtype=torch.long, device=self.device)

        # 使用 torch.equal 进行精确比较
        if torch.equal(last_generated_tokens, stop_ids_tensor):
            return True

        return False


model_path="D:\Development\LLM\Models\HuggingFace\Qwen\Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model=AutoModelForCausalLM.from_pretrained(model_path, device_map='cpu').eval()

messages = [
    {"role": "system", "content": "你是一个严谨的代码解读专家并且精通业务."},
    {"role": "user", "content": "请帮我解读这段代码: System.out.println(\"你好，由于大模型\"),"}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([input_ids], return_tensors="pt").to("cpu")
eos_token_id = [151645, 151643]

#这里构造一个停止词，比如当模型生成"由于”token的时候就停止输出，并基于stop_word来实例化CharlesStoppingCriteria（Charles自定义停止条件类）
stop_word ="由于"# 例如，我们希望模型在生成 "由于" 时停止
custom_stop_criteria = CharlesStoppingCriteria(
    stop_word, tokenizer, model_inputs.input_ids.shape[1], "cpu"
)

#把当前构造的停止条件CharlesStoppingCriteria添加到停止条件列表stopping_criteria_list中
stopping_criteria_list = StoppingCriteriaList([custom_stop_criteria])


# 设置一个较大的 max_new_tokens ，这样确保有机会遇到停止序列，而不是直接就超过max token上限了。
max_tokens_for_custom_stop =200
print(f"设定最大停止token数:{max_tokens_for_custom_stop}")
print(f"传入用户自定义停止条件的stop word为:{stop_word}")
print('\n\n')


#generated_ids_custom_no_stop是不用停止条件的结果。
#把最大停止token数,eos_token_id传递给model.generate方法
generated_ids_custom_no_stop = model.generate(
    **model_inputs,
    eos_token_id=eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=max_tokens_for_custom_stop,
    num_return_sequences=1
)

#generated_ids_custom_stop是采用停止条件的结果。
#用把最大停止token数，eos_token_id,以及停止条件列表stopping_criteria_list传递给model.generate方法
generated_ids_custom_stop = model.generate(
    **model_inputs,
    eos_token_id=eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=max_tokens_for_custom_stop,
    num_return_sequences=1,
    stopping_criteria=stopping_criteria_list
)

print('使用与不使用停止词的实验对照组结果如下：\n')
#获得并打印不用停止词的LLM输出的结果
output_ids_custom_no_stop=generated_ids_custom_no_stop[0][len(model_inputs.input_ids[0]):].tolist()
print(f"[不用停止词生成的文本为]: {tokenizer.decode(output_ids_custom_no_stop, skip_special_token=True)}")

#获得并打印采用停止词的LLM输出的结果
output_ids_custom_stop = generated_ids_custom_stop[0][len(model_inputs.input_ids[0]):].tolist()
print(f"[用停止词生成的文本为]: {tokenizer.decode(output_ids_custom_stop, skip_special_token=True)}")