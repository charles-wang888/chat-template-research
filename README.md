# chat-template-research
关于chat-template的研究


# 背景

与 `chat_template` 相关的几个问题：

1. 不是 chat 模型导致 LLM 输出无法对齐真实期望结果的问题。
2. 去年解决的 Bad Case，即未使用 chat_template 而导致 LLM 做不着边际的输出问题。
3. 想通过指定特定的 token（非 LLM 默认的 eos_token）作为 LLM 输出终止条件的问题。
4. Qwen3 的快思考/慢思考问题。

因此，想深入研究下 chat_template，探索其中的原理，并从底层了解遇到的 Bad Case 的本质。

---

## 什么是 chat_template？

在聊天上下文中，模型是由一个或多个消息组成的对话，每个消息都包含角色（如 user 或 system）和消息文本。不同模型对聊天输入格式有自己的要求，这就是 chat_template 的作用——它把问答的对话内容转化为模型的输入 prompt。

---

## 不是 chat 的模型会怎样？

以 Llama-2-7b-hf 大模型为例，它不是 chat 模型，所以如果要用 `tokenizer.apply_chat_template` 时，会报错：

> Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed!

---

## 对于 chat 模型，如何查看 chat_template 渲染效果？

### 方式一

使用 `AutoTokenizer.from_pretrained(模型本地根目录)` 获得 tokenizer，再 `apply_chat_template`。  
以 qwen2:0.5b 为例，应用 chat_template 后会显示 system、user、assistant 的三段式，每段都以 `<|im_start|>` 和 `<|im_end|>` 作为头尾标识符。  
chat_template 的结果就是把 message 中的 role、content 等字段分别填入模板，构造出最终发给 LLM 的 Prompt。

chat_template 的逻辑：

1. 根据是否有 system 角色，如果没有就用默认系统提示词 "You are a helpful assistant"，否则用用户自定义的 system 内容。系统提示词放在 `<|im_start|>system` 与 `<|im_end|>` 之间。
2. 用户的指令（role 为 user 的 content）放在 `<|im_start|>user` 与 `<|im_end|>` 之间。
3. 根据 `add_generation_prompt` 开关判断是否拼接 `<|im_start|>assistant`。assistant 表示模型生成的回复内容，只有开启该开关，LLM 才会生成回复内容。

其中 `<|im_start|>`、`<|im_end|>`、`<|endoftext|>` 都是 Qwen 预定义的 special token。

### 方式二

使用 `AutoTokenizer.from_pretrained(模型id)` 获得 tokenizer，再 `apply_chat_template`。  
以 qwen2-1.5b 为例，chat_template 与 qwen2:0.5b 一样，都是 system、user、assistant 三段式，每段以 `<|im_start|>` 和 `<|im_end|>` 作为头尾标识符。  
qwen2.5-1.5b 与 qwen3-4b 的输出也类似。qwen2.5 和 qwen3 的 chat_template 中包含了 tool 的判断和 think 的支持，模板更复杂，但实验未用到这些能力，实际输入与 qwen2 一致。

---

## 复盘 Bad Case：未设置 eos_token_id 导致 LLM 输出不停

做一组对比实验测试 eos_token_id 的作用。  
假设让 LLM 解读一段代码并返回结果，如果注释掉第 27 行的 `eos_token_id=eos_token_id`，即不提供 eos_token_id，则 LLM 会一直输出直到最大 token 数，且输出内容与问题无关。这是去年遇到的 Bad Case。

---

## 原理分析

配置了 eos_token_id 后，相当于告诉 LLM 这是内容的结束符，遇到 eos_token_id 就代表对话或文本序列结束。

- 在训练数据中使用 eos_token，可以帮助模型学习何时停止生成对话或文本序列。
- 在推理时使用 eos_token，可以让 LLM 判断何时停止生成。

如果 `skip_special_tokens=False`，不会跳过特殊 token，因此 LLM 最终输出时会把 `<|im_end|>` 打印出来。  
配置了 `eos_token_id=[151645,151643]`，说明 LLM 输出时遇到这些特殊字符就会停止。

LLM 为什么遇到结束符就停止？  
可以 debug `utils.py` 的 `_get_stopping_criteria.py`，定义了两个停止条件（Stopping Criteria）：

- 最大长度停止条件：`MaxLengthCriteria`
- 基于 EosToken 的停止条件：`EosTokenCriteria`

无论哪个条件先满足，都会让 LLM 停止输出。

---

## 自定义 LLM 的输出终止条件 StoppingCriteria

如果想自定义停止词，比如遇到“由于”就停止输出，而不是默认的 `<|im_end|>` 或 `<|endoftext|>`，可以自定义 StoppingCriteria。

- 自定义 StoppingCriteria 类（如 CharlesStoppingCriteria），覆写其核心方法 `__call__`。
- 入参 `input_ids` 是当前已生成的完整序列，`scores` 是下一个 token 的概率分数（通常不用）。
- 返回 True 表示停止生成，False 表示继续。

使用方式：

1. 定义 stop_word，比如“由于”。
2. 基于 stop_word 构造 CharlesStoppingCriteria。
3. 把 CharlesStoppingCriteria 实例添加到 stopping_criteria_list。
4. 设置较大的最大 token 数，优先触发自定义停止条件。
5. 把 stopping_criteria_list 和最大 token 数作为参数传给 `model.generate()`。

执行后，输出遇到“由于”就停止，而不是完整解读完毕再停止。

---

## 关键知识总结

1. 之前遇到 LLM 无限输出直到 token 上限的 Bad Case，主要原因是没有设置 eos_token_id，即没有指定停止词。停止词在训练和推理阶段都会用到。
2. 不同大模型的停止词不同，qwen 系列默认的 eos_token 是 `<|endoftext|>` 和 `<|im_end|>`。
3. 大模型停止输出的条件封装在 StoppingCriteria 中，包括 max_token（最大 token 数）和 eos_token（停止词），任一条件满足即停止输出。
4. 可以自定义 StoppingCriteria，指定想要的停止词，让大模型遇到自定义停止词时停止输出。
