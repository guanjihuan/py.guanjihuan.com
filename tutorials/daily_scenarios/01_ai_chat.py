# AI 对话
import guan

response, history = guan.openai_chat("你好", model="qwen-plus") # 需要 API Key
print()

response = guan.langchain_chat_without_memory(prompt="你好", model="qwen-plus") # 需要 API Key
response == guan.langchain_chat_with_memory(prompt="你好", model="qwen-plus") # 需要 API Key
print()

response = guan.ollama_chat(prompt='你好', model="qwen2.5:0.5b")  # 需要运行 Ollama 和下载对应的模型
print()

llm = guan.load_llama_model(model_path="D:/models/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf") # 需要有模型文件
response, history = guan.llama_chat(llm, '你好 /no_think')
print()

model, tokenizer = guan.load_modelscope_model(model_name="D:/models/Qwen/Qwen3-0.6B") # 需要有模型文件
response, history = guan.modelscope_chat(model, tokenizer, '你好 /no_think')
