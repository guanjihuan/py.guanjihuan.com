# Module: AI_chat

# AI 对话（当前默认模型为 'hunyuan-lite'，无记忆）
def chat(prompt='你好', model=1, stream=1, stream_label=0):
    import requests
    url = "http://api.guanjihuan.com/chat"
    data = {
        "prompt": prompt, 
        "model": model,
    }
    if stream == 1:
        if stream_label == 1:
            print('\n--- Start Chat Stream Message ---\n')
    requests_response = requests.post(url, json=data, stream=True)
    response = ''
    if requests_response.status_code == 200:
        for line in requests_response.iter_lines():
            if line:
                if stream == 1:
                    print(line.decode('utf-8'), end='', flush=True)
                response += line.decode('utf-8')
        print()
    else:
        pass
    if stream == 1:
        if stream_label == 1:
            print('\n--- End Chat Stream Message ---\n')
    return response

# 加上函数代码的 AI 对话（当前默认模型为 'hunyuan-lite'，无记忆）
def chat_with_function_code(function_name, prompt='', model=1, stream=1):
    import guan
    function_source = guan.get_source(function_name)
    if prompt == '':
        response = guan.chat(prompt=function_source, model=model, stream=stream)
    else:
        response = guan.chat(prompt=function_source+'\n\n'+prompt, model=model, stream=stream)
    return response

# 机器人自动对话（当前默认模型为 'hunyuan-lite'，无记忆）
def auto_chat(prompt='你好', round=2, model=1, stream=1):
    import guan
    response0 = prompt
    for i0 in range(round):
        print(f'\n【对话第 {i0+1} 轮】\n')
        print('机器人 1: ')
        response1 = guan.chat(prompt=response0, model=model, stream=stream)
        print('机器人 2: ')
        response0 = guan.chat(prompt=response1, model=model, stream=stream)

# 机器人自动对话（引导对话）（当前默认模型为 'hunyuan-lite'，无记忆）
def auto_chat_with_guide(prompt='你好', guide_message='（回答字数少于30个字，最后反问我一个问题）', round=5, model=1, stream=1):
    import guan
    response0 = prompt
    for i0 in range(round):
        print(f'\n【对话第 {i0+1} 轮】\n')
        print('机器人 1: ')
        response1 = guan.chat(prompt=response0+guide_message, model=model, stream=stream)
        print('机器人 2: ')
        response0 = guan.chat(prompt=response1+guide_message, model=model, stream=stream)

# 通过 LangChain 加载模型（需要 API Key)
def load_langchain_model(model="qwen-plus", temperature=0.7, load_env=1):
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    import os
    if load_env:
        import dotenv
        from pathlib import Path
        import inspect
        caller_frame = inspect.stack()[1]
        caller_dir = Path(caller_frame.filename).parent
        env_path = caller_dir / ".env"
        if env_path.exists():
            dotenv.load_dotenv(env_path)
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        model=model,
        temperature=temperature,
        streaming=True,
    )
    return llm

# 使用 LangChain 无记忆对话（需要 API Key)
def langchain_chat_without_memory(prompt="你好", model="qwen-plus", temperature=0.7, system_message=None, print_show=1, load_env=1):
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    import os
    if load_env:
        import dotenv
        from pathlib import Path
        import inspect
        caller_frame = inspect.stack()[1]
        caller_dir = Path(caller_frame.filename).parent
        env_path = caller_dir / ".env"
        if env_path.exists():
            dotenv.load_dotenv(env_path)
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        model=model,
        temperature=temperature,
        streaming=True,
    )
    if system_message == None:
        langchain_prompt = ChatPromptTemplate.from_messages([
            ("human", "{question}")
        ])
    else:
        langchain_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{question}")
        ])
    chain = langchain_prompt | llm
    response = ''
    for chunk in chain.stream({"question": prompt}):
        response += chunk.content
        if print_show:
            print(chunk.content, end="", flush=True)
    if print_show:
        print()
    return response

# 使用 LangChain 有记忆对话（记忆临时保存在函数的属性上，需要 API Key)
def langchain_chat_with_memory(prompt="你好", model="qwen-plus", temperature=0.7, system_message=None, session_id="default", print_show=1, load_env=1):
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_community.chat_message_histories import ChatMessageHistory
    import os
    if load_env:
        import dotenv
        from pathlib import Path
        import inspect
        caller_frame = inspect.stack()[1]
        caller_dir = Path(caller_frame.filename).parent
        env_path = caller_dir / ".env"
        if env_path.exists():
            dotenv.load_dotenv(env_path)
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        model=model,
        temperature=temperature,
        streaming=True,
    )
    if system_message == None:
        langchain_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("history"),
            ("human", "{question}") 
        ])
    else:
        langchain_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder("history"),
            ("human", "{question}") 
        ])
    chain = langchain_prompt | llm
    if not hasattr(langchain_chat_with_memory, "store"):
        langchain_chat_with_memory.store = {}
    
    def get_session_history(sid: str):
        if sid not in langchain_chat_with_memory.store:
            langchain_chat_with_memory.store[sid] = ChatMessageHistory()
        return langchain_chat_with_memory.store[sid]
    
    chatbot = RunnableWithMessageHistory(
        chain,
        lambda sid: get_session_history(sid),
        input_messages_key="question",
        history_messages_key="history",
    )
    response = ''
    for chunk in chatbot.stream({"question": prompt}, config={"configurable": {"session_id": session_id}}):
        response += chunk.content
        if print_show:
            print(chunk.content, end="", flush=True)
    if print_show:
        print()
    return response

# 使用 Ollama 本地模型对话（需要运行 Ollama 和下载对应的模型）
def ollama_chat(prompt='你好/no_think', model="qwen3:0.6b", temperature=0.8, print_show=1):
    import ollama
    response_stream = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}], stream=True, options={"temperature": temperature})
    response = ''
    start_thinking = 1
    for part in response_stream:
        response += part['message']['content']
        if print_show == 1:
            thinking = part['message'].get('thinking')
            if thinking is not None:
                if start_thinking == 1:
                    print('<think>')
                    start_thinking = 0
                print(f"{thinking}", end='', flush=True)
            else:
                if start_thinking == 0:
                    print('</think>')
                    start_thinking = 1
                print(part['message']['content'], end='', flush=True)
    if print_show == 1:
        print()
    return response

# ModelScope 加载本地模型和分词器（只加载一次）
def load_modelscope_model(model_name="D:/models/Qwen/Qwen3-0.6B"):
    from modelscope import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

# 使用 ModelScope 本地模型聊天
def modelscope_chat(model, tokenizer, prompt='你好 /no_think', history=[], temperature=0.7, top_p=0.8, print_show=1):
    from threading import Thread
    from transformers import TextIteratorStreamer
    messages = history + [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=32768,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        repetition_penalty=1.2
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    response = ""
    for new_text in streamer:
        if print_show:
            print(new_text, end="", flush=True)
        response += new_text
    if print_show:
        print()
    new_history = history + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    return response, new_history

# LLaMA 加载本地模型（只加载一次）
def load_llama_model(model_path="D:/models/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"):
    from llama_cpp import Llama
    llm = Llama(
        model_path=model_path,
        n_ctx=32768,
        verbose=False,
        chat_format="chatml",
        logits_all=False
    )
    return llm

# 使用 LLaMA 本地模型聊天
def llama_chat(llm, prompt='你好 /no_think', history=[], temperature=0.7, top_p=0.8, print_show=1):
    new_history = history + [{"role": "user", "content": prompt}]
    llm_response = llm.create_chat_completion(
        messages=new_history,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=1.5,
        stream=True,
    )
    response = ''
    for chunk in llm_response:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta:
            token = delta['content']
            response += token
            if print_show:
                print(token, end="", flush=True)
    if print_show:
        print()
    new_history.append({"role": "assistant", "content": response})
    return response, new_history