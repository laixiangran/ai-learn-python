from RAG.libs.chat import Chat


def ollama_chat_test():
    chat_model = Chat(model="qwen2.5:14b", provider="ollama")
    res = chat_model.invoke(input="什么是少儿编程教育？")
    print(res)


if __name__ == "__main__":
    ollama_chat_test()
