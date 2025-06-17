from RAG.libs.chat import Chat
from RAG.libs.prompts import Prompts


def demo():
    question = "什么是少儿编程教育？"
    chat_model = Chat(model="qwen2.5:14b", provider="ollama")

    # 问题改写（改写成多个语义相同的新问题）
    input = Prompts.synonymy_rewritten_prompt.format(num=3, question=question)
    print(input)
    chat_result = chat_model.invoke(input=input)
    print(chat_result.content)

    # 问题扩写（分解成多个不同视角的子问题）
    input = Prompts.sub_rewritten_prompt.format(num=3, question=question)
    print(input)
    chat_result = chat_model.invoke(input=input)
    print(chat_result.content)

    # 问题补充上下文（生成相关背景信息）
    input = Prompts.context_supplement_prompt.format(maxLen=200, question=question)
    print(input)
    chat_result = chat_model.invoke(input=input)
    print(chat_result.content)


if __name__ == "__main__":
    demo()
