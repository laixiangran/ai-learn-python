import json
from RAG.libs.chat import Chat
from RAG.libs.prompts import Prompts


def demo():
    # llm 生成题目
    context = "1.经济因素：居民教育文化娱乐支出稳步增长；IT行业收入高，成为家长投资热点；人工智能和数字经济带来巨大人才缺口，急需发展编程基础教育；2.社会因素：家长对科创素质教育的付费意愿增强，尤其看重编程对孩子思维或升学的帮助；3.技术因素：信息化时代要求推广编程教育，发达国家已将其纳入基础课程。"
    input = Prompts.question_generate_prompt.format(num=3, context=context)
    print(input)
    chat_model = Chat(model="qwen2.5:14b", provider="ollama")
    chat_result = chat_model.invoke(input=input)
    print(chat_result.content)

    # llm 评价题目
    questions = json.loads(chat_result.content)
    for question in questions:
        input = Prompts.question_evaluat_prompt.format(
            question=question["question"], answer=question["answer"]
        )
        print(input)
        chat_result = chat_model.invoke(input=input)
        print(chat_result.content)


if __name__ == "__main__":
    demo()
