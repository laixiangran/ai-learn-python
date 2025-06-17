from RAG.libs.chat import Chat
from RAG.libs.evaluator import Evaluator


def ollama_chat_test():
    test_data = {
        "question": "生成式AI技术对少儿编程教学有何具体影响？",
        "referenceAnswer": "1.教学模式变革：从“师-生”变为“师-机-生”，更注重个性化学习；2.提升效率：加快教学内容生产，降低门槛；3.降低师资门槛：帮助缓解师资不足问题；4.个性化学习：定制学习素材，优化教学内容。",
        "referenceAnswerStatements": [
            "注重个性化学习",
            "加快教学内容生产，降低门槛",
            "缓解师资不足问题",
            "定制学习素材，优化教学内容",
        ],
        "answer": "生成式AI技术对少儿编程教学的具体影响包括：\n\n1. **推动教育范式的转变**：从供给主导转变为需求主导，即教育更加注重学生的需求和个性化学习体验。\n2. **改变教学模式**：由传统的“师-生”二元结构变为“师-机-生”的三元结构。这使得教师能够更多地关注学生的自主学习能力和创造力的培养，而不是单纯的知识传授。\n3. **提升教学内容生产效率**：利用生成式AI可以更高效、便捷地创建编程教育所需的课程材料和资源，同时降低了对专业师资的需求门槛。\n4. **个性化学习支持**：\n   - 内容定制化生成：根据学生个人的学习需求量身定制合适难度的编程素材。\n   - 课后评估与反馈机制：学生的学业进展可以实时上传到云端，帮助进一步优化个性化的学习材料和体验。\n5. **互动形式创新**：引入少儿专属AI虚拟人物作为辅助角色，在教育过程中提供持续性的陪伴交流，提升课程趣味性和学生参与度。此外，通过深入分析理解每个孩子特点，促进更加精细化的内容开发以更好地满足不同学生的独特需求。\n\n综上所述，生成式人工智能技术为少儿编程教学带来了显著的革新力量，极大促进了教学效率与学习体验两方面的改善与发展。",
        "retrievedContext": [
            "$\\cdot$ 对于学生而言，大模型工具将激发学生主体的动机和潜能，为全过程个性化学习提供重要支撑。具体体现在：内容生成方面，针对个体的学习需求，可定制化生成相应水平的学习素材；课后评估方面，教学成果可实时反馈云端，用于提升个性化内容生产水平；学习互动方面，可开发少儿专属 AI 虚拟人物，一方面持续性陪伴交流，增加趣味性、提升积极性，另一方面挖掘学生多方面特征，助力定制化内容开发。\n\n![](https://cdn-mineru.openxlab.org.cn/extract/c5ab109a-a161-4b6a-97ef-b906c0b2f3c1/03593c2b9cdfe1663b421d473bd5b9688771c1fc32dedada7aaa2382e1bb5a76.jpg)  \n生成式 AI 在少儿编程教育中发挥的作用  \n数据来源：专家访谈、多鲸教育研究院整理",
            "#### 生成式 AI 推动教育范式从供给主导转变为需求主导，生成式 AI 催生“师-机-生”教学模式，提效增趣\n\n$\\cdot$ 生成式 AI 发展，推动编程教学模式从“师-生”二元结构转变为“师-机-生”三元结构。这种结构性变革推动教学从“以教为中心”转变为“以学为中心”，教育范式从供给主导转变为需求主导。与传统的二元结构相比，技术赋能“师-机-生”三元结构，将教师从低效、重复、繁重的知识传授中解放出来，聚焦于更具创造性的方面，注重培养学生的自主学习能力和创造力。\n\n$\\cdot$ 对于教师而言，教学内容的生产方式先后经历 PGC（专业生成内容）到 UGC（用户生成内容），再到 AIGC（人工智能生成内容）的发展演变，每一次变革都在降低教学内容生产的门槛并且提高内容生产效率。生成式 AI 的发展一方面提高了编程教学内容生产的效率，提供了创作虚拟课程材料的便捷途径，另一方面也通过大模型定制化开发降低了编程教育师资建设的门槛。",
            "$\\cdot$ 对于学生而言，大模型工具将激发学生主体的动机和潜能，为全过程个性化学习提供重要支撑。具体体现在：内容生成方面，针对个体的学习需求，可定制化生成相应水平的学习素材；课后评估方面，教学成果可实时反馈云端，用于提升个性化内容生产水平；学习互动方面，可开发少儿专属 AI 虚拟人物，一方面持续性陪伴交流，增加趣味性、提升积极性，另一方面挖掘学生多方面特征，助力定制化内容开发。\n\n![](https://cdn-mineru.openxlab.org.cn/extract/c5ab109a-a161-4b6a-97ef-b906c0b2f3c1/03593c2b9cdfe1663b421d473bd5b9688771c1fc32dedada7aaa2382e1bb5a76.jpg)  \n生成式 AI 在少儿编程教育中发挥的作用  \n数据来源：专家访谈、多鲸教育研究院整理",
        ],
        "answerStatements": [
            "推动教育范式的转变",
            "改变教学模式",
            "提升教学内容生产效率",
            "个性化学习支持",
            "互动形式创新",
        ],
        "simulationQuestions": [
            "生成式AI是如何实现教育范式的转变，从供给主导变为需求主导的？",
            "在新的三元结构中，教师的角色发生了哪些变化？",
            "如何利用生成式AI提升个性化学习支持和互动形式创新？",
        ],
    }
    chat_model = Chat(model="qwen2.5:14b", provider="ollama")

    evaluate_result = {}

    providers = [
        "contextRecall",
        "contextRelevance",
        "faithfulness",
        "answerRelevance",
        "answerCorrectness",
    ]
    force_generate = False
    for provider in providers:
        evaluator = Evaluator(chat_model=chat_model, provider=provider)
        result = 0
        if provider == "contextRecall":
            result = evaluator.evaluate(
                context="\n".join(test_data["retrievedContext"]),
                referenceAnswer=test_data["referenceAnswer"],
                referenceAnswerStatements=test_data["referenceAnswerStatements"],
                force_generate=force_generate,
            )
        elif provider == "contextRelevance":
            result = evaluator.evaluate(
                question=test_data["question"],
                contexts=test_data["retrievedContext"],
            )
        elif provider == "faithfulness":
            result = evaluator.evaluate(
                context="\n".join(test_data["retrievedContext"]),
                answer=test_data["answer"],
                answerStatements=test_data["answerStatements"],
                force_generate=force_generate,
            )
        elif provider == "answerRelevance":
            result = evaluator.evaluate(
                question=test_data["question"],
                answer=test_data["answer"],
                simulationQuestions=test_data["simulationQuestions"],
                force_generate=force_generate,
            )
        elif provider == "answerCorrectness":
            result = evaluator.evaluate(
                answer=test_data["answer"],
                referenceAnswer=test_data["referenceAnswer"],
                referenceAnswerStatements=test_data["referenceAnswerStatements"],
                force_generate=force_generate,
            )
        evaluate_result[provider] = result["score"]
    print("-" * 20, "最终评分", "-" * 20)
    print(evaluate_result)


if __name__ == "__main__":
    ollama_chat_test()
