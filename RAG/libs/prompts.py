class Prompts:
    # 提供上下文后的回答问题提示模板
    rag_answer_prompt = """
你是一位负责回答问题的专家，你的任务是根据提供的上下文来回答问题。
如果你无法从提供的上下文中直接得出答案，你只需要回复 “我无法根据现有信息回答这个问题”，不要输出其他无关内容。

问题：
{question}

上下文：
{context}

回答：
    """

    # 根据上下文生成题目、答案和依据的提示模板
    question_generate_prompt = """
你是一位负责出题的专家，你的任务是根据提供的上下文生成{num}个题目，并生成答案和依据。

说明：
1. 题目要与提供的上下文相关，不要出现类似“这个题目的答案在哪一章”这样的题目；
2. 依据必须与提供的上下文的内容保持一致，不要进行缩写、扩写、改写、摘要、替换词语等；
3. 答案请保持完整且简洁，无须重复题目。答案要能够独立回答题目，而不是引用现有的章节、页码等；
4. 如果提供的上下文主要是目录，或者是一些人名、地址、电子邮箱等没有办法生成有意义的题目时，可以返回[]；
5. 严格按JSON对象格式返回，对象包含字段：question（题目）、answer（答案）、reason（依据），不要输出```json```，也不要输出其他无关内容。

上下文：
{context}

回答：
    """

    # 审核题目和答案并打分的提示模板
    question_evaluat_prompt = """
你是一位负责审核题目的专家，你的任务是根据题目和答案来进行打分，并给出依据。

说明：
1. 好的题目，应该是询问事实、观点等，而不是类似于“这一段描述了什么”；
2. 好的答案，应该能够直接回答题目，而不是给出在原文中的引用，例如“在第3页中”等；
3. 结合题目和答案进行打分，并给出打分依据，分值是一个整数，取值范围为1-5；
4. 严格按JSON对象格式返回，对象包含字段：score（得分）、reason（依据），不要输出```json```，也不要输出其他无关内容。


题目：
{question}

答案：
{answer}

回答：
    """

    # 将原始问题改写成多个语义相同但表达方式不同的新问题的提示模板
    synonymy_rewritten_prompt = """
你是一位语言方面的专家，你的任务是将提供的原始问题改写成{num}个语义相同但表达方式不同的新问题。

说明：
1. 严格按以下JSON数组格式返回：["问题1", "问题2", ...]，不要输出```json```，也不要输出其他无关内容。

原始问题：
{question}

回答：
    """

    # 将原始问题分解成多个不同视角的子问题的提示模板
    sub_rewritten_prompt = """
你是一位语言方面的专家，你的任务是将提供的原始问题分解成{num}个不同视角的子问题。

说明：
1. 严格按以下JSON数组格式返回：["问题1", "问题2", ...]，不要输出```json```，也不要输出其他无关内容。

原始问题：
{question}

回答：
    """

    # 根据原始问题生成相关背景信息的提示模板
    context_supplement_prompt = """
你是一位语言方面的专家，你的任务是根据提供的原始问题生成一段与原始问题相关的背景信息。

说明：
1. 背景信息最大不超过{maxLen}个字符；
2. 只能输出背景信息，不要输出其他无关内容。

原始问题：
{question}

回答：
    """

    # 从上下文中提取所有关键信息的提示模板
    statement_split_prompt = """
你是一位语言方面的专家，你的任务是提取出提供的上下文中所有的关键信息。

说明：
1. 严格按以下JSON数组格式返回：["关键信息1", "关键信息2", ...]，不要输出```json```，也不要输出其他无关内容。

上下文：
${context}

回答：
    """

    # 根据上下文生成用户可能问的问题的提示模板
    simulation_question_generate_prompt = """
你是一位语言方面的专家，你的任务是根据提供的上下文来生成{num}个用户可能问的问题。

说明：
1. 严格按以下JSON数组格式返回：["问题1", "问题2", ...]，不要输出```json```，也不要输出其他无关内容。

上下文：
{context}

回答：
    """

    # 分析关键信息是否可归因于上下文并给出得分的提示模板
    context_recall_evaluat_prompt = """
你是一位语言方面的专家，你的任务是分析提供的关键信息是否可归因于提供的上下文并给出得分。

说明：
1. 如果关键信息不能归因于上下文，则得分为0；
2. 如果关键信息能够归因于上下文，则得分为1；
3. 严格按JSON对象格式返回，对象包含字段：score（得分），不要输出```json```，也不要输出其他无关内容。

上下文：
{context}

关键信息：
{statement}

回答：
    """

    # 分析上下文是否与问题有关并给出得分的提示模板
    context_relevance_evaluat_prompt = """
你是一位语言方面的专家，你的任务是分析提供的上下文是否与提供的问题有关并给出得分。

说明：
1. 如果上下文与问题无关，则得分为0；
2. 如果上下文与问题有关，则得分为1；
3. 严格按JSON对象格式返回，对象包含字段：score（得分），不要输出```json```，也不要输出其他无关内容。

问题：
{question}

上下文：
{context}

回答：
    """

    # 分析关键信息是否可归因于上下文并给出得分的提示模板
    faithfulness_evaluat_prompt = """
你是一位语言方面的专家，你的任务是分析提供的关键信息是否可归因于提供的上下文并给出得分。

说明：
1. 如果关键信息不能归因于上下文，则得分为0；
2. 如果关键信息能够归因于上下文，则得分为1；
3. 严格按JSON对象格式返回，对象包含字段：score（得分），不要输出```json```，也不要输出其他无关内容。

上下文：
{context}

关键信息：
{statement}

回答：
    """

    # 分析模拟问题和实际问题是否相似并给出得分的提示模板
    answer_relevance_evaluat_prompt = """
你是一位语言方面的专家，你的任务是分析模拟问题和实际问题是否相似并给出得分。

说明：
1. 如果模拟问题与实际问题不相似，则得分为0；
2. 如果模拟问题与实际问题相似，则得分为1；
3. 严格按JSON对象格式返回，对象包含字段：score（得分），不要输出```json```，也不要输出其他无关内容。

实际问题：
{question}

模拟问题：
{simulationQuestion}

回答：
    """

    # 分析关键信息是否可归因于上下文并给出得分的提示模板
    answer_correctness_evaluat_prompt = """
你是一位语言方面的专家，你的任务是分析提供的关键信息是否可归因于提供的上下文并给出得分。

说明：
1. 如果关键信息不能归因于上下文，则得分为0；
2. 如果关键信息能够归因于上下文，则得分为1；
3. 严格按JSON对象格式返回，对象包含字段：score（得分），不要输出```json```，也不要输出其他无关内容。

关键信息：
{statement}

上下文：
{context}

回答：
    """
