import json
from typing import Dict, List, Literal
from RAG.libs.chat import Chat
from RAG.libs.embedding import Embedding
from RAG.libs.prompts import Prompts
from RAG.libs.vector_store import VectorStore


class ContextRecallEvaluator:
    """
    上下文召回率评估器。

    实现步骤：

    1. 将参考答案拆分成多个关键信息；

    2. 逐个分析每个关键信息是否可归因于给定的上下文；

    3. 根据每个关键信息的得分，计算上下文召回率。
    """

    def __init__(
        self,
        chat_model=None,  # 评估大模型
    ):
        self.chat_model = chat_model

    def evaluate(
        self,
        evaluate_data: List[Dict],
        force_generate: bool = False,
    ):
        """
        上下文召回率评估。

        参数:

        evaluate_data (List[Dict]): 评估数据。

        force_generate（bool）: 是否强制根据参考答案拆分关键信息。

        返回:

        evaluate_result（{"score": score, "data": data}）: 评估结果。
        """
        retrievedContext = evaluate_data["retrievedContext"]
        referenceAnswer = evaluate_data["referenceAnswer"]
        if (
            force_generate == True
            or "referenceAnswerStatements" not in evaluate_data
            or not evaluate_data["referenceAnswerStatements"]
        ):
            input = Prompts.statement_split_prompt.format(context=referenceAnswer)
            result = self.chat_model.invoke(input=input)
            print("statement_split: ", result.content)
            referenceAnswerStatements = json.loads(result.content)
            evaluate_data["referenceAnswerStatements"] = referenceAnswerStatements
        datas = []
        for referenceAnswerStatement in evaluate_data["referenceAnswerStatements"]:
            input = Prompts.context_recall_evaluat_prompt.format(
                context=("\n").join(retrievedContext),
                statement=referenceAnswerStatement,
            )
            print("input：", input)
            evaluate_result = self.chat_model.invoke(input=input)
            print(evaluate_result.content)
            # 偶尔出现json格式不对，LLM 重新生成一次
            try:
                data = json.loads(evaluate_result.content)
            except:
                evaluate_result = self.chat_model.invoke(input=input)
                data = json.loads(evaluate_result.content)
            data["referenceAnswerStatement"] = referenceAnswerStatement
            datas.append(data)
        score = 0
        if len(datas) > 0:
            score = round(sum([float(item["score"]) for item in datas]) / len(datas), 2)
        result = {"score": score, "data": datas}
        print("ContextRecallEvaluator: ", result)
        return result


class ContextRelevanceEvaluator:
    """
    上下文相关性评估器。

    实现步骤：

    1. 逐个分析每个上下文片段是否与用户问题相关；

    2. 根据每个上下文片段的得分，计算上下文相关性。
    """

    def __init__(
        self,
        chat_model=None,  # 评估大模型
    ):
        self.chat_model = chat_model

    def evaluate(
        self,
        evaluate_data: List[Dict],
        force_generate: bool = False,
    ):
        """
        上下文相关性评估。

        参数:

        evaluate_data (List[Dict]): 评估数据。

        force_generate（bool）: --

        返回:

        evaluate_result（{"score": score, "data": data}）: 评估结果。
        """
        question = evaluate_data["question"]
        retrievedContext = evaluate_data["retrievedContext"]
        datas = []
        for context in retrievedContext:
            input = Prompts.context_relevance_evaluat_prompt.format(
                question=question, context=context
            )
            print("input：", input)
            evaluate_result = self.chat_model.invoke(input=input)
            print(evaluate_result.content)
            # 偶尔出现json格式不对，LLM 重新生成一次
            try:
                data = json.loads(evaluate_result.content)
            except:
                evaluate_result = self.chat_model.invoke(input=input)
                data = json.loads(evaluate_result.content)
            data["retrievedContext"] = context
            datas.append(data)
        score = 0
        if len(datas) > 0:
            score = round(sum([float(item["score"]) for item in datas]) / len(datas), 2)
        result = {"score": score, "data": datas}
        print("ContextRelevanceEvaluator: ", result)
        return result


class FaithfulnessEvaluator:
    """
    答案忠实度评估器。

    实现步骤：

    1. 将实际答案拆分成多个关键信息；

    2. 逐个分析每个关键信息是否可归因于给定的上下文；

    3. 根据每个关键信息的得分，计算答案忠实度。
    """

    def __init__(
        self,
        chat_model=None,  # 评估大模型
    ):
        self.chat_model = chat_model

    def evaluate(
        self,
        evaluate_data: List[Dict],
        force_generate: bool = False,
    ):
        """
        答案忠实度评估。

        参数:

        evaluate_data (List[Dict]): 评估数据。

        force_generate（bool）: 是否强制根据实际答案拆分关键信息。

        返回:

        evaluate_result（{"score": score, "data": data}）: 评估结果。
        """
        retrievedContext = evaluate_data["retrievedContext"]
        answer = evaluate_data["answer"]
        if (
            force_generate == True
            or "answerStatements" not in evaluate_data
            or not evaluate_data["answerStatements"]
        ):
            input = Prompts.statement_split_prompt.format(context=answer)
            result = self.chat_model.invoke(input=input)
            print("statement_split: ", result.content)
            answerStatements = json.loads(result.content)
            evaluate_data["answerStatements"] = answerStatements
        datas = []
        for answerStatement in evaluate_data["answerStatements"]:
            input = Prompts.faithfulness_evaluat_prompt.format(
                context=retrievedContext, statement=answerStatement
            )
            print("input：", input)
            evaluate_result = self.chat_model.invoke(input=input)
            # 偶尔出现json格式不对，LLM 重新生成一次
            try:
                data = json.loads(evaluate_result.content)
            except:
                evaluate_result = self.chat_model.invoke(input=input)
                data = json.loads(evaluate_result.content)
            data["answerStatement"] = answerStatement
            datas.append(data)
        score = 0
        if len(datas) > 0:
            score = round(sum([float(item["score"]) for item in datas]) / len(datas), 2)
        result = {"score": score, "data": datas}
        print("FaithfulnessEvaluator: ", result)
        return result


class AnswerRelevanceEvaluator:
    """
    答案相关性评估器。

    实现步骤：

    1. 根据实际答案推导出多个模拟问题；

    2. 逐个分析每个模拟问题是否与用户问题相似；

    3. 根据每个模拟问题的得分，计算答案相关性。
    """

    def __init__(
        self,
        chat_model=None,  # 评估大模型
    ):
        self.chat_model = chat_model

    def evaluate(
        self,
        evaluate_data: List[Dict],
        force_generate: bool = False,
    ):
        """
        答案相关性评估。

        参数:

        evaluate_data (List[Dict]): 评估数据。

        force_generate（bool）: 是否强制根据实际答案推导模拟问题。

        返回:

        evaluate_result（{"score": score, "data": data}）: 评估结果。
        """
        question = evaluate_data["question"]
        answer = evaluate_data["answer"]
        if (
            force_generate == True
            or "simulationQuestions" not in evaluate_data
            or not evaluate_data["simulationQuestions"]
        ):
            input = Prompts.simulation_question_generate_prompt.format(
                context=answer, num=3
            )
            result = self.chat_model.invoke(input=input)
            print("simulation_question: ", result)
            simulationQuestions = json.loads(result.content)
            evaluate_data["simulationQuestions"] = simulationQuestions
        datas = []
        for simulationQuestion in evaluate_data["simulationQuestions"]:
            input = Prompts.answer_relevance_evaluat_prompt.format(
                question=question, simulationQuestion=simulationQuestion
            )
            print("input：", input)
            evaluate_result = self.chat_model.invoke(input=input)
            print(evaluate_result.content)
            # 偶尔出现json格式不对，LLM 重新生成一次
            try:
                data = json.loads(evaluate_result.content)
            except:
                evaluate_result = self.chat_model.invoke(input=input)
                data = json.loads(evaluate_result.content)
            data["simulationQuestion"] = simulationQuestion
            datas.append(data)
        score = 0
        if len(datas) > 0:
            score = round(sum([float(item["score"]) for item in datas]) / len(datas), 2)
        result = {"score": score, "data": datas}
        print("AnswerRelevanceEvaluator: ", result)
        return result


class AnswerCorrectnessEvaluator:
    """
    答案正确性评估器。

    实现步骤：

    1. 将参考答案拆分成多个关键信息；

    2. 逐个分析每个关键信息是否可归因于给定的实际答案；

    3. 根据每个关键信息的得分，计算答案正确性。
    """

    def __init__(
        self,
        chat_model=None,  # 评估大模型
    ):
        self.chat_model = chat_model

    def evaluate(
        self,
        evaluate_data: List[Dict],
        force_generate: bool = False,
    ):
        """
        答案正确性评估评估。

        参数:

        evaluate_data (List[Dict]): 评估数据。

        force_generate（bool）: 是否强制根据参考答案拆分关键信息。

        返回:

        evaluate_result（{"score": score, "data": data}）: 评估结果。
        """
        answer = evaluate_data["answer"]
        referenceAnswer = evaluate_data["referenceAnswer"]
        if (
            force_generate == True
            or "referenceAnswerStatements" not in evaluate_data
            or not evaluate_data["referenceAnswerStatements"]
        ):
            input = Prompts.statement_split_prompt.format(context=referenceAnswer)
            result = self.chat_model.invoke(input=input)
            print("statement_split: ", result.content)
            referenceAnswerStatements = json.loads(result.content)
            evaluate_data["referenceAnswerStatements"] = referenceAnswerStatements
        datas = []
        for referenceAnswerStatement in evaluate_data["referenceAnswerStatements"]:
            input = Prompts.answer_correctness_evaluat_prompt.format(
                context=answer, statement=referenceAnswerStatement
            )
            print("input：", input)
            evaluate_result = self.chat_model.invoke(input=input)
            print(evaluate_result.content)
            # 偶尔出现json格式不对，LLM 重新生成一次
            try:
                data = json.loads(evaluate_result.content)
            except:
                evaluate_result = self.chat_model.invoke(input=input)
                data = json.loads(evaluate_result.content)
            data["referenceAnswerStatement"] = referenceAnswerStatement
            datas.append(data)
        score = 0
        if len(datas) > 0:
            score = round(sum([float(item["score"]) for item in datas]) / len(datas), 2)
        result = {"score": score, "data": datas}
        print("AnswerCorrectnessEvaluator: ", result)
        return result


def Evaluator(
    chat_model=None,
    provider: Literal[
        "contextRecall",
        "contextRelevance",
        "faithfulness",
        "answerRelevance",
        "answerCorrectness",
    ] = "contextRecall",
    **kwargs,
):
    if provider == "contextRecall":
        return ContextRecallEvaluator(chat_model, **kwargs)
    elif provider == "contextRelevance":
        return ContextRelevanceEvaluator(chat_model, **kwargs)
    elif provider == "faithfulness":
        return FaithfulnessEvaluator(chat_model, **kwargs)
    elif provider == "answerRelevance":
        return AnswerRelevanceEvaluator(chat_model, **kwargs)
    elif provider == "answerCorrectness":
        return AnswerCorrectnessEvaluator(chat_model, **kwargs)
    else:
        raise ValueError(f"Provider {provider} not supported.")


def BatchEvaluator(
    chat_model,
    vector_store,
    qa_data,
    top_k=3,
    filter=None,
    force_generate=False,
    output_path=None,
    providers=[
        "contextRecall",
        "contextRelevance",
        "faithfulness",
        "answerRelevance",
        "answerCorrectness",
    ],
):
    evaluate_data = []
    evaluate_result = {}
    for test_data in qa_data:
        question = test_data["question"]

        # 向量检索
        search_result = vector_store.similarity_search_with_score(
            query=question, k=top_k, filter=filter
        )

        # 根据检索结果和问题构造提示词
        test_data["retrievedContext"] = []
        for result in search_result:
            test_data["retrievedContext"].append(result[0].page_content)
        input = Prompts.rag_answer_prompt.format(
            context="\n".join(test_data["retrievedContext"]), question=question
        )
        print(search_result[0][0].metadata["type"])
        print("input：", input)

        # llm 回答
        chat_result = chat_model.invoke(input=input)
        test_data["answer"] = chat_result.content
        print("answer：", test_data["answer"])

        # 评估
        for provider in providers:
            evaluator = Evaluator(chat_model=chat_model, provider=provider)
            result = 0
            if provider == "contextRecall":
                result = evaluator.evaluate(
                    evaluate_data=test_data,
                    force_generate=force_generate,
                )
            elif provider == "contextRelevance":
                result = evaluator.evaluate(
                    evaluate_data=test_data,
                    force_generate=force_generate,
                )
            elif provider == "faithfulness":
                result = evaluator.evaluate(
                    evaluate_data=test_data,
                    force_generate=force_generate,
                )
            elif provider == "answerRelevance":
                result = evaluator.evaluate(
                    evaluate_data=test_data,
                    force_generate=force_generate,
                )
            elif provider == "answerCorrectness":
                result = evaluator.evaluate(
                    evaluate_data=test_data,
                    force_generate=force_generate,
                )
            test_data[provider] = result
            if provider not in evaluate_result:
                evaluate_result[provider] = []
            evaluate_result[provider].append(result["score"])

        evaluate_data.append(test_data)
        # 保存评估数据
        if output_path != None:
            json_data = json.dumps(evaluate_data, indent=2, ensure_ascii=False)
            with open(
                output_path,
                "w",
                encoding="utf-8",
            ) as f:
                f.write(json_data)
    score = {
        key: round(sum(values) / len(values), 2)
        for key, values in evaluate_result.items()
    }
    return {"score": score, "data": evaluate_data}
