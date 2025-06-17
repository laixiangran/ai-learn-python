from RAG.libs.embedding import Embedding


def ollama_embendding_test():
    embedding = Embedding(model="nomic-embed-text", provider="ollama")
    res1 = embedding.embed_query(text="什么是少儿编程教育？")
    print(res1)
    res2 = embedding.embed_documents(
        texts=["什么是少儿编程教育？", "国家政策对少儿编程教育的影响体现在哪些方面？"]
    )
    print(res2)


if __name__ == "__main__":
    ollama_embendding_test()
