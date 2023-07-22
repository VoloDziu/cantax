import os
from langchain.vectorstores import Pinecone
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone


def qa(query: str, index_name: str):
    embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENV"],
    )

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=Pinecone.from_existing_index(index_name, embedding).as_retriever(),
        # chain_type_kwargs={"verbose": True},
    )

    return qa.run(query)


if __name__ == "__main__":
    answer = search(
        "how many tax brackets are there in canada?",
        "canada-tax",
    )
    print(">>>", answer)
