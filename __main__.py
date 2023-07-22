import os
from langchain.vectorstores import Pinecone
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone


def main(args):
    index_name = args.get("index_name", "canada-tax")
    query = args.get("question", "what kinds of taxes exist in canada?")

    embedding = OpenAIEmbeddings(openai_api_key=os.getenv["OPENAI_API_KEY"])

    pinecone.init(
        api_key=os.getenv["PINECONE_API_KEY"],
        environment=os.getenv["PINECONE_ENV"],
    )

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=Pinecone.from_existing_index(index_name, embedding).as_retriever(),
        chain_type_kwargs={"verbose": True},
    )

    answer = qa.run(query)

    return {"question": query, "answer": answer}
