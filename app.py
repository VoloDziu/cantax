import os
from langchain.vectorstores import Pinecone
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from flask import Flask, request, abort, jsonify
from flask_api import status
from qa import qa


app = Flask(__name__)


@app.route("/qa", methods=["POST"])
def qa_route():
    body = request.get_json(force=True, silent=True)

    print(body)

    if not body:
        return (
            jsonify({"error": "Please provide request body with a 'question' prop"}),
            status.HTTP_400_BAD_REQUEST,
        )

    if "question" not in body:
        return (
            jsonify({"error": "Request body must contain a non-empty 'question' prop"}),
            status.HTTP_400_BAD_REQUEST,
        )

    query = body["question"]

    index = "canada-tax"
    if "index" in body:
        index = body["index"]

    try:
        answer = qa(query=query, index_name=index)

        return jsonify({"q": query, "a": answer}), 200
    except Exception as e:
        return (
            jsonify({"error": "unknown"}),
            status.HTTP_500_BAD_REQUEST,
        )


@app.route("/500")
def err500():
    return (
        jsonify({"error": "force fail"}),
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    app.run()
