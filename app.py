import os
from langchain.vectorstores import Pinecone
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from flask import Flask, request, abort, jsonify
from qa import qa


app = Flask(__name__)


@app.route("/qa")
def qa_route():
    query = request.args["q"]
    index = request.args.get("index")

    if not index:
        index = "canada-tax"

    try:
        answer = qa(query=query, index=index)

        return jsonify({"q": query, "a": answer})
    except:
        abort(500)


@app.route("/500")
def err500():
    abort(500)
