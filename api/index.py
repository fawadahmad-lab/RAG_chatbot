from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# Initialize Flask
app = Flask(__name__, template_folder="../templates", static_folder="../static")

# Load documents and embeddings
loader = TextLoader("Cleaned_UIIT_Info_processed.txt")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
documents = text_splitter.split_documents(docs)
text_data = [doc.page_content for doc in documents]

# Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
model.encode(text_data, convert_to_numpy=True)  # warm-up
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(text_data, embedding_model)
retriever = vector_store.as_retriever()

# LLM setup
llm = ChatMistralAI(mistral_api_key=api_key)
prompt = ChatPromptTemplate.from_template("""
Answer strictly based on the context below in one line. If the question is not answerable from the context, say "I don't know." Ignore greetings/general questions.

<context>
{context}
</context>

Question: {input}
Answer: 
""")
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    try:
        result = retrieval_chain.invoke({"input": question})
        return jsonify({"response": result["answer"]})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

# Required for Vercel
def handler(environ, start_response):
    return app(environ, start_response)
