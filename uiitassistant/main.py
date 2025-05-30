from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain_nomic import NomicEmbeddings

from dotenv import load_dotenv
import os
import getpass

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not os.getenv("NOMIC_API_KEY"):
    os.environ["NOMIC_API_KEY"] = getpass.getpass("nk-L8dDKRHKz1WaLrR3khfSOPrOcWTvhsZOq8JTZZAdRRE")

app = FastAPI()

# ✅ Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ✅ Load documents and create retriever
loader = TextLoader("Cleaned_UIIT_Info_processed.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
split_docs = splitter.split_documents(docs)
text_data = [doc.page_content for doc in split_docs]

embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5")
vector_store_path = "faiss_index"
vector_store = FAISS.from_texts(text_data, embedding_model)
vector_store.save_local(vector_store_path)
vector_store = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever()


llm = ChatGroq(api_key=groq_api_key, model_name="llama3-70b-8192")
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below in **one short sentence**. If not answerable, say "I'am sorry".

<context>
{context}
</context>

Question: {input}
Answer:
""")
doc_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, doc_chain)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "")
    try:
        result = retrieval_chain.invoke({"input": question})
        return JSONResponse({"response": result["answer"]})
    except Exception as e:
        return JSONResponse({"response": f"Error: {str(e)}"})
