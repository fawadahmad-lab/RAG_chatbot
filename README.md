# RAG Chatbot with FastAPI and LangChain

A Retrieval-Augmented Generation (RAG) chatbot web app built using FastAPI and LangChain.
It loads documents, creates vector embeddings, and uses a large language model (LLM) to answer user questions based on contextual document retrieval.


## Features

* **Document loading and chunking** for effective retrieval
* **Vector search** using FAISS and Nomic embeddings
* **LLM integration** with Groq's ChatGroq model
* Interactive **chat interface** styled with ChatGPT-like UI
* Asynchronous REST API built with **FastAPI**
* Templating with **Jinja2** and static assets support


## Installation

1. Clone the repository

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. Create and activate a Python virtual environment

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

4. Set environment variables
   Create a `.env` file or export these variables:

   ```
   GROQ_API_KEY=your_groq_api_key
   NOMIC_API_KEY=your_nomic_api_key
   ```

5. Prepare your documents
   Place your document file at the path specified in the code or update accordingly.

## Running the App

Start the FastAPI server with:

```bash
uvicorn main:app --reload
```

Open your browser and go to:
`http://127.0.0.1:8000`

---

## Project Structure

```
├── main.py            # FastAPI app and LangChain setup
├── requirements.txt   # Python dependencies
├── static/            # CSS and JS files
├── templates/         # Jinja2 HTML templates
└── data/              # Your document files
└── requirements.txt   # requirements
```

## Usage

* Ask questions in the chat box.
* The chatbot uses the RAG approach: it retrieves relevant document snippets and generates a concise answer based on them.
* Responses are limited to **one short sentence** for clarity.
## Notes

* Be sure to trust the document sources when loading vector stores (FAISS) due to deserialization security warnings.
* Adjust chunk size and overlap in `RecursiveCharacterTextSplitter` for best performance.
* Currently uses Groq's LLM and Nomic embeddings, you can switch models as needed.


## License

MIT License © Fawad Ahmad
