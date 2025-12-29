README.md
markdownDownloadCopy code# 🤖 Agentic RAG Assistant (LangGraph + LangChain + Streamlit + FAISS)

A Streamlit app that implements an **agentic RAG** workflow: the LLM decides **when** to retrieve, **grades** retrieved context for relevance, **rewrites** the query if retrieval fails, and then **generates** an answer grounded in the retrieved sources.

---

## 🧭 Table of Contents (Overview)

1.  🤖 [Overview](#overview)  
2.  ✨ [Features](#features)  
3.  🧰 [Tech Stack](#tech-stack)  
4.  🧠 [Architecture](#architecture)  
    -  🔁 [LangGraph Flow](#langgraph-flow)  
    -  📚 [Knowledge Base & Retrieval Tools](#knowledge-base--retrieval-tools)  
5.  🚀 [Getting Started](#getting-started)  
    -  ✅ [Prerequisites](#prerequisites)  
    -  📦 [Installation](#installation)  
    -  ▶️ [Run the App](#run-the-app)  
6.  ⚙️ [Configuration (UI)](#configuration-ui)  
7.  🔎 [How It Works (Step‑by‑step)](#how-it-works-step-by-step)  
8.  🛠️ [Customization](#customization)  
9.  🧯 [Troubleshooting](#troubleshooting)  
10. ⚠️ [Known Limitations](#known-limitations)  
11. 🔐 [Security Notes](#security-notes)  
12. 🗺️ [Roadmap Ideas](#roadmap-ideas)  
13. 🙏 [Acknowledgements / Sources](#acknowledgements--sources)  
14. 📄 [License](#license)  
15. 📁 [Project Structure](#project-structure)  

---

<a id="overview"></a>
## 🤖 Overview

This project is a single‑page Streamlit application called **Agentic RAG Assistant**. It lets users ask questions about:

- 🧩 **AI Agents** (ReAct prompting, tool use, agent loops)  
- 📏 **RAGAS** (RAG evaluation framework and metrics)  

Instead of always retrieving documents, the assistant behaves more like an **agent**:

- ✅ It can **answer directly** when retrieval isn’t needed.  
- 🔧 Or it can **invoke retrieval tools**, **check relevance**, and **retry** with a rewritten query if the retrieved content doesn’t match the question.

---

<a id="features"></a>
## ✨ Features

- 🧠 **Multi‑provider UI** (OpenAI / Groq + placeholders for OpenRouter / Gemini)  
- 🎛️ **Model selection** from the sidebar  
- 🔑 **API key input** in the sidebar (password field)  
- 🧾 **Max response tokens** control  
- 🧪 **Agent step tracing** (“Show Agent Steps”) to inspect node execution and outputs  
- 🧰 **Two retriever tools** over two different web‑based knowledge sets  
- ✅ **Relevance grading** step to detect bad retrieval results  
- ♻️ **Query rewriting loop** when retrieval is not relevant  
- 🗄️ **FAISS vector stores** built on‑the‑fly and cached with `st.cache_resource`  
- 💡 **Suggested question buttons** for quick testing  
- 🧹 **Clear chat** button  
- 🔁 **Rebuild Pipeline** button to clear cache and rebuild the knowledge base  

---

<a id="tech-stack"></a>
## 🧰 Tech Stack

- 🖥️ **UI**: Streamlit  
- 🧭 **Agent Orchestration**: LangGraph  
- 🤝 **LLM Integration**: LangChain (`ChatOpenAI`, `ChatGroq`)  
- 🧠 **Retrieval / Vector DB**: FAISS (in‑memory)  
- 🧬 **Embeddings**: HuggingFace embeddings (local path if available, fallback otherwise)  
- 🌐 **Data Loading**: `WebBaseLoader` (loads web pages as documents)  

---

<a id="architecture"></a>
## 🧠 Architecture

<a id="langgraph-flow"></a>
### 🔁 LangGraph Flow

```mermaid
flowchart TD
    A[START] --> B[agent (LLM w/ tools)]
    B -->|tool call| C[retrieve (ToolNode)]
    B -->|no tool call| Z[END]

    C --> D{grade_documents}
    D -->|relevant| E[generate (RAG answer)]
    D -->|not relevant| F[rewrite (improve query)]
    F --> B

    E --> Z[END]
Key idea: the LLM is “tool‑aware.” If it chooses to call a retriever tool, the graph routes into retrieval. Otherwise, it can respond immediately and exit.

📚 Knowledge Base & Retrieval Tools
On startup, the app downloads and embeds content from four URLs, then builds two FAISS indexes:
🧩 Agents sources

* https://lilianweng.github.io/posts/2023-06-23-agent/
* https://lilianweng.github.io/posts/2024-04-12-diffusion-video/

📏 RAGAS sources

* https://www.analyticsvidhya.com/blog/2024/05/a-beginners-guide-to-evaluating-rag-pipelines-using-ragas/
* https://docs.ragas.io/en/stable/#why-ragas

Each FAISS index is exposed to the agent as a LangChain retriever tool:

* 🔎 Agentic_blog: “Search and run information about agents”
* 🔎 Ragas_blog: “Search and run information about RAGAS (RAG Assessment) framework”

Retrieval configuration

* ✂️ Splitter: RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
* 🎯 Retriever: MMR search with k=5 and lambda_mult=0.5



🚀 Getting Started

✅ Prerequisites

* 
🐍 Python 3.10+ recommended

* 
🌍 Internet access (documents are loaded from the web at runtime)

* 
🔑 An API key for at least one supported provider:

✅ OpenAI (implemented)
✅ Groq (implemented)




💡 Note: OpenRouter/Gemini appear in the UI, but the provided code currently falls back to ChatOpenAI for “other” providers (see Known Limitations).


📦 Installation
Create and activate a virtual environment, then install dependencies:
bashDownloadCopy codepython -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
Example requirements.txt (adjust versions as needed):
txtDownloadCopy codestreamlit
langchain
langchain-openai
langchain-groq
langchain-community
langchain-text-splitters
langchain-huggingface
langgraph
faiss-cpu
sentence-transformers
pydantic
beautifulsoup4
requests

▶️ Run the App
bashDownloadCopy codestreamlit run app.py


⚙️ Configuration (UI)
In the Streamlit sidebar:
OptionDescriptionAI ProviderOpenAI / Groq / OpenRouter / GeminiModelProvider‑specific model listAPI KeyEntered securely via password inputMax Response TokensCaps generation lengthShow Agent StepsReveals node‑by‑node execution output + JSON payloadsRebuild PipelineClears st.cache_resource, rebuilds:• embeddings• FAISS indexes• retriever tools


🔎 How It Works (Step‑by‑step)

1. 
📚 Build vector stores (cached)

Tries to load a local HuggingFace embedding model from a hardcoded path.
If not found, falls back to: sentence-transformers/all-MiniLM-L6-v2.
Loads the 4 web pages, splits them into chunks, embeds them, and creates 2 FAISS indexes.
Wraps each retriever as a tool the agent can call.


2. 
❓ User asks a question

Query comes from the input box or suggested question buttons.
The app instantiates the selected LLM (ChatOpenAI or ChatGroq) with:
temperature=0
max_tokens from the sidebar


3. 
🤖 Agent node (LLM + tools)
The LLM is bound to retriever tools (bind_tools). It either:

✅ Responds directly (no tool calls) → graph ends, OR
🔧 Requests retrieval (tool call) → graph routes to retrieve.


4. 
🧰 Retrieve node
Executes the chosen retriever tool and returns documents as ToolMessage outputs.

5. 
✅ Grade documents
A structured‑output check returns a strict "yes" / "no" relevance signal.

If any retrieved chunk is graded relevant → proceed to generate.
Otherwise → proceed to rewrite.


6. 
♻️ Rewrite
The model rewrites the query to better match the knowledge base.
Special handling: If the query mentions “Ragas” and retrieval failed, it clarifies it means the AI evaluation framework.
Loops back to the agent node to try again.

7. 
✍️ Generate
A RAG prompt combines user question + retrieved context.
The model produces a grounded answer (or says “I don’t know”).

8. 
💬 UI rendering
Conversation history is shown with st.chat_message.
✅ Optional: Node execution logs appear in the “Agent Reasoning Steps” expander.




🛠️ Customization

1. 🌐 Add / change knowledge sources
Edit the URL lists in build_vectorstore():


* urls_1 → agent‑related sources
* urls_2 → RAGAS‑related sources


1. 🎚️ Tune chunking & retrieval (in build_vectorstore())


* chunk_size / chunk_overlap → controls context granularity
* Retriever config:

k: number of chunks retrieved
lambda_mult: MMR diversity vs. similarity balance




1. 
🧬 Swap embedding model
Replace model_name= with any HuggingFace embedding model compatible with HuggingFaceEmbeddings.

2. 
💾 Persist FAISS indexes (recommended improvement)
Currently, FAISS is built in‑memory (cached per Streamlit session). Extend the project to save/load FAISS to disk for faster startups.

3. 
🔌 Implement OpenRouter / Gemini properly
The UI lists them, but the code uses a generic fallback for “other” providers. Add the correct LangChain integrations + environment variables to fully support them.




🧯 Troubleshooting
IssueSolution❌ “Please enter your API Key…”Enter a valid provider key and rerun.⚠️ Embedding model load fails1. The hardcoded local path may not exist on your machine.2. Ensure sentence-transformers is installed.3. Verify internet access (fallback downloads the model).🌐 Web pages fail to load / empty docs• Source sites may block scraping or change HTML.• Your network may block requests.Fix: Replace URLs or use a different loader.🧩 Nothing appears in final answer✅ Enable Show Agent Steps to inspect which node executed and what it returned.🔁 Rebuild doesn’t change anythingClick Rebuild Pipeline to clear st.cache_resource and rebuild the knowledge base.


⚠️ Known Limitations

1. 
💬 No true conversation memory in the graph
The UI shows chat history, but each run sends only the latest query into the graph:
HumanMessage(content=final_query).

2. 
🧪 OpenRouter/Gemini are not fully implemented
They appear in the UI, but the “else” branch uses ChatOpenAI as a generic fallback.

3. 
💾 No persistent vector database
FAISS is created in‑memory (cached by Streamlit). Restarting the app rebuilds embeddings unless persistence is added.

4. 
🪟 Hardcoded local embedding path
The Windows‑specific path likely won’t exist on other machines (fallback handles this, but consider replacing it).




🔐 Security Notes

1. 
🔑 API Key Handling
Your key is entered in the sidebar and assigned to environment variables at runtime (os.environ[...]).
Best Practices:

✅ Never commit secrets to Git!
✅ Use Streamlit secrets or deployment‑level environment variables.
✅ Add .env to .gitignore (see Project Structure).


2. 
🌍 External Web Requests
The app fetches external web pages at runtime. If deploying publicly, consider:

Rate limiting
Caching / persistence
Allowlists for trusted domains





🗺️ Roadmap Ideas

* 🔌 Add real multi‑provider support (OpenRouter, Gemini) with proper SDKs
* 💾 Persist FAISS indexes; rebuild only when sources change
* 🧠 Add conversation memory (feed prior messages into the graph)
* 🧾 Add citations (source URL + chunk snippet) in the final answer
* 📏 Integrate RAGAS evaluation directly (very on‑theme!)



🙏 Acknowledgements / Sources
This demo knowledge base is built from:
Lilian Weng

* https://lilianweng.github.io/posts/2023-06-23-agent/
* https://lilianweng.github.io/posts/2024-04-12-diffusion-video/

RAGAS references

* https://docs.ragas.io/en/stable/#why-ragas
* https://www.analyticsvidhya.com/blog/2024/05/a-beginners-guide-to-evaluating-rag-pipelines-using-ragas/



📄 License
Add your preferred license (e.g., MIT) in a LICENSE file.


📁 Project Structure
Below is a clean, scalable structure for this project (recommended for organizing the provided code):
agentic-rag-project/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (gitignored)
├── .gitignore                # Git ignore rules
│
├── utils/                    # Utility functions
│   ├── vector_store.py       # FAISS & embedding management
│   └── agent_graph.py        # LangGraph workflow definition
│
├── config/                   # Configuration files
│   └── prompts.py            # Prompt templates
│
└── README.md                 # This documentation

✅ Suggested Module Responsibilities
utils/vector_store.py

* build_vectorstore()
* Document loading (URLs)
* Splitting + embedding + FAISS creation
* Retriever tool creation

utils/agent_graph.py

* AgentState definition
* initialize_graph(llm_model, tools_list)
* Nodes: agent, retrieve, grade_documents, rewrite, generate

config/prompts.py

* Grading prompt template
* RAG generation prompt template
* Rewrite prompt template


📄 .gitignore (minimal)
gitignoreDownloadCopy code.env
.venv/
__pycache__/
*.pyc
.streamlit/
