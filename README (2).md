<think>
**Cleaning up README content**

The user wants to simplify their README by removing excessive symbols like 🎚️, keeping the content basic. They provided a mixed format that needs to be adjusted into markdown format for easy copying. I should retain a few emojis in headings but eliminate most within bullet points to enhance readability. 

Additionally, I need to address formatting issues, such as ensuring the mermaid code block has proper closures and moving "Key idea: ..." outside of it. I'll prepare a clean README.md file reflecting these changes.

**Polishing Markdown formatting**

Okay, the user has pointed out several issues with their Markdown, such as missing headings like "Getting Started" that need to be formatted properly. There are also bullet lists that are not correctly formatted, especially items like "✅ OpenAI" appearing on separate lines.

I’ll create a polished version of the Markdown. They want internal links, and while we have anchor links, we can use GitHub's automatic anchors with `<a id=...>`. For emojis, I’ll keep them limited to title and section headings and remove excess from bullet points, while possibly retaining a few in the Table of Contents.

**Finalizing README content**

The user indicated they want to simplify their README by keeping some basic icons, like 🤖 for the title and a few small emojis for section headings. For the Table of Contents, I'll keep it plain without emojis. 

The sections might have icons like 🤖 Overview, ✨ Features, and others. I’ll remove emojis from bullet points and lists to cut down on excess, including the 🎚️ from customization point 2. I’ll also fix any formatting issues, ensure code blocks have correct language tags, and incorporate suggestions like the `.gitignore` file. Let's craft the final README!

**Finalizing README details**

I need to add disclaimers about provider support and mention that we have a `.env` file for environment variables, even though the code uses sidebar input. While the user didn't request changes beyond emojis, I’ll ensure the content remains intact and formatted correctly.

I'll create the final Markdown version as a code block. We should keep the anchors for headings and use GitHub's auto anchors, but I must consider that emojis can cause mismatches. I can keep the user's existing HTML anchors for the TOC to ensure everything works properly.

**Crafting the README structure**

Alright, I'm organizing the README with the title and a short description. For the Table of Contents, I’ll use a numbered list in plain text while linking to anchors but consider removing emojis from TOC items to keep things clean. I can keep emojis in section headings, aligning with the user's desire for minimal clutter.

I'll structure the sections with headings using the user-defined anchors. Within sections, I must remove emojis from bullet points while keeping it clear with tech stack, architecture diagrams, and lists as specified. 

Finally, I’ll include a .gitignore suggestion in a code block and ensure the entire README content stays concise, adhering to the markdown format the user requested. Let’s do that!
</think>```md
# 🤖 Agentic RAG Assistant (LangGraph + LangChain + Streamlit + FAISS)

A Streamlit app that implements an **agentic RAG** workflow: the LLM decides **when** to retrieve, **grades** retrieved context for relevance, **rewrites** the query if retrieval fails, and then **generates** an answer grounded in the retrieved sources.

---

## Table of Contents (Overview)

1. [Overview](#overview)  
2. [Features](#features)  
3. [Tech Stack](#tech-stack)  
4. [Architecture](#architecture)  
   - [LangGraph Flow](#langgraph-flow)  
   - [Knowledge Base & Retrieval Tools](#knowledge-base--retrieval-tools)  
5. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Installation](#installation)  
   - [Run the App](#run-the-app)  
6. [Configuration (UI)](#configuration-ui)  
7. [How It Works (Step-by-step)](#how-it-works-step-by-step)  
8. [Customization](#customization)  
9. [Troubleshooting](#troubleshooting)  
10. [Known Limitations](#known-limitations)  
11. [Security Notes](#security-notes)  
12. [Roadmap Ideas](#roadmap-ideas)  
13. [Acknowledgements / Sources](#acknowledgements--sources)  
14. [License](#license)  
15. [Project Structure](#project-structure)  

---

<a id="overview"></a>
## Overview

This project is a single-page Streamlit application called **Agentic RAG Assistant**. It lets users ask questions about:

- **AI Agents** (ReAct prompting, tool use, agent loops)
- **RAGAS** (RAG evaluation framework and metrics)

Instead of always retrieving documents, the assistant behaves more like an **agent**:

- It can **answer directly** when retrieval isn’t needed.
- Or it can **invoke retrieval tools**, **check relevance**, and **retry** with a rewritten query if the retrieved content doesn’t match the question.

---

<a id="features"></a>
## Features

- Multi-provider UI (OpenAI / Groq + placeholders for OpenRouter / Gemini)
- Model selection from the sidebar
- API key input in the sidebar (password field)
- Max response tokens control
- Agent step tracing (“Show Agent Steps”) to inspect node execution and outputs
- Two retriever tools over two different web-based knowledge sets
- Relevance grading step to detect bad retrieval results
- Query rewriting loop when retrieval is not relevant
- FAISS vector stores built on-the-fly and cached with `st.cache_resource`
- Suggested question buttons for quick testing
- Clear chat button
- Rebuild Pipeline button to clear cache and rebuild the knowledge base

---

<a id="tech-stack"></a>
## Tech Stack

- UI: Streamlit  
- Agent Orchestration: LangGraph  
- LLM Integration: LangChain (`ChatOpenAI`, `ChatGroq`)  
- Retrieval / Vector DB: FAISS (in-memory)  
- Embeddings: HuggingFace embeddings (local path if available, fallback otherwise)  
- Data Loading: `WebBaseLoader` (loads web pages as documents)

---

<a id="architecture"></a>
## Architecture

<a id="langgraph-flow"></a>
### LangGraph Flow

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
```

**Key idea:** the LLM is tool-aware. If it chooses to call a retriever tool, the graph routes into retrieval. Otherwise, it can respond immediately and exit.

<a id="knowledge-base--retrieval-tools"></a>
### Knowledge Base & Retrieval Tools

On startup, the app downloads and embeds content from **four URLs**, then builds **two FAISS indexes**:

**Agents sources**
- https://lilianweng.github.io/posts/2023-06-23-agent/
- https://lilianweng.github.io/posts/2024-04-12-diffusion-video/

**RAGAS sources**
- https://www.analyticsvidhya.com/blog/2024/05/a-beginners-guide-to-evaluating-rag-pipelines-using-ragas/
- https://docs.ragas.io/en/stable/#why-ragas

Each FAISS index is exposed to the agent as a LangChain retriever tool:

- `Agentic_blog`: “Search and run information about agents”
- `Ragas_blog`: “Search and run information about RAGAS (RAG Assessment) framework”

Retrieval configuration:

- Splitter: `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)`
- Retriever: MMR search with `k=5` and `lambda_mult=0.5`

---

<a id="getting-started"></a>
## Getting Started

<a id="prerequisites"></a>
### Prerequisites

- Python 3.10+ recommended
- Internet access (documents are loaded from the web at runtime)
- An API key for at least one supported provider:
  - OpenAI (implemented)
  - Groq (implemented)

> Note: OpenRouter/Gemini appear in the UI, but the provided code currently falls back to `ChatOpenAI` for “other” providers (see [Known Limitations](#known-limitations)).

<a id="installation"></a>
### Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

Example `requirements.txt` (adjust versions as needed):

```txt
streamlit
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
```

<a id="run-the-app"></a>
### Run the App

```bash
streamlit run app.py
```

---

<a id="configuration-ui"></a>
## Configuration (UI)

In the Streamlit sidebar:

- AI Provider: OpenAI / Groq / OpenRouter / Gemini
- Model: provider-specific model list
- API Key: entered securely via password input
- Max Response Tokens: caps generation length
- Show Agent Steps: reveals node-by-node execution output + JSON payloads
- Rebuild Pipeline: clears Streamlit cache (`st.cache_resource`) and rebuilds:
  - embeddings
  - FAISS indexes
  - retriever tools

---

<a id="how-it-works-step-by-step"></a>
## How It Works (Step-by-step)

1. **Build vector stores (cached)**
   - Tries to load a local HuggingFace embedding model from a hardcoded path.
   - If not found, falls back to: `sentence-transformers/all-MiniLM-L6-v2`
   - Loads the 4 web pages, splits them into chunks, embeds them, and creates 2 FAISS indexes.
   - Wraps each retriever as a tool the agent can call.

2. **User asks a question**
   - Query comes from the input box or suggested question buttons.
   - The app instantiates the selected LLM (`ChatOpenAI` or `ChatGroq`) with:
     - `temperature=0`
     - `max_tokens` from the sidebar

3. **Agent node (LLM + tools)**
   - The LLM is bound to retriever tools (`bind_tools`).
   - It either:
     - responds directly (no tool calls) → graph ends, OR
     - requests retrieval (tool call) → graph routes to `retrieve`.

4. **Retrieve node**
   - Executes the chosen retriever tool and returns documents as tool outputs (`ToolMessage`).

5. **Grade documents**
   - A structured-output check returns a strict `"yes"` / `"no"` relevance signal.
   - If any retrieved chunk is graded relevant → proceed to `generate`.
   - Otherwise → proceed to `rewrite`.

6. **Rewrite**
   - The model rewrites the query to better match the knowledge base.
   - Special handling: if the query mentions “Ragas” and retrieval failed, it clarifies it means the AI evaluation framework.
   - Loops back to the agent node to try again.

7. **Generate**
   - A RAG prompt combines user question + retrieved context.
   - The model produces a grounded answer (or says it doesn’t know).

8. **UI rendering**
   - Conversation history is shown with `st.chat_message`.
   - Optional: node execution logs appear in the “Agent Reasoning Steps” expander.

---

<a id="customization"></a>
## Customization

1) **Add / change knowledge sources**  
Edit the URL lists in `build_vectorstore()`:
- `urls_1` → agent-related sources  
- `urls_2` → RAGAS-related sources  

2) **Tune chunking & retrieval**  
In `build_vectorstore()`:
- `chunk_size` / `chunk_overlap` changes context granularity
- Retriever config:
  - `k`: how many chunks retrieved
  - `lambda_mult`: MMR diversity vs similarity balance

3) **Swap embedding model**  
Currently:
- tries a local Windows path
- falls back to MiniLM  
Replace `model_name=` with your preferred HuggingFace embedding model.

4) **Persist FAISS indexes (recommended improvement)**  
Right now FAISS is created in-memory (cached per Streamlit session). Extend the project to save/load FAISS to disk for faster startups.

5) **Implement OpenRouter / Gemini properly**  
The UI includes them, but the code uses a generic fallback for “other” providers. Add the correct LangChain integrations + environment variables to fully support them.

---

<a id="troubleshooting"></a>
## Troubleshooting

- **“Please enter your API Key in the sidebar.”**  
  Enter a valid provider key and rerun.

- **Embedding model load fails**
  - The hardcoded local embedding path may not exist on your machine.
  - Ensure `sentence-transformers` is installed.
  - Ensure your environment can download models if using the fallback.

- **Web pages fail to load / empty docs**
  - Source sites may block scraping or change HTML.
  - Your network may block requests.
  - Try replacing URLs or using a different loader.

- **Nothing appears in the final answer**
  - Enable **Show Agent Steps** to see which node executed and what it returned.

- **Rebuild doesn’t seem to change anything**
  - Click **Rebuild Pipeline** to clear `st.cache_resource` and rebuild the knowledge base.

---

<a id="known-limitations"></a>
## Known Limitations

- **No true conversation memory in the graph**  
  The UI displays chat history, but each run sends only the latest query into the graph:  
  `HumanMessage(content=final_query)`.

- **OpenRouter/Gemini are not fully implemented**  
  They appear in the UI, but the “else” branch uses `ChatOpenAI` as a generic fallback.

- **No persistent vector database**  
  FAISS is created in memory (cached by Streamlit). Restarting the app typically requires rebuilding embeddings unless you add persistence.

- **Hardcoded local embedding path**  
  The local model path is Windows-specific and machine-specific. Expect to change it for portability.

---

<a id="security-notes"></a>
## Security Notes

- Your API key is entered in the sidebar and assigned to environment variables at runtime (`os.environ[...]`).  
  Best practices:
  - Never commit secrets to Git
  - Use Streamlit secrets or deployment environment variables
  - Add `.env` to `.gitignore` (see [Project Structure](#project-structure))

- The app fetches external web pages at runtime. If deploying publicly, consider:
  - rate limiting
  - caching/persistence
  - allowlists for trusted domains

---

<a id="roadmap-ideas"></a>
## Roadmap Ideas

- Add real multi-provider support (OpenRouter, Gemini) with proper SDKs
- Persist FAISS indexes and only rebuild when sources change
- Add memory (feed prior messages into the graph)
- Add citations (source URL + chunk snippet) in the final answer
- Add RAG evaluation using RAGAS directly (very on-theme)

---

<a id="acknowledgements--sources"></a>
## Acknowledgements / Sources

This demo knowledge base is built from:

- Lilian Weng:
  - https://lilianweng.github.io/posts/2023-06-23-agent/
  - https://lilianweng.github.io/posts/2024-04-12-diffusion-video/

- RAGAS references:
  - https://docs.ragas.io/en/stable/#why-ragas
  - https://www.analyticsvidhya.com/blog/2024/05/a-beginners-guide-to-evaluating-rag-pipelines-using-ragas/

---

<a id="license"></a>
## License

Add your preferred license (e.g., MIT) in a `LICENSE` file.

---

<a id="project-structure"></a>
## Project Structure

Below is a clean, scalable structure for this project (recommended for organizing the code you shared):

```text
agentic-rag-project/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (gitignored)
├── .gitignore                # Git ignore rules
│
├── utils/                    # Utility functions
│   ├── vector_store.py       # FAISS and embedding management
│   └── agent_graph.py        # LangGraph workflow definition
│
├── config/                   # Configuration files
│   └── prompts.py            # Prompt templates
│
└── README.md                 # This documentation
```

### Suggested module responsibilities

- `utils/vector_store.py`
  - `build_vectorstore()`
  - document loading (URLs)
  - splitting + embedding + FAISS creation
  - retriever tool creation

- `utils/agent_graph.py`
  - `AgentState` definition
  - `initialize_graph(llm_model, tools_list)`
  - nodes: `agent`, `retrieve`, `grade_documents`, `rewrite`, `generate`

- `config/prompts.py`
  - grading prompt template
  - RAG generation prompt template
  - rewrite prompt template

### `.gitignore` suggestion (minimal)

```gitignore
.env
.venv/
__pycache__/
*.pyc
.streamlit/
```
```