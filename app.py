import streamlit as st
import os
from typing import List, Annotated, Literal, TypedDict, Sequence
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

# --- Page Configuration ---
st.set_page_config(page_title="Agentic RAG Assistant", layout="wide")

# --- UI: Sidebar Configuration ---
st.sidebar.header("Configuration Panel")

provider = st.sidebar.selectbox(
    "AI Provider",
    ["OpenAI", "Groq", "OpenRouter", "Gemini"]
)

if provider == "OpenAI":
    model_options = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
elif provider == "Groq":
    model_options = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
elif provider == "Gemini":
    model_options = ["gemini-2.0-flash", "gemini-2.5-flash"]
else:
    model_options = ["openai/gpt-4o", "meta-llama/llama-3-70b-instruct"]

selected_model = st.sidebar.selectbox("Model", model_options)

api_key = st.sidebar.text_input("API Key", type="password", help=f"Enter your {provider} API Key")

# New: Token Control
max_tokens = st.sidebar.number_input("Max Response Tokens", min_value=50, max_value=4096, value=500, step=50)

show_steps = st.sidebar.toggle("Show Agent Steps", value=False)

if st.sidebar.button("Rebuild Pipeline"):
    st.cache_resource.clear()
    st.session_state.pop("graph", None)
    st.rerun()

st.sidebar.markdown("---")
with st.sidebar.expander("About"):
    st.write("Tech Stack: LangChain, LangGraph, Streamlit, FAISS.")

# --- Helper Functions & Setup ---

@st.cache_resource
def build_vectorstore():
    """
    Builds the vector store and retrievers.
    Includes fallback if your local path isn't found (for portability).
    """
    local_path = r"C:\Users\Abdelrahman\.cache\huggingface\hub\models--google--embeddinggemma-300m\snapshots\64614b0b8b64f0c6c1e52b07e4e9a4e8fe4d2da2"
    
    try:
        if os.path.exists(local_path):
            embedding = HuggingFaceEmbeddings(model_name=local_path)
        else:
            st.warning(f"Local path not found: {local_path}. Falling back to 'all-MiniLM-L6-v2' for demo purposes.")
            embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
         st.error(f"Error loading embedding model: {e}")
         return None

    # --- Data Loading ---
    urls_1 = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]
    
    urls_2 = [
        "https://www.analyticsvidhya.com/blog/2024/05/a-beginners-guide-to-evaluating-rag-pipelines-using-ragas/",
        "https://docs.ragas.io/en/stable/#why-ragas"
    ]

    loaders_1 = [WebBaseLoader(url) for url in urls_1]
    docs_1 = []
    for loader in loaders_1:
        docs_1.extend(loader.load())

    loaders_2 = [WebBaseLoader(url) for url in urls_2]
    docs_2 = []
    for loader in loaders_2:
        docs_2.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_splits_1 = text_splitter.split_documents(docs_1)
    doc_splits_2 = text_splitter.split_documents(docs_2)

    vectorstore_1 = FAISS.from_documents(documents=doc_splits_1, embedding=embedding)
    vectorstore_2 = FAISS.from_documents(documents=doc_splits_2, embedding=embedding)

    retriever_1 = vectorstore_1.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'lambda_mult': 0.5})
    retriever_2 = vectorstore_2.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'lambda_mult': 0.5})

    retriever_tool_1 = create_retriever_tool(
        retriever_1,
        "Agentic_blog",
        "Search and run information about agents"
    )
    retriever_tool_2 = create_retriever_tool(
        retriever_2,
        "Ragas_blog",
        "Search and run information about RAGAS (RAG Assessment) framework"
    )

    return [retriever_tool_1, retriever_tool_2]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def initialize_graph(llm_model, tools_list):
    
    llm_with_tools = llm_model.bind_tools(tools_list)

    def agent(state):
        messages = state['messages']
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
        """
        Determines whether the retrieved documents are relevant to the question.
        """
        class grade(TypedDict):
            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        # Enforce strict 'yes' or 'no'
        model = llm_model
        llm_with_structured = model.with_structured_output(grade)

        # Updated Prompt: Added explicit instruction to handle the Ragas ambiguity
        prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )
        
        chain = prompt | llm_with_structured
        messages = state['messages']
        question = messages[0].content
        
        tool_outputs = [msg.content for msg in messages if isinstance(msg, ToolMessage)]

        if not tool_outputs:
            return "rewrite"

        # Check ANY valid document
        relevant_found = False
        for doc in tool_outputs:
            try:
                scored_result = chain.invoke({"question": question, "context": doc})
                score = scored_result.get('binary_score', 'no')
                if score == "yes":
                    relevant_found = True
                    break 
            except:
                continue

        if relevant_found:
            return "generate"
        else:
            return "rewrite"

    def generate(state):
        messages = state['messages']
        question = messages[0].content
        last_message = messages[-1]
        
        # If we got here via retrieve -> generate, last msg is ToolMessage
        if isinstance(last_message, ToolMessage):
             docs = last_message.content
        else:
             # Fallback: search history for tool outputs
             docs = "\n".join([m.content for m in messages if isinstance(m, ToolMessage)])

        prompt = PromptTemplate.from_template(
            """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            
            Question: {question} 
            Context: {context} 
            
            Answer:"""
        )

        # Ensure we don't exceed max tokens defined in sidebar
        rag_chain = prompt | llm_model | StrOutputParser()
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}

    def rewrite(state):
        messages = state['messages']
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f"""Look at the input and try to reason about the underlying semantic intent / meaning. 
                If the query mentions 'Ragas' and retrieval failed, clarify it is about the AI Framework.
                Here is the initial question:
                \n ------- \n
                {question} 
                \n ------- \n
                Formulate an improved question: """,
            )
        ]

        response = llm_model.invoke(msg)
        return {"messages": [response]}

    # --- Graph Construction ---
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent)
    retrieve_node = ToolNode(tools_list)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "agent")
    
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )
    
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    return workflow.compile()


# --- Main Application Logic ---

st.title("Agentic RAG Assistant")
st.subheader("Ask questions about AI Agents or RAGAS evaluation framework.")

# 1. Initialize VectorStore (Embed Docs) BEFORE questions
# This runs once due to @st.cache_resource
with st.spinner("Embedding documents and preparing knowledge base..."):
    tools = build_vectorstore()
    if tools:
        st.success("Documents embedded and Vector Store ready!", icon="✅")
    else:
        st.stop()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Input Box
user_input = st.text_input("Enter your question", key="query_input")

# Suggested Questions as Buttons (Below Input)
st.markdown("##### Suggested Questions:")
examples = [
    "What is RAGAS?",
    "How do agents differ from traditional LLMs?",
    "Explain ReAct prompting",
    "What metrics does RAGAS use?"
]

# Create columns for buttons
cols = st.columns(len(examples))
triggered_example = None
for i, ex in enumerate(examples):
    if cols[i].button(ex, use_container_width=True):
        triggered_example = ex

# Determine what to run
final_query = None
if st.button("Run Query", type="primary"):
    final_query = user_input
elif triggered_example:
    final_query = triggered_example

col_clear = st.button("Clear Chat")
if col_clear:
    st.session_state.messages = []
    st.rerun()

# Processing Logic
if final_query:
    if not api_key:
        st.error("Please enter your API Key in the sidebar.")
        st.stop()
    
    # 2. Setup LLM with Max Tokens
    if provider == "OpenAI":
        os.environ["OPENAI_API_KEY"] = api_key
        try:
            llm = ChatOpenAI(model=selected_model, temperature=0, max_tokens=max_tokens)
        except Exception as e:
            st.error(f"Error initializing OpenAI: {e}")
            st.stop()
    elif provider == "Groq":
        os.environ["GROQ_API_KEY"] = api_key
        try:
            llm = ChatGroq(model=selected_model, temperature=0, max_tokens=max_tokens)
        except Exception as e:
            st.error(f"Error initializing Groq: {e}")
            st.stop()
    else:
        # Generic fallback
        os.environ["OPENAI_API_KEY"] = api_key
        llm = ChatOpenAI(model=selected_model, temperature=0, max_tokens=max_tokens)

    # 3. Compile Graph
    app = initialize_graph(llm, tools)

    # 4. Display User Message
    st.session_state.messages.append({"role": "user", "content": final_query})
    
    # 5. Run Graph
    inputs = {"messages": [HumanMessage(content=final_query)]}
    
    # UI Containers
    result_container = st.container()
    steps_container = st.expander("Agent Reasoning Steps", expanded=show_steps)
    
    final_response_text = ""
    
    try:
        with st.spinner("Agent is thinking..."):
            for event in app.stream(inputs):
                for key, value in event.items():
                    if show_steps:
                        steps_container.write(f"**Node '{key}'** executed.")
                        steps_container.json(value)
                    
                    # Capture generation
                    if key == "generate":
                        messages = value.get("messages", [])
                        if messages:
                            final_response_text = messages[-1].content
                    
                    # Capture direct answer from agent (no retrieval needed)
                    if key == "agent":
                        messages = value.get("messages", [])
                        if messages:
                            last_msg = messages[-1]
                            if isinstance(last_msg, BaseMessage) and not last_msg.tool_calls:
                                final_response_text = last_msg.content

            # If loop finished but variable empty (fallback)
            if not final_response_text:
                 final_response_text = "The agent finished but provided no text output. Check 'Show Agent Steps' for details."

            st.session_state.messages.append({"role": "assistant", "content": final_response_text})
        
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Render Conversation History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])