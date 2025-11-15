import os
import re
from typing import List, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langgraph.graph import END, StateGraph

# ==============================
#       LOAD ENV + MODELS
# ==============================

load_dotenv()

# --- Azure OpenAI Config ---
AZURE_EMBEDDING_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBED_MODEL")

AZURE_LLM_KEY = os.getenv("AZURE_OPENAI_LLM_API_KEY")
AZURE_LLM_ENDPOINT = os.getenv("AZURE_OPENAI_LLM_ENDPOINT")
AZURE_LLM_MODEL_NAME = os.getenv("AZURE_OPENAI_LLM_MODEL")

AZURE_API_VERSION = "2023-05-15"

if not all([AZURE_EMBEDDING_KEY, AZURE_EMBEDDING_ENDPOINT, AZURE_EMBEDDING_MODEL_NAME,
            AZURE_LLM_KEY, AZURE_LLM_ENDPOINT, AZURE_LLM_MODEL_NAME]):
    raise ValueError("Azure OpenAI environment variables not completely set.")

# --- Chroma Config for Vercel ---
# Vercel copies the project files to /var/task/, so we build the path from there.
# The 'chroma_db' directory must be committed to your repository.
current_dir = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(current_dir, "chroma_db")
COLLECTION_NAME = "traffic_law_2024"

# --- LangChain Clients ---
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_EMBEDDING_MODEL_NAME,
    model=AZURE_EMBEDDING_MODEL_NAME,
    azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
    api_key=AZURE_EMBEDDING_KEY,
    api_version=AZURE_API_VERSION,
)

llm = AzureChatOpenAI(
    azure_deployment=AZURE_LLM_MODEL_NAME,
    model=AZURE_LLM_MODEL_NAME,
    azure_endpoint=AZURE_LLM_ENDPOINT,
    api_key=AZURE_LLM_KEY,
    api_version=AZURE_API_VERSION,
    temperature=0,
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embeddings,
)

# ==============================
#   CUSTOM SMART RETRIEVER (TOOL)
# ==============================

class SmartRerankingRetriever(BaseRetriever):
    vectorstore: Chroma
    top_k: int = 8

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=self.top_k)

smart_retriever = SmartRerankingRetriever(vectorstore=vectorstore)

# ==============================
#       LANGGRAPH AGENTIC RAG
# ==============================

class GraphState(TypedDict):
    original_question: str
    question: str
    generation: str
    documents: List[Document]
    chat_history: List[BaseMessage]

# --- NODES ---

def condense_question(state):
    original_question = state["question"]
    chat_history = state["chat_history"]

    if not chat_history:
        return {"question": original_question, "original_question": original_question}

    condense_prompt = PromptTemplate.from_template("""
    Dựa vào lịch sử trò chuyện và câu hỏi mới, hãy viết lại câu hỏi mới thành một câu hỏi độc lập, đầy đủ ngữ nghĩa.
    
    Lịch sử trò chuyện:
    {chat_history}
    
    Câu hỏi mới: {question}
    
    Câu hỏi độc lập:""")
    
    condenser = condense_prompt | llm | StrOutputParser()
    
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
    condensed_question = condenser.invoke({"chat_history": history_str, "question": original_question})
    
    return {"question": condensed_question, "original_question": original_question}

def retrieve_documents(state):
    question = state["question"]
    documents = smart_retriever.invoke(question)
    return {"documents": documents}

def generate_answer(state):
    original_question = state["original_question"]
    documents = state["documents"]
    chat_history = state["chat_history"]

    if not documents:
        generation = "Rất tiếc, tôi không tìm thấy thông tin nào liên quan trong kho dữ liệu để trả lời câu hỏi của bạn."
    else:
        system_prompt = """Bạn là trợ lý pháp lý tiếng Việt, chuyên về Luật Giao thông đường bộ.
        Bạn CHỈ được trả lời dựa trên NGỮ CẢNH được cung cấp.
        Sử dụng lịch sử trò chuyện để cuộc hội thoại tự nhiên hơn."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            *chat_history,
            ("user", "Ngữ cảnh:\n{context}\n---\nCâu hỏi: {question}"),
        ])

        rag_chain = (
            {"context": lambda x: format_docs(x["documents"]), "question": lambda x: x["original_question"]}
            | prompt
            | llm
            | StrOutputParser()
        )
        generation = rag_chain.invoke({"documents": documents, "original_question": original_question})

    return {"generation": generation}

# --- BUILD GRAPH ---

workflow = StateGraph(GraphState)

workflow.add_node("condense_question", condense_question)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)

workflow.set_entry_point("condense_question")
workflow.add_edge("condense_question", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# ==============================
#       HELPER FUNCTIONS
# ==============================

def format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join([f"Nguồn: {doc.metadata.get('source_file')} - Điều {doc.metadata.get('article_number')}\n{doc.page_content}" for doc in docs])
