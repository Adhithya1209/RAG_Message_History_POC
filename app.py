from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq 
import chromadb

load_dotenv()

groq_api = os.getenv("GROQ_API")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize ChromaDB client with explicit settings
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Retrieval Augmented Generation from PDF uploads")
st.write("Upload Pdfs and retrieve information from them")
session_id = st.text_input("Provide session id", value="default_session")

def load_pdf():
    pdf_files = st.file_uploader("Select the pdfs", type="pdf", accept_multiple_files=True)
    if pdf_files:
        docs = []
        for files in pdf_files:
            temp_pdf = f"./temp_{files.name}"
            with open(temp_pdf, "wb") as pdf:
                pdf.write(files.getvalue())

            pdf_loader = PyPDFLoader(temp_pdf)
            documents = pdf_loader.load()
            docs.extend(documents)
            os.remove(temp_pdf)  # Clean up temp file
            st.success(f"Loaded {files.name} successfully")
        
        return docs

def create_session_id():
    if 'store' not in st.session_state:
        st.session_state.store = {}

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

class ConversationalLLM:
    def __init__(self, docs):
        self.docs = docs
        self.llm = None
        self.q_prompt = None
        self.answer_prompt = None
            
    def split_docs_vector_store(self):
        if self.docs:
            docs_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splitted_docs = docs_splitter.split_documents(self.docs)
            
            # Explicit ChromaDB collection creation
            vector_store = Chroma.from_documents(
                documents=splitted_docs,
                embedding=embeddings,
                client=chroma_client,
                collection_name="pdf_collection"
            )
            return vector_store

    def load_llm_model(self):
        try:
            self.llm = ChatGroq(groq_api_key=groq_api, model_name="Gemma2-9b-It")
        except Exception as e:
            st.error(f"Unable to load LLM model: {str(e)}")

    def create_prompts(self):
        q_system_prompt = ("Formulate a question based on the latest user question,"
                          "and the chat history. Do not answer it just reformulate,"
                          "the user question if required or return as it is")
        
        self.q_prompt = ChatPromptTemplate.from_messages([
            ("system", q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        answer_sys_prompt = ("You are an AI assistant trained to answer user questions,"
                           "using the following retrieved context. Use 4 sentences maximum"
                           "to answer.\n{context}")
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", answer_sys_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

    def create_retriever(self):
        if self.docs:
            self.load_llm_model()
            vector_store = self.split_docs_vector_store()
            if vector_store:
                self.create_prompts()
                retriever = vector_store.as_retriever()
                history_aware_retriever = create_history_aware_retriever(
                    self.llm, retriever, self.q_prompt
                )
                return history_aware_retriever
        else:
            st.warning("Upload the PDFs to begin")
            return None
    
    def create_run_chains(self):
        create_session_id()
        history_aware_retriever = self.create_retriever()
        
        if history_aware_retriever and self.llm and self.answer_prompt:
            qa_chain = create_stuff_documents_chain(self.llm, self.answer_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
            
            runnable_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            
            user_input = st.text_input("Ask your question")
            if user_input:
                session_history = get_session_history(session_id)
                response = runnable_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                st.write("AI:", response["answer"])

if __name__ == "__main__":
    docs = load_pdf()
    if docs:
        conversational_llm = ConversationalLLM(docs)
        conversational_llm.create_run_chains()