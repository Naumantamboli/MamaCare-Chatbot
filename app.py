import streamlit as st
import pdfplumber
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv  

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") 

st.set_page_config(page_title="Pregnancy AI Assistant", layout="wide")

DATASET_DIR = "./"

def get_pdf_text(pdf_files):
    """Extracts text from PDFs using pdfplumber."""
    text = ""
    for pdf in pdf_files:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  
    return text.strip()

def get_text_chunks(text):
    """Splits text into chunks for processing."""
    if not text:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def store_pregnancy_embeddings(api_key):
    """Preprocesses and stores pregnancy-related embeddings from PDFs."""
    dataset_text = ""
    
    pdf_files = [file for file in os.listdir(DATASET_DIR) if file.lower().endswith(".pdf")]
    
    if not pdf_files:
        return False  # No PDFs found

    for file in pdf_files:
        file_path = os.path.abspath(file)  # Get absolute path
        dataset_text += get_pdf_text([file_path]) + "\n\n"
    
    if dataset_text:
        text_chunks = get_text_chunks(dataset_text)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("pregnancy_faiss_index")
        return True
    
    return False

def get_conversational_chain(api_key):
    """Creates a chatbot chain with a pregnancy-related prompt."""
    prompt_template = """
    You are an expert assistant for pregnancy-related questions.
    Provide answers based on reliable medical and health information.
    
    If the question is beyond general knowledge, suggest consulting a healthcare professional.
    
    Context:\n {context}\n
    Question: \n{question}\n
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question, api_key):
    """Handles user queries for pregnancy-related chatbot."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    try:
        store_name = "pregnancy_faiss_index"
        new_db = FAISS.load_local(store_name, embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain(api_key)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply:", response["output_text"])
    except Exception as e:
        st.error(f"Error processing request: {e}")

def main():
    """Main Streamlit UI logic."""
    st.header("      MotherCare-Expert🩷👩‍⚕️")

    # **Background Processing of Dataset (Runs Once)**
    if "dataset_processed" not in st.session_state:
        st.session_state["dataset_processed"] = False

    if not st.session_state["dataset_processed"]:
        with st.spinner("Processing dataset in the background..."):
            success = store_pregnancy_embeddings(api_key)
            if success:
                st.session_state["dataset_processed"] = True
            else:
                st.warning("No valid PDF files found. Please check the dataset folder.")

    user_question = st.text_input(
    "🔍 Ask a pregnancy-related question:", 
    key="user_question", 
    placeholder="Type your question here..."
)

    if user_question and api_key:
        user_input(user_question, api_key)

if __name__ == "__main__":
    main()
