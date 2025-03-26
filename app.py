import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    
    if not text.strip():
        st.error("⚠️ No extractable text found in the uploaded PDF(s). Try another file.")
    
    return text.strip()

def get_text_chunks(text):
    """Splits text into manageable chunks."""
    if not text:
        return []
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        st.error("⚠️ Text splitting failed. No chunks were created.")
    
    return chunks

def get_vectorstore(text_chunks):
    """Generates vector embeddings using Hugging Face and stores them in FAISS."""
    if not text_chunks:
        st.error("⚠️ No text chunks available to generate embeddings.")
        return None

    try:
        embeddings = HuggingFaceInstructEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"⚠️ Error while generating embeddings: {str(e)}")
        return None

def get_conversation_chain(vectorstore):
    """Creates a chatbot conversation chain using FAISS and Hugging Face API."""
    if not vectorstore:
        return None

    try:
        llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct", model_kwargs={"temperature": 0.5})
        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        st.error(f"⚠️ Error creating conversation chain: {str(e)}")
        return None

def handle_userinput(user_question):
    """Handles user input and generates chatbot responses."""
    if "conversation" in st.session_state and st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.error("⚠️ No conversation chain found. Please upload and process a document first.")

def main():
    """Main function to run the Streamlit chatbot app."""
    load_dotenv()
    st.set_page_config(page_title="ChatBot", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Interactive RAG- based LLM for Multi-PDF Document Analysis :books:")
    
    user_question = st.text_input("Ask a question about your PDF Files:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDF Files here and click 'Process' Button", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Step 1: Extract text from PDFs
                raw_text = get_pdf_text(pdf_docs)

                # Step 2: Split text into chunks
                text_chunks = get_text_chunks(raw_text)
                if not text_chunks:
                    return  # Stop processing if no chunks are available

                # Step 3: Generate vector store embeddings
                vectorstore = get_vectorstore(text_chunks)
                if not vectorstore:
                    return  # Stop if embedding fails

                # Step 4: Create the chatbot conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
                if st.session_state.conversation:
                    st.success("✅ Processing complete! You can now ask questions.")

if __name__ == '__main__':
    main()
