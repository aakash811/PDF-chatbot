import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import time

# Global QA chain
qa_chain = None

# Load and process PDF
def load_pdf(file):
    loader = PyPDFLoader(file.name)
    docs = loader.load_and_split()
    
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    retriever = vectorstore.as_retriever()
    
    llm = ChatGroq(
        groq_api_key="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",
        model_name="llama3-8b-8192"
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa

# Called on file upload
def set_pdf(file):
    global qa_chain
    # Show loading status
    status = "⏳ Processing PDF, please wait..."
    yield status  # Intermediate result to show loading

    qa_chain = load_pdf(file)

    # Show success after processing
    yield "✅ PDF uploaded and processed! You can now ask questions."

# Handles chat input
def answer_question(message, history):
    if qa_chain is None:
        return "⚠️ Please upload a PDF first."
    
    try:
        response = qa_chain.invoke({"query": message})
        if isinstance(response, dict) and "result" in response:
            return response["result"]
        return str(response)
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📄 PDF Chatbot with LangChain + Gradio")

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_button = gr.Button("Process PDF")
    
    status_output = gr.Markdown()

    # Use generator function with `upload_button.click` to allow step-by-step updates
    upload_button.click(fn=set_pdf, inputs=[pdf_input], outputs=[status_output])

    gr.ChatInterface(fn=answer_question, title="💬 Ask your PDF")

demo.launch()
