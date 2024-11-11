import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class RAGApplication:
    def __init__(self, pdf_path, model_name="gemini-1.5-pro"):
        self.pdf_path = pdf_path
        self.model_name = model_name
        self.loader = PyPDFLoader(self.pdf_path)
        self.data = self.loader.load()
        self.docs = self._split_text()
        self.vectorstore = self._create_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        self.llm = self._initialize_llm()

    def _split_text(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        return text_splitter.split_documents(self.data)

    def _create_vectorstore(self):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return Chroma.from_documents(documents=self.docs, embedding=embeddings)

    def _initialize_llm(self):
        return ChatGoogleGenerativeAI(model=self.model_name, temperature=0, max_tokens=None, timeout=None)

    def create_prompt(self, query):
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", query),
            ]
        )

    def get_response(self, query):
        if query:
            prompt = self.create_prompt(query)
            question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
            rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
            response = rag_chain.invoke({"input": query})
            return response.get("answer", "No response provided.")
        return "Please enter a query."

def main():
    st.title("RAG Application built on Gemini Model")
    rag_app = RAGApplication(pdf_path="documents/12.pdf")
    
    query = st.chat_input("Say something: ")
    response = rag_app.get_response(query)
    
    if response:
        st.write(response)

if __name__ == "__main__":
    main()


