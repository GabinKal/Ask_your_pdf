
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain





def main():
    load_dotenv()
    st.set_page_config(page_title= "Ast your PDF")
    st.header("Ask you PDF 💬")

    pdf= st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        # split into chunks

        text_splitter= CharacterTextSplitter(
            separator= "\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks= text_splitter.split_text(text)
        
        
        #create embeddings
        embeddings= OpenAIEmbeddings()
        knowledge_base= FAISS.from_texts(chunks, embeddings)

        # user prompt 

        user_question= st.text_input("Ask your question about the PDF")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            



            llm = ChatOpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response= chain.run(input_documents=docs, question=user_question)
            st.write(response)


if __name__== '__main__':
    main()
