import streamlit as st
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os


# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os

    name, extension = os.path.splitext(file)

    if extension == ".pdf":
        from langchain.document_loaders import PyPDFLoader

        print(f"Loading {file}")
        loader = PyPDFLoader(file)
    elif extension == ".docx":
        from langchain.document_loaders import Docx2txtLoader

        print(f"Loading {file}")
        loader = Docx2txtLoader(file)
    elif extension == ".txt":
        from langchain.document_loaders import TextLoader

        loader = TextLoader(file)
    elif extension == ".csv":
        from langchain_community.document_loaders.csv_loader import CSVLoader

        loader = CSVLoader(file)
    else:
        print("Document format is not supported!")
        return None

    data = loader.load()
    return data


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(data)
    return chunks


def create_embeddings_chroma(chunks, persist_directory="./chroma_db"):
    from langchain.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        model_name="sentence-transformers/all-MiniLM-l6-v2",
    )

    # Create a Chroma vector store using the provided text chunks and embedding model,
    # configuring it to save data to the specified directory
    vector_store = Chroma.from_documents(
        chunks, embeddings, persist_directory=persist_directory
    )

    return vector_store  # Return the created vector store


def ask_and_get_answer(vector_store, q, k=10):
    from langchain.chains import RetrievalQA
    from langchain_community.llms import HuggingFaceEndpoint

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=128,
        temperature=0.5,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )

    answer = chain.invoke(q)
    return answer


# def calculate_embedding_cost(texts):
#     import tiktoken
#     enc = tiktoken.encoding_for_model('text-embedding-3-small')
#     total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
#     # check prices here: https://openai.com/pricing
#     print(f'Total Tokens: {total_tokens}')
#     print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.00002:.6f}')

#     return total_tokens, total_tokens / 1000 * 0.00002


def clear_history():
    if "history" in st.session_state:
        del st.session_state["history"]


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(), override=True)

    st.image("logo.png", width=50)
    st.subheader("APP LLM Q&A con delle cose da migliorare")

    with st.sidebar:

        uploaded_file = st.file_uploader(
            "Upload a file: ", type=["pdf", "docx", "txt", "csv"]
        )
        chunk_size = st.number_input(
            "Chunk size: ", min_value=100, max_value=2048, value=512
        )
        k = st.number_input("k: ", min_value=1, max_value=20, value=10)
        add_data = st.button("Add Data")
        clear_data = st.button("Clear Data", type="primary", on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner("Sto facendo cose..."):
                bytes_data = uploaded_file.read()
                file_name = os.path.join("./", uploaded_file.name)

                with open(file_name, "wb") as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data=data, chunk_size=chunk_size)

                # tokens, embedding_cost = calculate_embedding_cost(chunks)
                # st.write(f'Embedding cost: ${embedding_cost:.4f}')

                vector_store = create_embeddings_chroma(chunks)

                st.session_state.vs = vector_store
                st.success("File uploaded")

    q = st.text_input("Ask a question: ")
    if q:
        if "vs" in st.session_state:
            vector_store = st.session_state.vs
            st.write(f"k: {k}")
            answer = ask_and_get_answer(vector_store, q, k)
            st.text_area("LLM Answer: ", value=answer["result"])

            st.divider()

            if "history" not in st.session_state:
                st.session_state.history = ""

            value = f'Q: {q} \nA: {answer["result"]}'
            st.session_state.history = (
                f'{value} \n {"-" * 100} \n {st.session_state.history}'
            )
            h = st.session_state.history
            st.text_area(label="Chat History", value=h, key="history", height=400)
