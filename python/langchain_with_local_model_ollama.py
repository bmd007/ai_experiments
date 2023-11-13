from langchain import hub
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.document_loaders import JSONLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.llms.ollama import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

persist_directory = './local_data/vector_db_persistence'
loader = JSONLoader(file_path="/local_data/try.json",
                    jq_schema=".[]",
                    text_content=False)
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=9000, chunk_overlap=500)
all_splits = text_splitter.split_documents(data)
vectorstore = Chroma.from_documents(documents=data,
                                    embedding=OllamaEmbeddings(base_url="http://localhost:11434", model="llama2")
                                    )
QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")
llm = Ollama(
    base_url="http://localhost:11434",
    model="llama2",
    verbose=True,
    callback_manager=StreamingStdOutCallbackHandler(),
    temperature=0
)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)
question = "Any customer having problem with enrollment?"
result = qa_chain({"query": question})
