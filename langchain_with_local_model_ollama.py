from langchain import hub
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.document_loaders import JSONLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.schema import LLMResult
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# load log file
loader = JSONLoader(file_path="/Users/mohami/workspacce/personal-repositories/ai_experiments/local_data/small.json",
                    jq_schema=".[].",
                    text_content=False)
data = loader.load()
# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=300)
all_splits = text_splitter.split_documents(data)
# Embed and store
vectorstore = Chroma.from_documents(documents=all_splits,
                                    embedding=OllamaEmbeddings(base_url="http://localhost:11434", model="llama2")
                                    )
# result = vectorstore.similarity_search(query="challenge", k=1)
# print(result)
#####
# RAG prompt
QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")


class GenerationStatisticsCallback(BaseCallbackHandler):
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print(response.generations[0][0].generation_info)


callback_manager = CallbackManager(
    [
        StreamingStdOutCallbackHandler(),
        # GenerationStatisticsCallback()
    ]
)

llm = Ollama(
    base_url="http://localhost:11434",
    model="llama2",
    verbose=True,
    callback_manager=callback_manager,
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

question = "Any customer having problem with enrollment? show the corresponding customerId and other identifiers if any"
# question = "Any errors"
# question = "what's going on with customer xxx"
result = qa_chain({"query": question})
