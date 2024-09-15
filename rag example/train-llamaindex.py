#Install dependencies
#------------------------------------------------------------------------
#pip install ollama-python pypdf2
#pip install llama-index-llms-ollama
#pip install llama-index llama-index-llms-openllm llama-index-embeddings-huggingface
#python -m pip install langchain
#pip install -U langchain-community llama-index-embeddings-langchain
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext
#from llama_index.llms.openllm import OpenLLMAPI
from llama_index.core.node_parser import SentenceSplitter
from langchain.embeddings import OllamaEmbeddings
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

emb = OllamaEmbeddings(model="llama3")

Settings.embed_model = emb
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 3900
Settings.transformations = [SentenceSplitter(chunk_size=1024)]

#listens to localhost:11434 by default
llm = Ollama(base_url="http://localhost:11434", model="llama3", request_timeout=120.0)

Settings.llm = llm

# Break down the document into manageable chunks (each of size 1024 characters, with a 20-character overlap)
text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

storage_context = StorageContext.from_defaults(
    vector_store=emb
)

# Load documents from the data directory
documents = SimpleDirectoryReader(
    input_files=["rag-training.txt"]
)

documents = documents.load_data();

# Build an index over the documents
index = VectorStoreIndex.from_documents(
    documents, embed_model=emb, transformations=Settings.transformations
)

# Query your data using the built index
query_engine = index.as_query_engine()
response = query_engine.query("Who is Patroclos?")
print(response)