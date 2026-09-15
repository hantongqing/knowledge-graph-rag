import os
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc

WORKING_DIR = "./dickens"
os.environ["OMP_NUM_THREADS"] = "1"
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="llama3.2",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    ),
)

TEXT_DIR = "./text"

# Check if the directory exists
if not os.path.exists(TEXT_DIR):
    raise FileNotFoundError(f"Text directory '{TEXT_DIR}' does not exist.")

# Read and insert content from all text files in the directory
for filename in os.listdir(TEXT_DIR):
    if filename.endswith(".txt"):  # Process only .txt files
        filepath = os.path.join(TEXT_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            logging.info(f"Inserting content from file: {filename}")
            rag.insert(f.read())

# Perform naive search
print(rag.query("硬盘物理结构", param=QueryParam(mode="naive")))

# Perform local search
print(rag.query("硬盘存储结构", param=QueryParam(mode="local")))

# Perform global search
print(rag.query("硬盘的数据组织方式", param=QueryParam(mode="global")))

# Perform hybrid search
print(rag.query("数据读写过程", param=QueryParam(mode="hybrid")))
