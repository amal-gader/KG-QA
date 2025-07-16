from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader




embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")



model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"

# llm = LlamaCPP(
#     model_url=model_url,
#     model_path=None,
#     temperature=0.1,
#     max_new_tokens=256,
#     context_window=5000,
#     generate_kwargs={},
#     model_kwargs={"n_gpu_layers": 1},
#     verbose=True,
# )

llm = LlamaCPP(
     model_url=model_url,
    temperature=0.1,
    max_new_tokens=256,
    context_window=5000,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 80},  # as high as VRAM allows
    verbose=True,
)

Settings.embed_model=embed_model
Settings.llm = llm


documents = SimpleDirectoryReader("pdfs").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(embed_model=embed_model)


response = query_engine.query("What are the difficulties when applying RAG?")
print(response)