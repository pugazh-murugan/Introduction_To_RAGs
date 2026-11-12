
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

long_text = """Python is a programming language. It is easy to learn.
Streamlit is used to build web apps. LangChain helps build AI apps.
RAG combines search with AI models. Embeddings turn text into numbers.
Vector stores save those numbers. Retrievers find the best matches."""

print("Total characters:", len(long_text))



splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,      # each chunk is about 80 characters
    chunk_overlap=10    # 10 characters repeat between chunks
)

chunks = splitter.split_text(long_text)

# print("Number of chunks:", len(chunks))
# for i, chunk in enumerate(chunks):
#     print(f"\nChunk {i+1}:")
#     print(chunk)


documents = TextLoader("sample.txt").load()
chunkss = splitter.split_documents(documents)
print("Number of chunkss:", len(chunkss))
for i, chunk in enumerate(chunkss):
    print(f"\chunkss {i+1}:")
    print(chunkss)