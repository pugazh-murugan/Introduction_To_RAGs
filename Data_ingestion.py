#create a sample file a load
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


text = """ Langchain is a framework for building AI App.
It helps connect language model with your own data.
Rag means Reterival Augments Generation."""

with open("sample.txt","w") as f:
    f.write(text)

print("sample.txt file created")



#TextLoader

loader =TextLoader("sample.txt")
documents = loader.load()

print("Number of the documents:",len(documents))
print("Context:")
print(documents[0].page_content)


#RecursiveCharacterTextSplitter

long_text = """Python is a programming language. It is easy to learn.
Streamlit is used to build web apps. LangChain helps build AI apps.
RAG combines search with AI models. Embeddings turn text into numbers.
Vector stores save those numbers. Retrievers find the best matches."""
splitter = RecursiveCharacterTextSplitter(
    chunk_size =80,
    chunk_overlap =10
)

chunks =splitter.split_text(long_text)

print("Number of chunks:", len(chunks))
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)

documents = TextLoader("sample.txt").load()
chunks = splitter.split_documents(documents)
for i, chunk in enumerate(chunks):
   print(f"\nChunk {i+1}:")
   print(chunk)