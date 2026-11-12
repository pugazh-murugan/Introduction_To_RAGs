from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Our small knowledge base (a list of sentences)
texts = [
    "Python is a programming language.",
    "Streamlit is used to build web apps.",
    "Dogs are friendly animals.",
    "Cats like to sleep a lot."
]

# Free, local embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Ready!")

#Store the text in FAISS
vector_store = FAISS.from_texts(texts, embeddings)

print("All texts are stored in the vector store!")

#Search by meaning
query = "Tell me about pets"

results = vector_store.similarity_search(query, k=2)   # k=2 means top 2 matches

print("Top matches for:", query)
for r in results:
    print("-", r.page_content)

#Save and load the store
vector_store.save_local("my_store")

# load it back later
vector_store = FAISS.load_local("my_store", embeddings, allow_dangerous_deserialization=True)