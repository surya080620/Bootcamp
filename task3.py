from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama

# Load PDFs
files = ["sample1.pdf", "sample2.pdf"]
docs = []

for f in files:
    loader = PyPDFLoader(f)
    docs.extend(loader.load())

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Exercise 1: Vector DB
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma.from_documents(chunks, embeddings)

# Exercise 2: Retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# Exercise 3: RAG
llm = ChatOllama(model="qwen2.5:1.5b")

def ask(question):
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Answer using only the context below.

Context:
{context}

Question:
{question}
"""

    answer = llm.invoke(prompt)

    return docs, answer.content

# Exercise 4: Test
questions = [
    "What is this document about?",
    "Summarize it",
    "What are the key points?"
]

for q in questions:
    print("\n" + "="*40)
    print("Question:", q)

    docs, ans = ask(q)

    print("\nChunks:")
    for d in docs:
        print("-", d.page_content[:150])

    print("\nAnswer:")
    print(ans)