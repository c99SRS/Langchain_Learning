import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama


load_dotenv()

print("Initializing components...")


# Pull once: ollama pull nomic-embed-text
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
        )
print(f"Embedding with Ollama model '{EMBED_MODEL}' at {OLLAMA_BASE_URL}...")

#llm = ChatOllama(model="llama3.1:8b", temperature=0) 
llm = ChatOllama(model="qwen3:1.7b", temperature=0)


#vectorstore = FAISS.from_documents(texts, embedding)


vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

    {context}
    Question: {question}
    Provide a detailed answer:"""
)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


# ============================================================================
# IMPLEMENTATION 1: Without LCEL (Simple Function-Based Approach)
# ============================================================================
def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)
    print(f"docs: {docs} ")
    print("\n\n")

    # Step 2: Format documents into context string
    context = format_docs(docs)
    print(f"context: {context}")
    print("\n\n")

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)
    print(f"messages: {messages}")
    print("\n\n")

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)
    print(f"response: {response}")
    print("\n\n")


    # Step 5: Return the content
    return response.content



def main():
    print("Hello from rag-gist!")


if __name__ == "__main__":
    main()
    print("Retrieving......")

    query="What is Pinecone in machine learning?"

     # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    # print("\n" + "=" * 70)
    # print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    # print("=" * 70)
    # result_raw=llm.invoke([HumanMessage(content=query)])
    # print("\nAnswer:")
    # print(result_raw.content)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

