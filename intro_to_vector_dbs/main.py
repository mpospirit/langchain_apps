from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


def format_docs(docs):
    """
    For the LCEL (LangChain Expression Language) example below
    """
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print("Retrieving...")

    # Initialize embeddings and language model
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    # Simple query example
    query = "What is the meaning of life?"
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})
    print(result.content)

    print("================================")

    # Initialize Pinecone vector store
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    # Create retrieval chain
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    # Query the retrieval chain
    result = retrieval_chain.invoke(input={"input": query})

    print(result["answer"])

    print("================================")

    # Another query example
    serious_query = "What is Pinecone?"

    result = retrieval_chain.invoke(input={"input": serious_query})

    print(result["answer"])


    # LCEL (LangChain Expression Language) example
    print("================================")

    template = """
    Use the following pieces of context as for the question at the end. 
    If you don't know the answer just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the asnwer as concise as possible.
    Always say "Thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:
    """

    custom_rag_prompt = PromptTemplate.from_template(template=template)

    rag_chain = (
        {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    # Query the custom RAG chain
    result = rag_chain.invoke(query)

    print(result.content)
    print("================================")

    result = rag_chain.invoke(serious_query)

    print(result.content)
