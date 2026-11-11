from langchain.tools import tool
from hr_assistant import config
from hr_assistant.reranker import rerank
from hr_assistant.vector_store import get_retriever


def create_search_tool(vector_store):
    """Create a LangChain Tool that searches the vector store and reranks results."""

    @tool
    def search_hr_policy(question: str) -> str:
        """Search the HR policy documents for information about leave,
        work from home,
        probation, notice period,
        reimbursement, code of conduct,
        holidays, maternity/
        paternity leave, travel expenses, or the exit process."""

        retriever = get_retriever(
            vector_store=vector_store, top_k=config.RETRIEVER_TOP_K
        )
        candidantes = retriever.invoke(question)
        if not candidantes:
            return "No relevant documents found."
        reranked_docs = rerank(question, candidantes, top_n=config.RETRIEVER_TOP_K)
        return "\n\n".join(
            f"[Source: {c.metadata['source']}]\n{c.page_content}" for c in reranked_docs
        )

    return search_hr_policy
