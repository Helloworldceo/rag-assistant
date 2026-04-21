import os
from typing import List, Tuple

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from rag.config import DEFAULT_LLM_MODEL, DEFAULT_TEMPERATURE, DEEPSEEK_BASE_URL

# ── Prompts ───────────────────────────────────────────────────────────────────

_CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given a chat history and the latest user question which might reference "
        "context in the chat history, formulate a standalone question that can be "
        "understood without the chat history. "
        "Do NOT answer the question — just reformulate it if needed, otherwise return it as-is.",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

_QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that answers questions strictly based on the "
        "provided document excerpts. "
        "If the answer is not contained in the context, say you don't know — "
        "never fabricate information. "
        "Be concise but complete. When applicable, mention which document the "
        "information comes from.\n\n"
        "Retrieved context:\n{context}",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ── Chain ─────────────────────────────────────────────────────────────────────

class RAGChain:
    """
    Conversational RAG chain built with pure LCEL (langchain_core only).

    Steps per query:
    1. If chat history exists, rewrite the question to be standalone.
    2. Retrieve the top-k relevant chunks from the vector store.
    3. Generate an answer grounded in the retrieved chunks.
    """

    def __init__(
        self,
        retriever,
        model: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self._retriever = retriever
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL,
        )
        self._contextualize_chain = _CONTEXTUALIZE_Q_PROMPT | llm | StrOutputParser()
        self._qa_chain = _QA_PROMPT | llm | StrOutputParser()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_lc_history(self, chat_history: List[Tuple[str, str]]):
        messages = []
        for human_msg, ai_msg in chat_history:
            messages.append(HumanMessage(content=human_msg))
            messages.append(AIMessage(content=ai_msg))
        return messages

    def _contextualize(self, question: str, lc_history: list) -> str:
        """Return a standalone version of the question when history is present."""
        if not lc_history:
            return question
        return self._contextualize_chain.invoke({
            "input": question,
            "chat_history": lc_history,
        })

    # ── Public API ────────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        chat_history: List[Tuple[str, str]],
    ) -> dict:
        """
        Ask a question given a conversation history.

        Returns:
            {"answer": str, "source_documents": List[Document]}
        """
        lc_history = self._to_lc_history(chat_history)

        # Step 1 – rewrite question to be standalone
        standalone_q = self._contextualize(question, lc_history)

        # Step 2 – retrieve relevant chunks
        docs = self._retriever.invoke(standalone_q)

        # Step 3 – build context string and generate answer
        context = "\n\n".join(doc.page_content for doc in docs)
        answer = self._qa_chain.invoke({
            "input": question,
            "chat_history": lc_history,
            "context": context,
        })

        return {
            "answer": answer,
            "source_documents": docs,
        }
