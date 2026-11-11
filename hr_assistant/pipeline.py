import logging
from hr_assistant import config
from hr_assistant.llm import get_llm
from hr_assistant.tools import create_search_tool
from hr_assistant.vector_store import collection_exists, load_vector_store
from hr_assistant.ingestion import upload_corpus_to_gcs
from hr_assistant.ingestion import ingest_hr_policies
from hr_assistant.agent import create_hr_agent
from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger(__name__)


def _bootstrap_application(collection_name: str, ingest_fn) -> None:
    """First-run convenience: if the Qdrant collection isn't there yet,
    upload the corpus and ingest once. After that this is a no-op and
    startup is just a connect."""
    if collection_exists(collection_name):
        return

    logger.info(
        "Collection '%s' not found — running first-time ingestion.", collection_name
    )
    upload_corpus_to_gcs()
    ingest_fn()


def build_hr_assistant(checkpointer=None) -> object:
    """Connect to the clean `hr_policies` collection and build the agent
    (search tool + system prompt + memory). Returns a bare agent; drive it
    with ask().

    checkpointer defaults to an InMemorySaver (real chat memory for the CLI
    and Streamlit UI). Pass checkpointer=False from studio_graph.py to build
    the graph with none — `langgraph dev` supplies persistence itself and
    rejects a graph with one baked in."""

    config.check_api_keys()
    _bootstrap_application(config.QDRANT_COLLECTION_NAME, ingest_hr_policies)

    if checkpointer is None:
        checkpointer = InMemorySaver()
    elif checkpointer is False:
        checkpointer = None

    vector_store = load_vector_store(config.QDRANT_COLLECTION_NAME)
    search_tool = create_search_tool(vector_store)
    llm = get_llm()
    return create_hr_agent(
        get_llm(), [create_search_tool(vector_store)], checkpointer=checkpointer
    )


def ask(agent, question: str, thread_id: str = None) -> str:
    """Ask the agent a question. If thread_id is provided, the agent will
    remember the conversation context for that thread; otherwise it will
    start a new conversation."""
    if thread_id is None:
        thread_id = "default"

    response = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return response["messages"][-1].text
