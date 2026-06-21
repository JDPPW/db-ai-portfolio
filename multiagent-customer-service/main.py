"""
Multi-Tenant Customer Service AI Agent
Author: Dujuan Brown | Iron Gate Solutions US LLC
GitHub: github.com/JDPPW/db-ai-portfolio

Enterprise customer service agent platform for RetailCore Group.
Serves three subsidiary brands from one system:
- NovaTech Electronics
- HomeBase Supply  
- StyleMax Retail

Agent modes: Resolve, Escalate, Clarify
Built with LangGraph for state management, RAG for brand-specific
policy retrieval, Claude for natural language generation.

Target roles: GE Appliances, Conduet, VetsEZ, QTech
"""

import os
import json
import datetime
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
MODEL = "claude-haiku-4-5-20251001"

# ── Brand configuration ───────────────────────────────────────────────────────
BRANDS = {
    "novatech": {
        "name": "NovaTech Electronics",
        "policy_file": "novatech_policy.txt",
        "color": "#0066CC",
        "tagline": "Smart Technology, Smarter Support"
    },
    "homebase": {
        "name": "HomeBase Supply",
        "policy_file": "homebase_policy.txt",
        "color": "#E87722",
        "tagline": "Built for the Job, Built for You"
    },
    "stylemax": {
        "name": "StyleMax Retail",
        "policy_file": "stylemax_policy.txt",
        "color": "#8B2FC9",
        "tagline": "Style That Works for You"
    }
}

# ── Agent state definition ────────────────────────────────────────────────────
# This is what LangGraph tracks as the conversation moves through nodes
class AgentState(TypedDict):
    brand_id: str
    brand_name: str
    customer_message: str
    conversation_history: list
    retrieved_context: str
    agent_response: str
    decision: Literal["resolve", "escalate", "clarify"]
    frustration_count: int
    exchange_count: int
    case_summary: str

# ── Vector store per brand ────────────────────────────────────────────────────
def build_vectorstore(brand_id: str) -> Chroma:
    """Load brand policy and build ChromaDB collection."""
    base_dir = os.path.dirname(__file__)
    
    # Load brand-specific policy
    brand_file = os.path.join(base_dir, "knowledge_base", BRANDS[brand_id]["policy_file"])
    global_file = os.path.join(base_dir, "knowledge_base", "retailcore_global_policy.txt")
    
    loader_brand = TextLoader(brand_file)
    loader_global = TextLoader(global_file)
    
    docs = loader_brand.load() + loader_global.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    # Each brand gets its own ChromaDB collection
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=None,
        collection_name=f"retailcore_{brand_id}",
        persist_directory=os.path.join(base_dir, "chroma_db")
    )
    return vectorstore

# Cache vectorstores so we don't rebuild on every message
_vectorstore_cache = {}

def get_vectorstore(brand_id: str) -> Chroma:
    if brand_id not in _vectorstore_cache:
        _vectorstore_cache[brand_id] = build_vectorstore(brand_id)
    return _vectorstore_cache[brand_id]

# ── LangGraph nodes ───────────────────────────────────────────────────────────

def retrieve_context(state: AgentState) -> AgentState:
    """Retrieve relevant policy chunks for this brand and message."""
    vectorstore = get_vectorstore(state["brand_id"])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(state["customer_message"])
    state["retrieved_context"] = "\n\n".join(doc.page_content for doc in docs)
    return state


def classify_intent(state: AgentState) -> AgentState:
    """
    Decide: resolve, escalate, or clarify.
    Auto-escalate if frustration is high or exchanges exceeded.
    """
    # Auto-escalate triggers
    if state["frustration_count"] >= 2 or state["exchange_count"] >= 2:
        state["decision"] = "escalate"
        return state
    
    llm = ChatAnthropic(model=MODEL, api_key=api_key, temperature=0)
    
    history_context = ""
    if state["conversation_history"]:
        history_context = "\n".join([
            f"{'Customer' if m['role'] == 'user' else 'Agent'}: {m['content']}"
            for m in state["conversation_history"][-4:]
        ])

    classify_prompt = f"""You are a routing agent for {state["brand_name"]} customer service.

Conversation so far:
{history_context}

Latest customer message: {state["customer_message"]}

Classify this as exactly one of:
- resolve: you have enough information to fully help this customer
- clarify: you need one more piece of information before helping
- escalate: this is complex, emotional, involves safety, or requires human judgment

Respond with only one word: resolve, clarify, or escalate"""

    response = llm.invoke(classify_prompt)
    decision = response.content.strip().lower()
    
    if decision not in ["resolve", "clarify", "escalate"]:
        decision = "resolve"
    
    state["decision"] = decision
    return state


def generate_response(state: AgentState) -> AgentState:
    """Generate the actual customer-facing response."""
    llm = ChatAnthropic(model=MODEL, api_key=api_key, temperature=0.3)
    
    history_text = ""
    if state["conversation_history"]:
        history_text = "\n".join([
            f"{'Customer' if m['role'] == 'user' else 'Agent'}: {m['content']}"
            for m in state["conversation_history"][-4:]
        ])
    
    if state["decision"] == "resolve":
        prompt = f"""You are a customer service agent for {state["brand_name"]}, part of RetailCore Group.
Your goal is genuine resolution — not policy deflection.

Brand policy context:
{state["retrieved_context"]}

Conversation history:
{history_text}

Customer message: {state["customer_message"]}

Provide a helpful, specific response that actually resolves their issue.
Be warm, direct, and clear. If policy allows resolution, resolve it.
Do not loop the customer or ask unnecessary questions."""

    elif state["decision"] == "clarify":
        prompt = f"""You are a customer service agent for {state["brand_name"]}.

Conversation history:
{history_text}

Latest customer message: {state["customer_message"]}

You need one specific piece of information to help this customer.
Ask exactly one clear question. Be brief and friendly."""

    else:  # escalate
        prompt = f"""You are a customer service agent for {state["brand_name"]}.

Customer message: {state["customer_message"]}
Conversation history:
{history_text}

This customer needs to speak with a human representative.
Acknowledge their frustration genuinely. Tell them a human will follow up.
Provide the direct contact: call 1-800-RETAILCORE or expect a callback within 10 minutes during business hours.
Do NOT loop them back to automated options."""

    response = llm.invoke(prompt)
    state["agent_response"] = response.content
    state["exchange_count"] = state.get("exchange_count", 0) + 1
    return state


def generate_case_summary(state: AgentState) -> AgentState:
    """When escalating, write a summary for the human agent."""
    if state["decision"] != "escalate":
        state["case_summary"] = ""
        return state
    
    llm = ChatAnthropic(model=MODEL, api_key=api_key, temperature=0)
    
    history_text = "\n".join([
        f"{'Customer' if m['role'] == 'user' else 'Agent'}: {m['content']}"
        for m in state["conversation_history"]
    ]) if state["conversation_history"] else "No prior history"
    
    summary_prompt = f"""Write a concise case summary for a human customer service agent at {state["brand_name"]}.

Conversation history:
{history_text}

Latest customer message: {state["customer_message"]}

Include:
- Issue summary (1-2 sentences)
- What was already attempted
- Recommended next action
- Urgency level (Low / Medium / High)

Keep it under 150 words. This is for internal use only."""

    response = llm.invoke(summary_prompt)
    state["case_summary"] = response.content
    return state


# ── Route decision ─────────────────────────────────────────────────────────────
def route_decision(state: AgentState) -> str:
    return state["decision"]


# ── Build the LangGraph ────────────────────────────────────────────────────────
def build_agent():
    graph = StateGraph(AgentState)
    
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("classify", classify_intent)
    graph.add_node("respond", generate_response)
    graph.add_node("summarize", generate_case_summary)
    
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "classify")
    graph.add_conditional_edges(
        "classify",
        route_decision,
        {
            "resolve": "respond",
            "clarify": "respond",
            "escalate": "respond"
        }
    )
    graph.add_edge("respond", "summarize")
    graph.add_edge("summarize", END)
    
    return graph.compile()


agent = build_agent()


# ── Public interface ──────────────────────────────────────────────────────────
def process_message(
    brand_id: str,
    customer_message: str,
    conversation_history: list,
    frustration_count: int = 0,
    exchange_count: int = 0
) -> dict:
    """Main entry point called by the API."""
    
    if brand_id not in BRANDS:
        return {"error": f"Unknown brand: {brand_id}"}
    
    initial_state = AgentState(
        brand_id=brand_id,
        brand_name=BRANDS[brand_id]["name"],
        customer_message=customer_message,
        conversation_history=conversation_history,
        retrieved_context="",
        agent_response="",
        decision="resolve",
        frustration_count=frustration_count,
        exchange_count=exchange_count,
        case_summary=""
    )
    
    result = agent.invoke(initial_state)
    
    # Log the interaction
    log_interaction(brand_id, customer_message, result)
    
    return {
        "brand_name": BRANDS[brand_id]["name"],
        "decision": result["decision"],
        "response": result["agent_response"],
        "case_summary": result["case_summary"],
        "exchange_count": result["exchange_count"]
    }


def log_interaction(brand_id: str, message: str, result: dict):
    """Log every interaction for compliance and QA."""
    base_dir = os.path.dirname(__file__)
    log_file = os.path.join(base_dir, "logs", f"{brand_id}_interactions.json")
    
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "brand": BRANDS[brand_id]["name"],
        "customer_message": message,
        "decision": result["decision"],
        "response": result["agent_response"],
        "exchange_count": result["exchange_count"]
    }
    
    logs = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            try:
                logs = json.load(f)
            except:
                logs = []
    
    logs.append(entry)
    
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)