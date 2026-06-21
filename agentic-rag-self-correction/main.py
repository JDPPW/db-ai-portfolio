"""
Agentic RAG with Self-Correction Loop
Author: Dujuan Brown | Iron Gate Solutions US LLC
GitHub: github.com/JDPPW/db-ai-portfolio

Enterprise IT knowledge base agent that evaluates its own answer
quality and loops back with refined queries when confidence is low.

The self-correction loop:
1. Retrieve relevant documentation chunks
2. Generate an answer
3. Grade the answer quality (relevant, grounded, complete)
4. If quality fails — rewrite the query and retrieve again
5. Maximum 3 correction attempts before escalating to human
6. Return final answer with confidence score and sources cited

Target roles: Kyndryl, Trilagen, Panasonic, QTech, Infosys
IBM RAG & Agentic AI Certificate — Courses 4, 7 (Advanced RAG, LangGraph)
"""

import os
import json
import datetime
from typing import TypedDict, Literal, List
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
MODEL = "claude-haiku-4-5-20251001"

# ── Agent state ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    refined_question: str
    retrieved_docs: List[str]
    answer: str
    grade: Literal["pass", "fail", "escalate"]
    grade_reason: str
    correction_count: int
    confidence_score: int
    sources_used: List[str]
    final_answer: str

MAX_CORRECTIONS = 3

# ── Vector store ──────────────────────────────────────────────────────────────
def build_vectorstore() -> Chroma:
    base_dir = os.path.dirname(__file__)
    kb_dir = os.path.join(base_dir, "knowledge_base")

    all_docs = []
    for filename in os.listdir(kb_dir):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(kb_dir, filename))
            all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(all_docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=None,
        collection_name="techcore_kb",
        persist_directory=os.path.join(base_dir, "chroma_db")
    )
    return vectorstore

_vectorstore = None

def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_vectorstore()
    return _vectorstore

# ── Node 1: Retrieve ──────────────────────────────────────────────────────────
def retrieve(state: AgentState) -> AgentState:
    """Retrieve relevant chunks using current question."""
    query = state.get("refined_question") or state["question"]
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)

    state["retrieved_docs"] = [doc.page_content for doc in docs]
    state["sources_used"] = [
        doc.metadata.get("source", "TechCore KB") for doc in docs
    ]
    return state

# ── Node 2: Generate ──────────────────────────────────────────────────────────
def generate(state: AgentState) -> AgentState:
    """Generate an answer from retrieved context."""
    llm = ChatAnthropic(model=MODEL, api_key=api_key, temperature=0.1)

    context = "\n\n---\n\n".join(state["retrieved_docs"])

    prompt = f"""You are a senior IT support specialist at TechCore Enterprise.
Answer the engineer's question using ONLY the documentation provided below.
Be specific — include exact steps, commands, contact information, and 
escalation paths from the documentation.
If the documentation does not contain enough information to fully answer,
say exactly what is missing.

Documentation:
{context}

Engineer's Question: {state["question"]}

Provide a complete, actionable answer:"""

    response = llm.invoke(prompt)
    state["answer"] = response.content
    return state

# ── Node 3: Grade ─────────────────────────────────────────────────────────────
def grade(state: AgentState) -> AgentState:
    """
    Self-correction: evaluate answer quality on three dimensions.
    1. Relevance — does it address the actual question?
    2. Groundedness — is it based on the retrieved documentation?
    3. Completeness — does it include actionable steps?
    """
    llm = ChatAnthropic(model=MODEL, api_key=api_key, temperature=0)

    grade_prompt = f"""You are a quality evaluator for an enterprise IT knowledge base.

Evaluate this answer on three criteria:
1. RELEVANCE: Does it directly address the engineer's question?
2. GROUNDED: Does it contain specific technical details, steps, or contacts?
3. COMPLETE: Can an engineer act on this answer without needing more information?

Engineer's Question: {state["question"]}

Answer to evaluate:
{state["answer"]}

Be generous in your grading. If the answer contains specific technical details,
named systems, contact information, or step-by-step procedures — it passes.
Only fail if the answer is vague, generic, or says "I don't know."

Respond in this exact JSON format:
{{
  "relevance": "pass" or "fail",
  "grounded": "pass" or "fail",
  "complete": "pass" or "fail",
  "confidence_score": number from 0 to 100,
  "reason": "one sentence explaining the grade",
  "overall": "pass" or "fail"
}}

JSON only, no other text:"""

    response = llm.invoke(grade_prompt)

    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        grade_result = json.loads(content.strip())

        state["grade"] = grade_result.get("overall", "fail")
        state["grade_reason"] = grade_result.get("reason", "")
        state["confidence_score"] = grade_result.get("confidence_score", 0)
    except Exception:
        state["grade"] = "fail"
        state["grade_reason"] = "Could not parse grade response"
        state["confidence_score"] = 0

    return state

# ── Node 4: Refine Query ──────────────────────────────────────────────────────
def refine_query(state: AgentState) -> AgentState:
    """When answer fails grade, rewrite the query to retrieve better context."""
    llm = ChatAnthropic(model=MODEL, api_key=api_key, temperature=0.3)

    refine_prompt = f"""The current search query did not retrieve enough information
to fully answer this IT support question.

Original question: {state["question"]}
Current answer quality issue: {state["grade_reason"]}
Correction attempt: {state["correction_count"] + 1} of {MAX_CORRECTIONS}

Rewrite the search query to find more specific and relevant documentation.
Use different technical terms, be more specific about the system or error,
or break the question into a more targeted search.

Return only the new search query, nothing else:"""

    response = llm.invoke(refine_prompt)
    state["refined_question"] = response.content.strip()
    state["correction_count"] = state.get("correction_count", 0) + 1
    return state

# ── Node 5: Finalize ──────────────────────────────────────────────────────────
def finalize(state: AgentState) -> AgentState:
    """Format the final answer with confidence score and metadata."""
    if state["grade"] == "escalate" or state["correction_count"] >= MAX_CORRECTIONS:
        state["final_answer"] = f"""⚠️ ESCALATION REQUIRED

This query exceeded the maximum self-correction attempts ({MAX_CORRECTIONS}).
The knowledge base may not contain sufficient documentation for this issue.

Best available answer (confidence: {state["confidence_score"]}%):
{state["answer"]}

Recommended Actions:
1. Contact your Tier 3 specialist directly
2. Check if documentation needs to be updated in the knowledge base
3. Escalate to: network-oncall@techcore.com or cloud-ops@techcore.com

Query that was searched: {state.get("refined_question") or state["question"]}"""
    else:
        corrections_note = ""
        if state["correction_count"] > 0:
            corrections_note = f"\n*Answer refined after {state['correction_count']} self-correction loop(s)*"

        state["final_answer"] = f"""✅ Answer (Confidence: {state["confidence_score"]}%)
{corrections_note}

{state["answer"]}

---
*Sources: TechCore Enterprise Knowledge Base*
*Grade: {state["grade_reason"]}*"""

    return state

# ── Routing logic ─────────────────────────────────────────────────────────────
def route_after_grade(state: AgentState) -> str:
    if state["grade"] == "pass":
        return "finalize"
    if state["correction_count"] >= MAX_CORRECTIONS:
        state["grade"] = "escalate"
        return "finalize"
    return "refine"

# ── Build LangGraph ────────────────────────────────────────────────────────────
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("grade", grade)
    graph.add_node("refine", refine_query)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "finalize": "finalize",
            "refine": "refine"
        }
    )
    graph.add_edge("refine", "retrieve")
    graph.add_edge("finalize", END)

    return graph.compile()

agent = build_agent()

# ── Public interface ──────────────────────────────────────────────────────────
def process_question(question: str) -> dict:
    """Main entry point called by the API."""
    initial_state = AgentState(
        question=question,
        refined_question="",
        retrieved_docs=[],
        answer="",
        grade="fail",
        grade_reason="",
        correction_count=0,
        confidence_score=0,
        sources_used=[],
        final_answer=""
    )

    result = agent.invoke(initial_state)

    log_query(question, result)

    return {
        "question": question,
        "answer": result["final_answer"],
        "confidence_score": result["confidence_score"],
        "correction_count": result["correction_count"],
        "grade": result["grade"],
        "grade_reason": result["grade_reason"]
    }

def log_query(question: str, result: dict):
    """Log all queries for observability."""
    base_dir = os.path.dirname(__file__)
    log_file = os.path.join(base_dir, "logs", "queries.json")

    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "grade": result["grade"],
        "confidence_score": result["confidence_score"],
        "correction_count": result["correction_count"],
        "grade_reason": result["grade_reason"]
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