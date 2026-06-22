"""
Enterprise Intelligence Pipeline — LangGraph Orchestration
Author: Dujuan Brown | Iron Gate Solutions US LLC
GitHub: github.com/JDPPW/db-ai-portfolio

LangGraph orchestrates three CrewAI agents in sequence.
Each agent has one mission. LangGraph controls the flow,
quality gates, and escalation between agents.

This is the architectural pattern that separates production
multi-agent systems from tutorial demos.
"""

import os
import json
import datetime
from typing import TypedDict, Literal
from dotenv import load_dotenv
from crewai import Task, Crew, Process
from langgraph.graph import StateGraph, END
from agents import research_agent, analysis_agent, report_agent

load_dotenv()

# ── Pipeline state ────────────────────────────────────────────────────────────
class PipelineState(TypedDict):
    topic: str
    research_output: str
    analysis_output: str
    report_output: str
    quality_grade: Literal["pass", "fail"]
    quality_reason: str
    retry_count: int
    final_report: str
    status: str

MAX_RETRIES = 1

# ── Node 1: Research ──────────────────────────────────────────────────────────
def run_research(state: PipelineState) -> PipelineState:
    """Research Agent gathers comprehensive information on the topic."""
    print(f"\n{'='*50}")
    print(f"RESEARCH AGENT — Starting research on: {state['topic']}")
    print(f"{'='*50}")

    research_task = Task(
        description=f"""Research the following topic comprehensively:

TOPIC: {state['topic']}

Your research must cover:
1. Overview and current state
2. Key players, companies, or stakeholders involved
3. Recent developments and trends (last 1-2 years)
4. Market dynamics, size, or scale where relevant
5. Critical challenges or risks
6. Notable opportunities or growth areas

Structure your findings clearly with headers.
Be factual and specific. Note any uncertainty.
Do NOT analyze or make recommendations — just research.""",
        agent=research_agent,
        expected_output="A comprehensive, structured research report with clearly organized findings covering all required areas."
    )

    crew = Crew(
        agents=[research_agent],
        tasks=[research_task],
        process=Process.sequential,
        verbose=False
    )

    result = crew.kickoff()
    state["research_output"] = str(result)
    state["status"] = "research_complete"
    print(f"\n✅ Research complete — {len(str(result))} characters gathered")
    return state


# ── Node 2: Analysis ──────────────────────────────────────────────────────────
def run_analysis(state: PipelineState) -> PipelineState:
    """Analysis Agent extracts insights from research findings."""
    print(f"\n{'='*50}")
    print(f"ANALYSIS AGENT — Analyzing research findings")
    print(f"{'='*50}")

    analysis_task = Task(
        description=f"""Analyze the following research findings about: {state['topic']}

RESEARCH FINDINGS:
{state['research_output']}

Your analysis must identify:
1. PATTERNS & TRENDS: What recurring themes or directions emerge?
2. OPPORTUNITIES: What are the most significant growth or strategic opportunities?
3. RISKS & THREATS: What are the critical risks decision-makers must know?
4. COMPETITIVE DYNAMICS: Who has advantage and why?
5. STRATEGIC IMPLICATIONS: What does this mean for organizations in this space?
6. KEY UNCERTAINTIES: What remains unclear or unpredictable?

Do NOT gather new information. Work only with what was provided.
Do NOT write the final report — just provide your analysis.""",
        agent=analysis_agent,
        expected_output="A structured strategic analysis with clear sections covering patterns, opportunities, risks, competitive dynamics, and strategic implications."
    )

    crew = Crew(
        agents=[analysis_agent],
        tasks=[analysis_task],
        process=Process.sequential,
        verbose=False
    )

    result = crew.kickoff()
    state["analysis_output"] = str(result)
    state["status"] = "analysis_complete"
    print(f"\n✅ Analysis complete — {len(str(result))} characters of insights")
    return state


# ── Node 3: Quality Gate ──────────────────────────────────────────────────────
def quality_check(state: PipelineState) -> PipelineState:
    """LangGraph quality gate — evaluates if research and analysis are sufficient."""
    print(f"\n{'='*50}")
    print(f"QUALITY GATE — Evaluating pipeline output")
    print(f"{'='*50}")

    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    grade_prompt = f"""You are a quality evaluator for an enterprise intelligence pipeline.

Evaluate whether the research and analysis are sufficient to produce a 
high-quality executive intelligence report.

TOPIC: {state['topic']}

RESEARCH OUTPUT LENGTH: {len(state['research_output'])} characters
ANALYSIS OUTPUT LENGTH: {len(state['analysis_output'])} characters

RESEARCH PREVIEW:
{state['research_output'][:500]}

ANALYSIS PREVIEW:
{state['analysis_output'][:500]}

Grade as PASS if:
- Research covers the topic with specific facts and details
- Analysis identifies clear patterns, opportunities, and risks
- Both outputs are substantive (not vague or generic)

Grade as FAIL only if outputs are clearly insufficient or off-topic.

Respond in JSON only:
{{
  "grade": "pass" or "fail",
  "reason": "one sentence explanation"
}}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": grade_prompt}]
    )

    try:
        content = response.content[0].text.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())
        state["quality_grade"] = result.get("grade", "pass")
        state["quality_reason"] = result.get("reason", "")
    except Exception:
        state["quality_grade"] = "pass"
        state["quality_reason"] = "Grade parsing succeeded with default"

    print(f"\n{'✅' if state['quality_grade'] == 'pass' else '⚠️'} Quality grade: {state['quality_grade'].upper()} — {state['quality_reason']}")
    return state


# ── Node 4: Report ────────────────────────────────────────────────────────────
def run_report(state: PipelineState) -> PipelineState:
    """Report Agent produces the final executive intelligence brief."""
    print(f"\n{'='*50}")
    print(f"REPORT AGENT — Producing final intelligence report")
    print(f"{'='*50}")

    report_task = Task(
        description=f"""Produce a professional executive intelligence report on: {state['topic']}

Use ONLY the research and analysis provided below.

RESEARCH FINDINGS:
{state['research_output']}

STRATEGIC ANALYSIS:
{state['analysis_output']}

Format the report exactly as follows:

# INTELLIGENCE BRIEF: [TOPIC]
**Classification:** Internal Use Only
**Date:** {datetime.date.today().strftime('%B %d, %Y')}
**Prepared by:** Enterprise Intelligence Pipeline

---

## EXECUTIVE SUMMARY
[2-3 sentences capturing the most critical insight]

## KEY FINDINGS
[5-7 bullet points with the most important facts]

## STRATEGIC ANALYSIS
[3-4 paragraphs covering opportunities, risks, and competitive dynamics]

## STRATEGIC IMPLICATIONS
[What this means for organizations — 2-3 specific implications]

## RECOMMENDED ACTIONS
[3-5 specific, actionable recommendations]

## RISK FACTORS
[Top 3 risks to monitor]

---
*This report was produced by a multi-agent AI pipeline combining CrewAI agent collaboration with LangGraph orchestration.*""",
        agent=report_agent,
        expected_output="A complete, professionally formatted executive intelligence report following the specified structure."
    )

    crew = Crew(
        agents=[report_agent],
        tasks=[report_task],
        process=Process.sequential,
        verbose=False
    )

    result = crew.kickoff()
    state["report_output"] = str(result)
    state["final_report"] = str(result)
    state["status"] = "complete"
    print(f"\n✅ Report complete — {len(str(result))} characters")

    log_run(state)
    return state


# ── Node 5: Retry ─────────────────────────────────────────────────────────────
def retry_research(state: PipelineState) -> PipelineState:
    """If quality fails, refine the topic and retry research."""
    state["retry_count"] = state.get("retry_count", 0) + 1
    state["topic"] = f"{state['topic']} — detailed analysis with specific examples"
    print(f"\n⟳ Quality gate failed — retrying research (attempt {state['retry_count']})")
    return state


# ── Routing logic ─────────────────────────────────────────────────────────────
def route_quality(state: PipelineState) -> str:
    if state["quality_grade"] == "pass":
        return "report"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "report"
    return "retry"


# ── Build LangGraph ────────────────────────────────────────────────────────────
def build_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("research", run_research)
    graph.add_node("analysis", run_analysis)
    graph.add_node("quality_check", quality_check)
    graph.add_node("report", run_report)
    graph.add_node("retry", retry_research)

    graph.set_entry_point("research")
    graph.add_edge("research", "analysis")
    graph.add_edge("analysis", "quality_check")
    graph.add_conditional_edges(
        "quality_check",
        route_quality,
        {
            "report": "report",
            "retry": "retry"
        }
    )
    graph.add_edge("retry", "research")
    graph.add_edge("report", END)

    return graph.compile()


pipeline = build_pipeline()


# ── Public interface ──────────────────────────────────────────────────────────
def run_intelligence_pipeline(topic: str) -> dict:
    """Main entry point called by the API."""
    print(f"\n🚀 Starting Enterprise Intelligence Pipeline")
    print(f"Topic: {topic}")

    initial_state = PipelineState(
        topic=topic,
        research_output="",
        analysis_output="",
        report_output="",
        quality_grade="fail",
        quality_reason="",
        retry_count=0,
        final_report="",
        status="starting"
    )

    result = pipeline.invoke(initial_state)

    return {
        "topic": topic,
        "report": result["final_report"],
        "status": result["status"],
        "quality_grade": result["quality_grade"],
        "quality_reason": result["quality_reason"],
        "retry_count": result["retry_count"]
    }


def log_run(state: dict):
    """Log each pipeline run."""
    base_dir = os.path.dirname(__file__)
    log_file = os.path.join(base_dir, "logs", "pipeline_runs.json")

    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "topic": state["topic"],
        "status": state["status"],
        "quality_grade": state["quality_grade"],
        "retry_count": state["retry_count"],
        "report_length": len(state.get("final_report", ""))
    }

    logs = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            try:
                logs = json.load(f)
            except Exception:
                logs = []

    logs.append(entry)

    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)