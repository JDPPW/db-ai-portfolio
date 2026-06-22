"""
Enterprise Intelligence Pipeline — Agent Definitions
Author: Dujuan Brown | Iron Gate Solutions US LLC
GitHub: github.com/JDPPW/db-ai-portfolio

Three specialized CrewAI agents, each with one mission.
Orchestrated by LangGraph in pipeline.py.

One agent. One mission. No overlap.
— Production AI principle (Dr. Maryam Miradi)

Target roles: Kyndryl, Trilagen, Panasonic, GreenLight
IBM RAG & Agentic AI Certificate — Course 8 (CrewAI)
"""

import os
from dotenv import load_dotenv
from crewai import Agent, LLM

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")

# ── LLM Configuration ─────────────────────────────────────────────────────────
# CrewAI uses its own LLM wrapper
llm = LLM(
    model="claude-haiku-4-5-20251001",
    api_key=api_key
)

# ── Agent 1: Research Agent ───────────────────────────────────────────────────
# ONE MISSION: Find and gather raw information on the topic
# Does NOT analyze, does NOT write reports
# Just researches and returns structured findings

research_agent = Agent(
    role="Senior Research Specialist",
    goal="Gather comprehensive, factual information about the given topic from your knowledge base. Focus on recent developments, key players, market dynamics, and critical facts.",
    backstory="""You are a senior research specialist with expertise in 
    gathering intelligence across industries. You have a systematic approach 
    to research — you identify the most important aspects of any topic and 
    compile structured findings. You stick strictly to facts and clearly note 
    when information may be outdated or uncertain. You never analyze or 
    editorialize — that is someone else's job. Your output is always 
    structured, comprehensive, and clearly organized.""",
    llm=llm,
    verbose=True,
    allow_delegation=False  # This agent does not hand off — it completes its mission
)

# ── Agent 2: Analysis Agent ───────────────────────────────────────────────────
# ONE MISSION: Extract insights and patterns from research findings
# Does NOT gather new information, does NOT write the final report
# Just analyzes what the Research Agent found

analysis_agent = Agent(
    role="Strategic Intelligence Analyst",
    goal="Analyze the research findings to identify patterns, opportunities, risks, competitive dynamics, and strategic insights. Transform raw research into actionable intelligence.",
    backstory="""You are a strategic intelligence analyst who specializes in 
    turning raw research into meaningful insights. You never gather new 
    information — you work exclusively with what the Research Specialist 
    provides. You excel at identifying patterns others miss, spotting risks 
    before they materialize, and finding opportunities hidden in data. Your 
    analysis is always grounded in the evidence provided, never speculative. 
    You think like a McKinsey consultant — structured, evidence-based, 
    and focused on what decision-makers actually need to know.""",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# ── Agent 3: Report Agent ─────────────────────────────────────────────────────
# ONE MISSION: Produce a polished, structured intelligence report
# Does NOT gather new information, does NOT re-analyze
# Just formats and delivers the final deliverable

report_agent = Agent(
    role="Intelligence Report Writer",
    goal="Transform research findings and strategic analysis into a polished, professional intelligence report that executives and decision-makers can act on immediately.",
    backstory="""You are a professional intelligence report writer who produces 
    executive-grade deliverables. You never conduct research or analysis 
    yourself — you take the Research Specialist's findings and the Analyst's 
    insights and craft them into a clear, structured report. Your reports 
    always include an executive summary, key findings, strategic implications, 
    and recommended actions. You write for busy executives who need to 
    understand the situation and make decisions quickly. Your output is always 
    professional, well-structured, and immediately actionable.""",
    llm=llm,
    verbose=True,
    allow_delegation=False
)