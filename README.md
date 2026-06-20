# RAG Clinical Document Intelligence Agent

**Dujuan Brown | Applied AI Engineer | Iron Gate Solutions US LLC**  
**Target Roles: MedOne Systems · VetsEZ · Katalyst Healthcare · QTech**

---

## What This Does

This agent ingests a clinical patient document, embeds it into a ChromaDB 
vector database, and answers natural language queries from authorized clinical 
staff — with responses grounded strictly in the document.

Ask it things like:
- *"What medications is this patient taking?"*
- *"Does this patient have any drug allergies?"*
- *"What is the primary diagnosis?"*

It retrieves the most relevant document chunks using semantic search, injects 
them into Claude's context, and generates answers grounded only in the 
document. It will not hallucinate. If the information isn't in the document, 
it says so.

---

## Why This Matters in Healthcare

In clinical environments, AI that hallucinates is a patient safety risk. 
This agent is designed around a core principle — answers come only from 
retrieved source context, never from model memory. That pattern is directly 
applicable to:

- EHR document query systems (MedOne BOLT platform)
- Clinical trial data extraction (Katalyst)
- Veterans healthcare platforms (VetsEZ / EZLabs)
- Any HIPAA-regulated document workflow

---

## HIPAA Architecture Notes

This project is built with HIPAA-awareness in mind:
- All document access is local — no data leaves your environment to a 
  third-party vector service
- ChromaDB runs locally with a persistent directory
- No PHI is logged or stored outside the local chroma_db folder
- The sample document uses entirely fictional patient data
- Production deployment would add: access controls, audit logging, 
  encryption at rest, and role-based retrieval filtering

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Anthropic Claude (claude-haiku-4-5-20251001) |
| Vector DB | ChromaDB (local persistent) |
| Orchestration | LangChain LCEL |
| Document Loading | LangChain TextLoader |
| Chunking | RecursiveCharacterTextSplitter |
| Embeddings | ChromaDB default (all-MiniLM-L6-v2) |
| Env Management | python-dotenv |

---

## IBM RAG & Agentic AI Certificate Coverage

| Course | Concept Applied |
|---|---|
| Course 2 — Build RAG Applications | End-to-end RAG pipeline, document ingestion |
| Course 3 — Vector Databases for RAG | ChromaDB, embeddings, similarity search |
| Course 4 — Advanced RAG | Retrieval accuracy, grounded generation |

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/JDPPW/db-ai-portfolio.git
cd db-ai-portfolio/rag-clinical-document-agent

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Add your API key
echo "ANTHROPIC_API_KEY=your-key-here" > .env

# 4. Run the agent
python3 main.py
```

---

## Example Output

See `example_outputs/sample_session.txt` for a full session transcript.

**Query:** What allergies does the patient have?

**Answer:** The patient has two documented allergies:
1. Penicillin — causes rash and hives
2. Sulfa drugs — causes anaphylaxis (patient carries an EpiPen)

---

## Author

**Dujuan Brown**  
Applied AI Engineer | RAG, Agentic Systems & AI Integration  
[LinkedIn](https://linkedin.com/in/dujuan-brown) | 
[GitHub](https://github.com/JDPPW) | dujuanbrown@gmail.com  
Corp-to-Corp available via Iron Gate Solutions US LLC