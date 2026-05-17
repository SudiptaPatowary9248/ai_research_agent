# AI Research Agent with ReAct, Tool Routing, and Conversational Memory

An AI agent built using FastAPI and HuggingFace models that performs dynamic tool selection, multi-step reasoning, conversational memory retrieval, and intent-aware response generation.

## Motivation

The goal of this project was to better understand how modern AI agents work internally instead of relying entirely on orchestration frameworks.

The system was built incrementally to explore:
- tool routing
- ReAct-style reasoning
- conversational memory
- prompt engineering
- retrieval-based context selection
- failure modes in small LLMs

## Design Decisions

- The agent is intentionally constrained to a maximum of one search iteration to prevent looping behavior in smaller LLMs.
- Retrieval-based memory selection is used instead of passing full chat history to reduce context drift.
- Rule-based overrides are combined with LLM planning to improve routing reliability.

## Features

- ReAct-style reasoning loop
- LLM-based tool routing
- Math tool integration
- Conversational memory
- Retrieval-based memory selection
- Intent-aware prompting
- Structured logging and error handling
- FastAPI backend API

## Architecture

The system follows a lightweight ReAct-style architecture:

User Query
↓
Planner (LLM)
↓
Tool Selection
↓
Search / Math Tool
↓
Observation
↓
Answer Generation
↓
Memory Storage

## Components

### Planner
Uses an instruction-tuned LLM to decide whether the agent should:
- search for information
- answer directly
- use tools

### Memory System
Stores recent conversational context and retrieves the most relevant base explanation for follow-up queries.

### Intent Detection
Detects whether the user wants:
- simplification
- examples
- summaries
- general answers

### Generator
Uses FLAN-T5 to generate final responses using structured prompts.

## Challenges and Learnings

### Context Drift
Passing full conversation history caused follow-up responses to anchor to previous examples instead of the original concept.

Solution:
Implemented a simple retrieval-based memory selection using `get_base_context()`.

---

### Model-Task Mismatch
A summarization model copied instructions instead of following them.

Solution:
Switched from a summarization pipeline to an instruction-tuned FLAN-T5 model.

---

### Repetition Loops
The model repeatedly generated phrases like "For example".

Solution:
Used few-shot prompting and repetition penalties during generation.

## Example Usage

### Query
/research?query=what%20is%20machine%20learning

### Response
{
  "status": "success",
  "route": "answer",
  "answer": "Machine learning is..."
}

## Current Limitations

- Small instruction-tuned models occasionally produce weak or repetitive examples.
- Memory is stored in-process and is not persistent across sessions.
- The planner currently supports only lightweight multi-step reasoning.

## Future Improvements

- Multi-search iterative reasoning
- Vector database memory retrieval
- Better instruction-tuned models
- Query rewriting
- Streaming responses
- Persistent memory storage (Redis / DB)
- Evaluation framework for response quality

## Tech Stack

- Python
- FastAPI
- HuggingFace Transformers
- FLAN-T5
- Uvicorn

## Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Example Agent Flow
User: What is machine learning?
→ Planner decides search is needed
→ Search tool retrieves information
→ Observation stored
→ Generator produces answer
→ Memory updated

User: Explain it simply
→ Planner identifies follow-up query
→ Memory retrieval selects base explanation
→ Generator simplifies response without new search
