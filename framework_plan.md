# 4-Week LangChain & LangGraph Mastery Sprint

**Goal:** Master the modern AI engineering stack (LangChain, LCEL, LangGraph, LangSmith) by building a highly advanced, production-ready Agentic RAG system. This sprint is designed to force deep learning of the framework internals rather than just relying on AI to write the code. You will build a system that stands out in job applications by demonstrating knowledge of complex state management, agentic routing, and robust engineering practices.

**Rule for the whole month:** Every day ends with a commit and a dev-log entry. *Crucially*, for this sprint, you must write out the code yourself and heavily comment *what* each framework abstraction (like a Runnable or a StateGraph node) is doing under the hood. Avoid using AI to write the core logic; use AI strictly for debugging or boilerplate (like Dockerfiles).

---

## WEEK 1 — Framework Fundamentals & LCEL
*Goal: Deconstruct LangChain and LangGraph to understand their core building blocks. Rebuild your raw pipeline using these native framework abstractions.*

**Day 1 — LCEL (LangChain Expression Language) Deep Dive**
- Read the LangChain documentation on LCEL and Runnables.
- Re-write your basic prompt-to-LLM call using pure LCEL (e.g., `prompt | model | output_parser`).
- Understand `RunnablePassthrough`, `RunnableParallel`, and how data flows through the chain.

**Day 2 — Intro to LangGraph: State & Nodes**
- Study LangGraph's core concept: treating LLM applications as state machines.
- Define a `TypedDict` for your application state.
- Create a simple, linear graph (Node A -> Node B) without any LLMs, just modifying state to understand the mechanics.

**Day 3 — Conditional Edges & Routing**
- Implement your first router using conditional edges in LangGraph.
- Write a node that decides whether to "Search Database", "Search Web", or "Direct Answer" based on the user's query.

**Day 4 — Memory & Checkpointing**
- Add a `MemorySaver` checkpointer to your graph.
- Test thread-level memory: send a message, then send a follow-up message in the same thread ID to verify the agent remembers context.

**Day 5 — Framework Pipeline Re-implementation**
- Re-implement your Week 1 "raw pipeline" (Ingestion, Chunking, Retrieval, Generation) using LangChain's document loaders, text splitters, vector stores, and LCEL.
- Integrate this LCEL chain as a node within your LangGraph setup.

**Day 6 — Observability with LangSmith**
- Set up a LangSmith account (free tier) and connect it to your app.
- Inspect the traces of your LangGraph execution. Understand exactly how much time each node takes, the tokens used, and the intermediate inputs/outputs. This is a massive flex for interviews.

**Day 7 — Write-up & Buffer**
- Document in your README: The transition from the raw pipeline to LangGraph. What did the framework abstract away? What new complexities did it introduce?

---

## WEEK 2 — Advanced Agentic Patterns
*Goal: Implement state-of-the-art Agentic RAG techniques (like CRAG or Self-RAG) that make your system resilient and highly accurate.*

**Day 1 — Query Translation / Sub-queries**
- Build a node that takes the user's raw query and rewrites it for better retrieval, or splits it into multiple sub-queries.

**Day 2 — Self-Reflective RAG (Grading)**
- Implement a "Document Grader" node. After retrieval, the LLM evaluates if the retrieved documents are actually relevant to the query.
- If irrelevant, route the graph back to rewrite the query and retrieve again.

**Day 3 — Fallbacks & Web Search Integration**
- Implement a fallback mechanism: If the database doesn't have the answer (or documents are graded poorly), route to a Web Search tool (like Tavily or DuckDuckGo) to augment the context.

**Day 4 — Tool Calling (Function Calling)**
- Move away from string parsing and use native LLM Tool Calling (bind_tools in LangChain).
- Give your agent tools: `search_eu_ai_act`, `search_web`, `calculate_compliance_cost`.

**Day 5 — Hallucination Checker & Answer Grader**
- Implement a final node that checks the generated answer against the retrieved context (Faithfulness) and against the original question (Answer Relevance).
- If it fails, regenerate the answer.

**Day 6 — Assembly: The Mega-Graph**
- Combine all patterns from this week into a single, cohesive LangGraph application.
- Visualize the graph and ensure all conditional edges route correctly.

**Day 7 — Buffer + Eval Run**
- Run your evaluation harness (from your previous work) on this new Agentic system. Did the Self-RAG loop improve precision/recall/faithfulness? Document the metrics!

---

## WEEK 3 — Productionizing the Agent
*Goal: Make the agent robust, secure, and interactive—features that separate toy projects from production engineering.*

**Day 1 — Streaming (Tokens and State)**
- Implement token-by-token streaming so the user sees the answer being typed out.
- Implement state streaming to yield updates like *"Searching database..."*, *"Grading documents..."*, *"Generating answer..."*.

**Day 2 — Human-in-the-Loop (Interrupts)**
- Add an interrupt before a sensitive action (e.g., before the agent finalized a legal summary).
- Write a script that pauses the graph, waits for human approval (or modification of the state), and then resumes.

**Day 3 — Time Travel (State Replay)**
- Use LangGraph's checkpointer to fetch a past state, modify it, and replay the graph from that exact point. (Excellent for debugging and a great talking point).

**Day 4 — Advanced Retrieval Techniques**
- Upgrade your LangChain vector store integration to use Parent Document Retriever (retrieve small chunks for accuracy, but pass large chunks to the LLM for context) or Multi-Query Retriever.

**Day 5 — Guardrails & Security**
- Implement a prompt injection detection node at the very beginning of your graph.
- Add output guardrails to ensure the agent doesn't use inappropriate language or confidently give illegal advice.

**Day 6 — Error Handling & Retries**
- Implement API retry logic using Langchain's built-in retry mechanisms.
- Handle edge cases: Context window overflow, rate limits, and empty API responses.

**Day 7 — Buffer**
- Catch up, clean up code, refactor large nodes into smaller, testable functions.

---

## WEEK 4 — Deployment, UI & Showcase
*Goal: Put a polished face on your complex backend and present it to the world.*

**Day 1 — LangServe / API API Endpoint**
- Wrap your LangGraph agent in a FastAPI application (or use LangServe).
- Expose endpoints for `/invoke`, `/stream`, and `/state`.

**Day 2 — Interactive UI (Streamlit/Gradio)**
- Build a frontend that doesn't just show the chat, but actually visualizes the agent's thought process (e.g., showing the intermediate steps like "Retrieving...", "Grading...", etc., which you enabled in Week 3 Day 1).

**Day 3 — Containerization**
- Write a clean `Dockerfile` and `docker-compose.yml` to spin up your application, vector store (e.g., Chroma/Qdrant), and UI together.

**Day 4 — CI/CD Pipeline**
- Create GitHub actions to run your tests, lint your code (using `ruff` or `flake8`), and verify your graph compiles correctly on push.

**Day 5 — Cloud Deployment**
- Deploy your dockerized application to a free/cheap tier provider (Render, Railway, Fly.io, or AWS EC2).

**Day 6 — The "Outshine" Portfolio Write-up**
- Write a spectacular README or a Medium/Dev.to blog post.
- Include: An architecture diagram of your LangGraph, traces from LangSmith showing complex routing, your evaluation metrics (Before vs. After Agentic patterns), and a section on "What I learned about State Management in LLMs."

**Day 7 — Visibility & Networking**
- Update your resume to feature: "Agentic Engineering", "LangGraph", "LangSmith Observability", "State Machine Architectures".
- Post a video walkthrough of your Streamlit app and the LangSmith trace side-by-side on LinkedIn.

---

## Rules for Learning (No Shortcuts)
- **Do not copy-paste complete Agent architectures.** When you see an example of "Self-RAG" in the LangChain docs, type it out line by line and comment what each parameter does.
- **Understand the Data Flow:** Always print or log the `state` at the beginning and end of every node. If you don't know exactly what the dictionary looks like at step 3, you are flying blind.
- **Read the Source Code:** LangChain can be "magic". When you use `create_retrieval_chain`, Cmd+Click into it in your IDE and read how they implemented it under the hood. This builds true mastery.
