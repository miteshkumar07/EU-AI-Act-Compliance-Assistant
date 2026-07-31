# 4-Week Applied LLM/RAG Engineering Sprint
**Goal:** A deployed, evaluated, agentic RAG system over EU AI Act / regulatory text (or arXiv ML papers as a substitute domain), built from raw components first, then wrapped in a framework — with a rigorous evaluation harness as the centerpiece.

**Rule for the whole month:** every day ends with something committed to GitHub and a short note in your README/dev-log about *why* you made the decisions you made that day. Interviewers will ask "why," not "what" — this log is your answer bank.

---

## WEEK 1 — Raw pipeline, no framework
*Goal: end the week with a working, ugly, fully-understood RAG pipeline you built without any orchestration framework.*

**Day 1 — Setup + fundamentals**
- Set up repo, virtualenv, GitHub Actions skeleton (empty for now)
- Hugging Face NLP course: tokenization + embeddings chapters only (skip the rest for now)
- Decide corpus: EU AI Act text + 2-3 related guidance docs (or arXiv papers in a domain you like)
- Download and do a first messy look at the data — note the structural challenges (nested articles/annexes, cross-references, footnotes) since this becomes your chunking design rationale

**Day 2 — Transformer/embedding fundamentals continued**
- Finish the HF NLP course sections on attention basics — enough to explain what an embedding model is actually doing, not just call it
- Pick an embedding model (e.g. a solid open sentence-embedding model) and justify the choice in 2-3 sentences in your README
- Write a tiny script that embeds 5 sample paragraphs and inspects similarity — sanity check before building the real pipeline

**Day 3 — Ingestion + chunking (attempt 1)**
- Parse your corpus into clean text
- Implement chunking strategy #1: fixed-size with overlap (simplest baseline)
- Implement chunking strategy #2: structure-aware (split on articles/sections) — this is the one that will likely win, but you need the baseline to prove it
- Store both chunk sets separately (you'll compare them in Week 2)

**Day 4 — Vector store + raw retrieval**
- Set up Chroma locally
- Embed and store both chunk sets as separate collections
- Write a raw retrieval function: query → embed → top-k similarity search → return chunks
- Manually inspect retrieval quality on 5-10 hand-picked questions — does it pull the right passage?

**Day 5 — Raw generation, end-to-end**
- Wire up a direct LLM API call: retrieved chunks + question → prompt → answer
- No framework — write the prompt template yourself, understand exactly what's going into the context window
- Get a full question → retrieved context → answer flow working end-to-end, even if rough

**Day 6 — Hardening the raw pipeline**
- Add basic error handling (empty retrieval, API failures, empty corpus edge cases)
- Add simple logging: what was retrieved, what was sent to the LLM, what came back — you'll want this for evaluation next week
- Try 10-15 more questions manually, note failure patterns (irrelevant retrieval, hallucinated answers, missed cross-references)

**Day 7 — Write-up + buffer**
- Document in README: architecture so far, the two chunking strategies, known failure modes observed
- Buffer day for whatever slipped — something always does in week 1

---

## WEEK 2 — Evaluation harness (your main differentiator)
*Goal: prove, with numbers, which design choices actually work — this is the part almost no other junior portfolio has.*

**Day 1 — Build the ground-truth eval set (part 1)**
- Generate 40-50 candidate question/answer pairs using an LLM prompted against your corpus
- Include a mix: simple factual lookups, multi-hop questions (need 2+ chunks), and a few "should refuse to answer" questions (not covered by the corpus at all)

**Day 2 — Ground-truth eval set (part 2) — manual verification**
- Go through every candidate pair by hand: correct the answer, confirm the actual source passage, discard bad ones
- This manual pass is what makes your eval trustworthy — do not skip it or rely purely on LLM-generated ground truth
- End with a clean eval set: question, correct answer, correct source chunk(s)

**Day 3 — Retrieval metrics**
- Implement retrieval precision/recall against your eval set's known source chunks
- Run it against both chunking strategies from Week 1 — get your first real comparison numbers

**Day 4 — Answer-quality metrics**
- Implement faithfulness/answer-quality scoring (RAGAS or a comparable framework, or a simple LLM-as-judge scorer you write yourself and can explain)
- Add latency and approximate cost-per-query tracking (token counts × pricing)

**Day 5 — Run the full comparison**
- Run all variants (2 chunking strategies × maybe 2 embedding models if time allows) through the full harness
- Build a results table: precision, recall, faithfulness, latency, cost per variant

**Day 6 — Pick a winner, document why**
- Choose your production configuration based on the numbers, not intuition
- Write the comparison + reasoning into the README — this becomes a genuine "here's how I made an engineering decision" story

**Day 7 — Buffer + polish**
- Clean up the eval code so it's re-runnable (you'll use it again in Week 3 to test the framework version)
- Buffer day

---

## WEEK 3 — Framework + agentic behavior
*Goal: add LangChain/LangGraph and real agent behavior on top of a pipeline you already understand from first principles.*

**Day 1 — Reimplement in LangChain/LangGraph**
- Rebuild the winning pipeline from Week 2 using LangChain or LangGraph
- Re-run your eval harness against this version — confirm it matches your raw implementation's numbers (if it doesn't, figure out why — great debugging story)

**Day 2 — Note the tradeoffs**
- Document explicitly: what the framework saved you in code, what it cost you in transparency/debuggability
- This is a strong interview answer waiting to be written

**Day 3 — Agentic capability #1: retrieval routing**
- Add logic so the agent decides whether a query needs retrieval at all (simple greeting vs. real question) or needs multi-hop retrieval (complex, multi-part questions)

**Day 4 — Agentic capability #2: citations**
- Make the agent cite specific source passages/articles in its answers, not just produce free text
- Test this against your eval set — does citation accuracy hold up?

**Day 5 — Robustness / red-team pass**
- Test your pipeline against a handful of public prompt-injection examples (e.g. a hidden instruction embedded in a retrieved document trying to override the system prompt)
- Document what breaks, and outline (doesn't need to be fully built) how you'd mitigate it — this alone puts you ahead of most junior candidates

**Day 6 — Fine-tuning experiment (optional if time allows)**
- If time permits: LoRA fine-tune a small open model on your eval Q&A pairs, compare against your prompted pipeline on cost/quality
- If you're behind schedule, cut this first — it's the most skippable piece this month, you can revisit post-thesis

**Day 7 — Buffer**
- Catch up on whatever slipped; this week has the most moving parts

---

## WEEK 4 — Deployment + packaging
*Goal: ship it, document it properly, and make it visible.*

**Day 1 — API wrapper**
- Wrap the pipeline in FastAPI: an endpoint that takes a question, returns an answer + sources + confidence/metrics
- Use Claude Code here for boilerplate — this is exactly the kind of infrastructure code where AI assistance is appropriate

**Day 2 — Containerize**
- Dockerfile, get it running locally in a container
- Again, fine to lean on Claude Code for standard Docker setup

**Day 3 — CI + deploy**
- Basic GitHub Actions: run your eval suite on push (this is a nice touch — "CI that checks RAG quality, not just tests passing")
- Deploy to a free tier (Render or Fly.io)

**Day 4 — n8n workflow (light touch)**
- One small automation: e.g. scheduled re-ingestion when the corpus updates, or a workflow that runs your eval suite and posts a summary somewhere (Slack/email/webhook)
- Keep this small — it's a garnish, not the main dish

**Day 5-6 — The README that actually gets read**
- Architecture diagram (a simple one is fine)
- Your eval results table front and center
- The design decisions and tradeoffs you made, in your own words
- What you'd do differently with more time (shows judgment, not just execution)
- Clear setup/run instructions

**Day 7 — Visibility**
- Update CV: add this project at the top, add the relevant keywords (RAG, LangGraph, evaluation, LLM, Docker, FastAPI)
- Update LinkedIn, post a short write-up of what you built and what you learned — public reasoning is itself a signal recruiters notice

---

## Guardrails for the month
- **Claude Code:** fine for Docker/FastAPI/CI/tests/debugging. Not for chunking logic, eval harness, or agent decision logic in Weeks 1-3 — you need to defend those line-by-line.
- **If you fall behind:** cut Week 3 Day 6 (fine-tuning) first, then reduce Week 3's agentic scope to just retrieval routing (drop citations) before cutting anything from Week 2 — the eval harness is the least skippable part of this whole project.
- **Daily minimum:** one commit, one line in the dev-log explaining a decision made that day.