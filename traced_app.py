# Week 2 Day 3: trace a LangChain app with LangSmith.
# Tracing is switched on ONLY by environment variables (LANGSMITH_TRACING,
# LANGSMITH_API_KEY, LANGSMITH_PROJECT). Nothing in this file talks to LangSmith.

import os
import time

from dotenv import load_dotenv

# Load keys from .env into the environment before LangChain is used.
load_dotenv()

from langchain_openai import ChatOpenAI  # noqa: E402

# --- Check tracing is really on (it can be silently off) --------------------
tracing = os.getenv("LANGSMITH_TRACING", "").strip().lower()
project = os.getenv("LANGSMITH_PROJECT", "").strip()
has_key = bool(os.getenv("LANGSMITH_API_KEY", "").strip())

print(f"LANGSMITH_TRACING = {tracing or '(not set)'}")
print(f"LANGSMITH_PROJECT = {project or '(not set -> runs go to \"default\")'}")
print(f"LANGSMITH_API_KEY set = {has_key}")
if tracing != "true" or not has_key or not project:
    print("WARNING: tracing looks OFF. Runs will not appear in LangSmith.\n")
else:
    print("Tracing looks ON.\n")

# --- The app ----------------------------------------------------------------
llm = ChatOpenAI(model="gpt-5.4-mini")

prompts = [
    "Say hello in one word.",
    "Explain what LangSmith is in one simple sentence.",
    "Explain how Retrieval Augmented Generation works, step by step, "
    "with a short example, and list its main pros and cons.",
]

results = []
for i, p in enumerate(prompts, start=1):
    start = time.perf_counter()
    reply = llm.invoke(p)
    seconds = time.perf_counter() - start
    tokens = (reply.usage_metadata or {}).get("total_tokens", 0)
    results.append((i, p, seconds, tokens))

    print(f"--- Prompt {i}: {p}")
    print(reply.content)
    print()

# --- Summary to compare with the LangSmith dashboard ------------------------
print(f"Project: {project or 'default'}")
most = max(r[3] for r in results)
for i, p, seconds, tokens in results:
    short = p if len(p) <= 40 else p[:37] + "..."
    flag = "  <- the expensive one" if tokens == most else ""
    print(f'run {i} "{short}" {seconds:.1f}s {tokens:,} tokens{flag}')
