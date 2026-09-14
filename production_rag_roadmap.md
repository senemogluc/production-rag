# Production RAG Project Roadmap

## Project Goal

The goal of this project is to build a **production-oriented Retrieval-Augmented Generation (RAG) system** step by step.

The purpose is not only to make a RAG application that works, but to understand:

- how retrieval works,
- why different retrieval strategies are needed,
- how to measure retrieval quality,
- how to separate retrieval problems from generation problems,
- how to expose the system as a real backend service,
- and how to observe, test, and deploy it like a production AI application.

The project deliberately starts simple and becomes more sophisticated only when the previous version exposes a limitation.

The progression is:

```text
V0 — Naive RAG
        ↓
V1 — Better Retrieval
        ↓
V2 — Evaluation-Driven RAG
        ↓
V3 — Production API
        ↓
V4 — Production Engineering
        ↓
V5 — Portfolio Release
```

A possible future extension is:

```text
V6 — Adaptive / Agentic RAG with LangGraph
```

However, LangGraph is intentionally **not** part of the first release because the initial system is mostly deterministic and linear.

---

# V0 — Naive RAG

## What V0 Is

V0 is the smallest complete version of the system.

Its purpose is to prove that the full RAG loop works from end to end:

1. load documents,
2. split them into chunks,
3. convert the chunks into embeddings,
4. store the embeddings in a vector database,
5. embed the user's question,
6. retrieve the most similar chunks,
7. give those chunks to an LLM,
8. generate an answer,
9. return the answer together with its sources.

At this stage, the system does **not** try to be optimal.

There is no hybrid search, reranker, observability platform, PostgreSQL application layer, or agentic workflow yet.

The purpose of V0 is to understand the basic mechanics before adding complexity.

## Architecture

```text
Document
   ↓
Parse
   ↓
Chunk
   ↓
Embed
   ↓
Qdrant

----------------

Question
   ↓
Embed
   ↓
Vector Search
   ↓
Top-K Chunks
   ↓
LLM
   ↓
Answer + Sources
```

## Suggested Stack

- Python
- LangChain
- Qdrant
- sentence-transformers / BGE embeddings
- LLM API or local LLM

## Deliverable

A working script, CLI, or minimal endpoint that can answer questions using the indexed documents.

Example:

```bash
python ask.py "What is Kubernetes HPA?"
```

Expected output:

```text
Answer:
Horizontal Pod Autoscaler automatically adjusts the number of Pods
based on observed resource utilization.

Sources:
- hpa.pdf, page 4
- autoscaling.pdf, page 7
```

## Checklist

- [ ] Select a small document collection
- [ ] Support PDF, Markdown, or TXT ingestion
- [ ] Extract document text
- [ ] Store useful metadata such as document name and page number
- [ ] Split documents into chunks
- [ ] Make chunk size configurable
- [ ] Make chunk overlap configurable
- [ ] Generate embeddings for each chunk
- [ ] Store chunk embeddings in Qdrant
- [ ] Generate an embedding for the user's query
- [ ] Perform Top-K vector similarity search
- [ ] Retrieve the most relevant chunks
- [ ] Insert retrieved context into the LLM prompt
- [ ] Ask the LLM to answer using the retrieved context
- [ ] Return source citations with the answer
- [ ] Manually test several queries

## Why We Are Doing This

### Why chunking?

A large document should not normally be represented by a single embedding.

If an 80-page PDF is converted into one vector, very different topics are compressed into the same representation.

Chunking allows us to retrieve smaller and more specific pieces of information.

### Why embeddings?

Embeddings let us search based on semantic similarity rather than exact keyword overlap.

For example:

```text
"How do I scale pods automatically?"
```

and:

```text
"Horizontal Pod Autoscaler dynamically adjusts replicas."
```

may use different words but express related meaning.

### Why Qdrant?

We need an efficient way to:

- store embeddings,
- search large numbers of vectors,
- persist them,
- and attach metadata to them.

Qdrant provides vector indexing, approximate nearest-neighbor search, persistence, and metadata filtering.

### Why Top-K retrieval?

We do not want to send every stored document to the LLM.

Instead, we retrieve only the most relevant chunks.

Top-K controls how many candidate chunks are passed to the next stage.

## Definition of Done

V0 is complete when:

- documents can be indexed,
- a user can ask a question,
- relevant chunks are retrieved,
- the LLM generates an answer using those chunks,
- and the answer includes source information.

You should also be able to explain, in your own words:

> What happens from the moment a user asks a question until the final answer is returned?

### Estimated effort

**3–5 focused hours**

---

# V1 — Better Retrieval

## What V1 Is

V0 gives us a working RAG pipeline, but it does not tell us whether the retrieved chunks are actually the best possible chunks.

V1 focuses entirely on **retrieval quality**.

The main idea is that semantic vector search is useful, but it is not perfect.

Some queries are semantic:

```text
"How can Kubernetes automatically scale workloads?"
```

Other queries depend heavily on exact terms:

```text
CUDA_ERROR_OUT_OF_MEMORY
```

Therefore, V1 combines:

- dense semantic retrieval,
- sparse lexical retrieval,
- result fusion,
- and reranking.

## Architecture

```text
                    Query
                   /     \
                  ↓       ↓
              Dense       BM25
             Retrieval   Retrieval
                  \       /
                   ↓     ↓
                    Fusion
                      ↓
                  Candidates
                      ↓
                   Reranker
                      ↓
                    Top-K
```

## Deliverable

A system that supports at least three retrieval configurations:

```text
dense
hybrid
hybrid + reranker
```

For example:

```text
RETRIEVAL_MODE=hybrid_reranker
```

## Checklist

- [ ] Keep dense retrieval as the baseline
- [ ] Make chunk size configurable
- [ ] Make chunk overlap configurable
- [ ] Test multiple chunk sizes
- [ ] Add BM25 sparse retrieval
- [ ] Implement hybrid dense + sparse retrieval
- [ ] Add a result-fusion strategy
- [ ] Retrieve a larger candidate set before reranking
- [ ] Add a cross-encoder reranker
- [ ] Rerank the candidate chunks
- [ ] Select the final Top-K chunks after reranking
- [ ] Log retrieval scores
- [ ] Compare the same queries across retrieval modes
- [ ] Test semantic queries
- [ ] Test exact-keyword queries
- [ ] Document retrieval trade-offs

## Why We Are Doing This

### Why BM25?

Dense retrieval is good at semantic similarity.

BM25 is good at lexical matching.

Queries containing exact identifiers, error codes, product names, or technical terminology can benefit from lexical search.

Example:

```text
AWS EC2 error 0x803F7001
```

A keyword-based retriever may identify the exact token more reliably than an embedding model.

### Why hybrid retrieval?

Dense retrieval and sparse retrieval fail in different ways.

Hybrid retrieval combines both signals.

Conceptually:

```text
Dense Retrieval → semantic relevance
Sparse Retrieval → lexical relevance

Dense + Sparse → more robust retrieval
```

### Why a reranker?

An embedding retriever is fast because query and document vectors are usually computed separately.

A reranker evaluates the query and candidate document together.

That makes it more computationally expensive, but often better at fine-grained relevance estimation.

The common pattern is:

```text
Large document collection
        ↓
Fast retriever
        ↓
Top 20 candidates
        ↓
More expensive reranker
        ↓
Top 5 final chunks
```

The retriever aims for **high recall**.

The reranker aims for **better precision and ordering**.

### Why experiment with chunk size?

Chunk size is a retrieval hyperparameter.

Small chunks may be precise but lose context.

Large chunks preserve more context but may contain more irrelevant information.

Example:

```text
Small chunk
+ specific
+ easier to retrieve precisely
- may lose surrounding context

Large chunk
+ preserves context
- may contain more noise
- may produce less specific embeddings
```

We should measure the effect instead of assuming a universal best value.

## Definition of Done

V1 is complete when:

- dense retrieval works,
- BM25 retrieval works,
- hybrid retrieval works,
- reranking works,
- multiple configurations can be compared,
- and retrieval behavior is understandable.

You should be able to explain:

> What is the difference between dense and sparse retrieval?

> Why can hybrid retrieval be more robust?

> What is the difference between a retriever and a reranker?

### Estimated effort

**~1 focused day**

---

# V2 — Evaluation-Driven RAG

## What V2 Is

V1 may feel better than V0, but engineering decisions should not depend only on intuition.

V2 introduces a reproducible evaluation pipeline.

The purpose is to answer questions such as:

- Did hybrid retrieval actually improve retrieval quality?
- Does reranking help?
- Which chunk size performs best?
- Are relevant documents retrieved?
- Are they ranked near the top?
- Does the LLM stay faithful to retrieved evidence?
- Did quality improve at the cost of significantly higher latency?

This version turns the project from a RAG demo into an **experimentally evaluated retrieval system**.

## Deliverable

An evaluation package such as:

```text
evaluation/
├── dataset.json
├── retrieval_eval.py
├── generation_eval.py
└── results.csv
```

And a comparison table such as:

| Configuration | Recall@5 | MRR | nDCG | Faithfulness |
|---|---:|---:|---:|---:|
| Dense | 0.xx | 0.xx | 0.xx | 0.xx |
| Hybrid | 0.xx | 0.xx | 0.xx | 0.xx |
| Hybrid + Reranker | 0.xx | 0.xx | 0.xx | 0.xx |

The values should come from real experiments.

## Checklist

- [ ] Create an evaluation dataset
- [ ] Prepare approximately 30–50 high-quality questions
- [ ] Label the relevant source or chunk for each question
- [ ] Add reference answers where useful
- [ ] Evaluate dense retrieval
- [ ] Evaluate hybrid retrieval
- [ ] Evaluate hybrid + reranker
- [ ] Calculate Recall@K
- [ ] Calculate MRR
- [ ] Calculate nDCG
- [ ] Measure retrieval latency
- [ ] Evaluate answer faithfulness
- [ ] Evaluate answer relevancy
- [ ] Evaluate context precision where useful
- [ ] Evaluate context recall where useful
- [ ] Store results in CSV or JSON
- [ ] Compare retrieval configurations
- [ ] Write a short interpretation of the results
- [ ] Select a default production configuration based on evidence

## Why We Are Doing This

### Why Recall@K?

Recall@K asks:

> Did the retriever include the information we needed within the top K results?

Suppose the required evidence is in `Chunk 174`.

The retriever returns:

```text
1. Chunk 82
2. Chunk 174  ← relevant
3. Chunk 221
4. Chunk 91
5. Chunk 45
```

For this query, the relevant chunk is present in the Top-5.

That is important because if the correct information is not retrieved, the LLM may not have the evidence needed to answer correctly.

Conceptually:

```text
Correct evidence retrieved
        ↓
LLM has a chance to answer correctly
```

but:

```text
Correct evidence not retrieved
        ↓
Generation quality is severely limited
```

This is why recall is extremely important for first-stage retrieval.

### Why MRR?

Recall tells us whether a relevant result appeared.

It does not tell us how highly it was ranked.

Consider two systems:

```text
System A
1. irrelevant
2. relevant
```

```text
System B
1. irrelevant
2. irrelevant
3. irrelevant
4. irrelevant
5. relevant
```

Both may have the same Recall@5.

However, System A ranks the relevant result much higher.

MRR — Mean Reciprocal Rank — rewards earlier relevant results.

```text
Rank 1 → 1 / 1 = 1.0
Rank 2 → 1 / 2 = 0.5
Rank 5 → 1 / 5 = 0.2
```

### Why nDCG?

Some queries can have several relevant documents with different relevance levels.

For example:

```text
Document 1 → highly relevant
Document 2 → relevant
Document 3 → partially relevant
```

nDCG helps evaluate whether the most useful results are ranked near the top.

### Why faithfulness?

Good retrieval does not guarantee good generation.

Example:

```text
Retrieved context:
AWS provides a two-minute interruption warning.

Generated answer:
AWS provides a five-minute interruption warning.
```

The retriever succeeded.

The generator failed.

Faithfulness evaluates whether the answer is supported by the retrieved context.

This is why we must separate:

```text
Retrieval quality
```

from:

```text
Generation quality
```

### Why measure latency?

The best-quality system is not automatically the best production system.

For example:

```text
Configuration A
Recall@5 = 0.88
Latency = 250 ms

Configuration B
Recall@5 = 0.89
Latency = 1800 ms
```

The additional quality may not justify the latency increase.

Production AI engineering involves quality–latency–cost trade-offs.

## Definition of Done

V2 is complete when the question:

> How do you know your RAG system is good?

can be answered with measurements rather than intuition.

A strong answer would be:

> I created a labeled retrieval benchmark and compared dense, hybrid, and reranked retrieval using Recall@K, MRR, and nDCG. I separately evaluated answer faithfulness and relevance so retrieval errors and generation errors could be analyzed independently.

### Estimated effort

**1–1.5 focused days**

---

# V3 — Production API

## What V3 Is

Until this point, the RAG engine may still be running through scripts or notebooks.

V3 turns the retrieval system into a real backend service.

The goal is to separate the RAG engine from the interface that uses it.

A frontend, another backend service, a mobile application, or another AI service should be able to interact with the system through a stable API.

This version also introduces a normal relational database for application-level data.

## Architecture

```text
                    Client
                       │
                    FastAPI
                    /     \
                   ↓       ↓
             PostgreSQL   RAG Engine
                             │
                         Retrieval
                             │
                          Qdrant
                             │
                            LLM
```

## Deliverable

A working REST API with Swagger/OpenAPI documentation.

Suggested endpoints:

```text
POST   /documents
GET    /documents
DELETE /documents/{id}

POST   /query
POST   /feedback

GET    /health
```

Example request:

```json
{
  "question": "How does Kubernetes autoscaling work?",
  "top_k": 5
}
```

Example response:

```json
{
  "answer": "...",
  "sources": [
    {
      "document": "kubernetes.pdf",
      "page": 12,
      "score": 0.91
    }
  ],
  "latency_ms": 840
}
```

## Checklist

- [ ] Create a clean FastAPI project structure
- [ ] Add Pydantic request models
- [ ] Add Pydantic response models
- [ ] Implement document upload
- [ ] Add document-processing status
- [ ] Add query endpoint
- [ ] Return source metadata with answers
- [ ] Add document deletion
- [ ] Add health endpoint
- [ ] Add proper error handling
- [ ] Add configuration management
- [ ] Add environment variables
- [ ] Add PostgreSQL
- [ ] Store document metadata
- [ ] Store ingestion status
- [ ] Store user feedback
- [ ] Optionally store query history
- [ ] Connect PostgreSQL document IDs with Qdrant metadata
- [ ] Add API integration tests

## Why We Are Doing This

### Why FastAPI?

A script is useful for experimentation.

A production system needs a stable interface.

FastAPI turns the RAG engine into a reusable service that other systems can call over HTTP.

Instead of:

```text
python rag.py
```

we now have:

```text
Client → HTTP API → RAG service
```

### Why PostgreSQL if Qdrant already stores data?

Qdrant and PostgreSQL solve different problems.

Qdrant is responsible for retrieval-oriented data:

```text
embedding
chunk text
document_id
page
retrieval metadata
```

PostgreSQL is responsible for application-oriented relational data:

```text
users
documents
ownership
processing status
feedback
timestamps
permissions
```

Example:

```text
PostgreSQL

document_id = 123
filename = kubernetes.pdf
status = READY
owner_id = 55
```

Qdrant:

```text
document_id = 123
page = 15
chunk = "..."
embedding = [...]
```

This creates a cleaner separation of responsibilities.

## Definition of Done

V3 is complete when:

- documents can be uploaded through the API,
- documents can be indexed,
- users can query the RAG system through HTTP,
- responses contain citations,
- metadata persists across restarts,
- and the API has basic integration tests.

You should be able to run:

```text
POST /documents
```

followed by:

```text
POST /query
```

and receive a JSON answer with sources.

### Estimated effort

**0.5–1 focused day**

---

# V4 — Production Engineering

## What V4 Is

V3 works as an application.

V4 focuses on making it easier to operate, debug, test, and deploy.

This stage introduces:

- containerization,
- observability,
- tracing,
- regression testing,
- health checks,
- and CI.

The main question becomes:

> If the system behaves badly in production, can we understand why?

## Deliverable

A system that can be started locally with:

```bash
docker compose up
```

and provides end-to-end traces of the RAG pipeline.

## Checklist

### Containerization

- [ ] Create API Dockerfile
- [ ] Add Qdrant container
- [ ] Add PostgreSQL container
- [ ] Add Docker Compose
- [ ] Add persistent volumes
- [ ] Add environment-variable configuration
- [ ] Add container health checks

### Observability

- [ ] Add Langfuse or another LLM observability platform
- [ ] Trace incoming queries
- [ ] Record retrieved chunks
- [ ] Record retrieval scores
- [ ] Record reranker scores
- [ ] Measure embedding latency
- [ ] Measure retrieval latency
- [ ] Measure reranking latency
- [ ] Measure LLM generation latency
- [ ] Track token usage
- [ ] Track total request latency

### Testing

- [ ] Add unit tests
- [ ] Add integration tests
- [ ] Add API tests
- [ ] Add evaluation regression tests
- [ ] Add GitHub Actions CI
- [ ] Fail CI on major retrieval-quality regressions where appropriate

## Why We Are Doing This

### Why observability?

Suppose a user reports:

> "This answer is wrong."

There are several possible failure points:

```text
Bad query interpretation
        ↓
Wrong retrieved chunks
        ↓
Bad reranking
        ↓
Good context but bad generation
```

Without traces, all of these failures may look identical from the outside.

Observability lets us inspect:

```text
Query
 ↓
Retrieved documents
 ↓
Retrieval scores
 ↓
Reranker scores
 ↓
Prompt
 ↓
LLM answer
 ↓
Latency / token usage
```

This makes production debugging possible.

### Why Docker?

Docker makes the application reproducible.

A new developer should not need to manually install and configure every dependency.

Instead:

```bash
docker compose up
```

should start the application stack.

### Why evaluation regression tests?

Traditional software tests answer:

> Does the code still run?

RAG evaluation should also answer:

> Does the system still retrieve and answer well?

Example:

```text
Before a chunking change:
Recall@5 = 0.88

After the change:
Recall@5 = 0.61
```

The API may still return HTTP 200.

Normal tests may still pass.

But retrieval quality has degraded badly.

That is why AI systems need quality regression testing in addition to normal software tests.

## Definition of Done

V4 is complete when:

- the full system starts with Docker Compose,
- important pipeline stages are observable,
- retrieval and generation traces can be inspected,
- automated tests run in CI,
- and retrieval quality can be checked for regressions.

### Estimated effort

**~1 focused day**

---

# V5 — Portfolio Release

## What V5 Is

V5 does not significantly change the RAG algorithm.

Its purpose is to make the engineering work understandable and impressive to someone who opens the repository for the first time.

A recruiter or engineer is unlikely to read the entire codebase.

The README and project presentation should make the important decisions immediately visible.

## Deliverable

A polished public GitHub repository and, ideally, a deployed demo or API.

## Checklist

### README

- [ ] Project overview
- [ ] Problem statement
- [ ] Architecture diagram
- [ ] Tech stack
- [ ] RAG pipeline explanation
- [ ] Retrieval strategies
- [ ] Evaluation methodology
- [ ] Experimental results
- [ ] Example questions
- [ ] Example answers
- [ ] API documentation
- [ ] Local setup instructions
- [ ] Docker instructions
- [ ] Limitations
- [ ] Future improvements

### Engineering decisions

- [ ] Explain why LangChain was used
- [ ] Explain why LangGraph was not needed yet
- [ ] Explain why Qdrant was selected
- [ ] Explain why BM25 was added
- [ ] Explain why hybrid retrieval was used
- [ ] Explain why reranking was added
- [ ] Explain how retrieval was evaluated
- [ ] Explain how generation was evaluated
- [ ] Explain important quality / latency trade-offs

### Presentation

- [ ] Add benchmark table
- [ ] Add architecture diagram
- [ ] Add sample traces if useful
- [ ] Add demo GIF or short video
- [ ] Clean repository structure
- [ ] Remove dead code
- [ ] Add `.env.example`
- [ ] Add license if appropriate
- [ ] Add CV-ready project bullets
- [ ] Deploy the API or demo if useful

## Why We Are Doing This

A strong project is not only technically good.

It must also communicate the engineering decisions clearly.

The strongest portfolio message is not:

> "I used LangChain, Qdrant, BM25, and FastAPI."

It is:

> "I started with dense retrieval, measured its limitations, added hybrid retrieval and reranking, evaluated the effect using retrieval and generation metrics, exposed the system as a FastAPI service, and added observability and regression testing."

That demonstrates engineering reasoning rather than technology collection.

## Definition of Done

V5 is complete when a technical reviewer can understand within a few minutes:

- what problem the project solves,
- how the architecture works,
- why the selected technologies were used,
- how retrieval quality was measured,
- what improvements were tested,
- and how the system could be operated in production.

### Estimated effort

**0.5–1 focused day**

---

# Why LangChain Instead of LangGraph?

For the first versions, the RAG pipeline is mostly linear:

```text
Query
  ↓
Retrieve
  ↓
Rerank
  ↓
Generate
```

There is no major need for:

- branching,
- loops,
- long-lived state,
- retries controlled by a graph,
- human approval,
- or tool-selection logic.

LangGraph would therefore add orchestration complexity without solving an immediate problem.

LangChain is sufficient for:

- document loaders,
- text splitters,
- embedding integrations,
- retriever abstractions,
- prompt construction,
- model calls,
- and structured pipeline components.

A strong explanation is:

> The initial RAG pipeline is deterministic and mostly linear, so LangGraph would add unnecessary orchestration complexity. LangChain is enough for retrieval and generation primitives. LangGraph becomes useful once the system introduces adaptive routing, retry loops, stateful workflows, or agentic retrieval.

---

# Possible V6 — Adaptive / Agentic RAG

LangGraph becomes more useful if the system evolves into something like:

```text
                    Query
                      ↓
                 Classifier
                 /        \
                ↓          ↓
             Simple      Complex
                            ↓
                        Retrieval
                            ↓
                   Context sufficient?
                     /           \
                   No             Yes
                   ↓               ↓
              Rewrite query     Generate
                   ↓
               Retrieve
                   ↑
                   └──── loop
```

Possible V6 features:

- query routing,
- query rewriting,
- self-correction,
- retrieval retries,
- context-quality checks,
- tool calling,
- human-in-the-loop approval,
- multi-source retrieval,
- stateful workflows.

This would be the point where LangGraph solves a real architectural problem rather than being included only for the sake of using it.

---

# Suggested Timeline

| Version | Main focus | Estimated effort |
|---|---|---:|
| V0 | Working naive RAG | 3–5 hours |
| V1 | Hybrid retrieval + reranking | ~1 day |
| V2 | Retrieval + generation evaluation | 1–1.5 days |
| V3 | FastAPI + PostgreSQL | 0.5–1 day |
| V4 | Docker + observability + testing | ~1 day |
| V5 | README + demo + portfolio polish | 0.5–1 day |

For someone already comfortable with Python, backend development, and ML tooling, a realistic target is roughly:

> **5–7 focused working days for a strong portfolio-ready project.**

The important point is that writing the code is not the main bottleneck.

Most of the value comes from:

- understanding each design decision,
- building a good evaluation set,
- comparing alternatives,
- interpreting metrics,
- and documenting the trade-offs.

---

# Working Principle

For every new feature, follow this sequence:

```text
1. What problem do we currently have?

2. What is the naive solution?

3. Why is the naive solution insufficient?

4. What improvement could solve the problem?

5. Which technology implements that improvement?

6. Implement the smallest useful version.

7. Test it.

8. Measure whether it actually helped.
```

This prevents the project from turning into a collection of trendy technologies.

---

# Completion Rule for Every Technology

A technology should not be considered part of your real stack just because it appears in the repository.

For each major component, you should be able to answer four questions:

### 1. What problem does it solve?

Example:

```text
Qdrant → efficient vector similarity search and vector persistence
```

### 2. What alternatives exist?

Example:

```text
Qdrant alternatives:
FAISS
Pinecone
Weaviate
pgvector
Milvus
```

### 3. Why did we choose this one?

The answer should relate to the project requirements.

### 4. What would happen if we removed it?

If you cannot explain what capability would be lost, the component may not be necessary.

---

# Final Learning Progression

```text
V0

"How does RAG work?"
        ↓

V1

"Are we retrieving the right information?"
        ↓

V2

"How can we prove that it is better?"
        ↓

V3

"How do we expose it as a real service?"
        ↓

V4

"How do we operate, test, and debug it in production?"
        ↓

V5

"How do we communicate the engineering work clearly?"
```

The most important progression is:

```text
V0 → V1 → V2
```

Once evaluation exists, later decisions no longer need to rely on statements such as:

> "It feels better."

Instead, you can say:

> "Recall@5 improved from 0.78 to 0.87 after adding hybrid retrieval, while median retrieval latency increased by 35 ms."

That is the difference between a tutorial RAG project and an AI engineering project.
