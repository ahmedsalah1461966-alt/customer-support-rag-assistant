# 🎧 Customer Support Knowledge Assistant

A Retrieval-Augmented Generation (RAG) assistant with conversational memory, grounded on the
[Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).

Built as a mid-term AI/LLM engineering project: instead of chunking a PDF, this system treats
every `(instruction, category, intent, response)` row of a structured support dataset as one
atomic knowledge unit, embeds it, indexes it in FAISS, and answers customer questions by
retrieving and rephrasing the closest approved support entries — never inventing a policy that
isn't in the data.

## Table of contents

- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Testing & evaluation](#testing--evaluation)
- [Limitations](#limitations)
- [License](#license)

## Features

- **Data integration** — loads the Bitext customer-support dataset directly from the Hugging
  Face Hub (26,872 rows, 27 intents across 11 categories).
- **Knowledge-base construction** — converts a de-duplicated, sampled set of rows into
  retrievable LangChain `Document`s, with a data-quality filter that drops malformed
  numbered-list fragments found in the raw dataset.
- **Embeddings** — free, local `sentence-transformers/all-MiniLM-L6-v2` embeddings (no API key
  required).
- **Vector search** — a FAISS index with an MMR retriever for relevant *and* diverse results.
- **RAG pipeline** — LangChain `create_retrieval_chain` + a grounding prompt that forbids the
  model from inventing order numbers, refund amounts, or policies not present in the retrieved
  context.
- **Relevance gate** — a minimum-cosine-similarity threshold that forces a fixed refusal
  sentence when nothing relevant was retrieved, instead of leaving refusal to LLM judgment
  alone.
- **Conversational memory** — full buffer memory per session (`RunnableWithMessageHistory` +
  `InMemoryChatMessageHistory`), with follow-up questions rewritten into standalone questions
  *before* retrieval, so pronouns like "that" and "it" resolve correctly.
- **Deterministic generation** — greedy decoding, so the same question with the same retrieved
  context always produces the same answer.
- **Testing & evaluation** — 12 hand-written test cases (grounded answers, follow-up/memory
  resolution, out-of-scope refusal, and a repeated-question determinism check), a qualitative
  evaluation table with an explicit pass rule, and retrieval-quality metrics.
- **Interface** — an in-notebook Gradio `ChatInterface` for interactive use, no separate server
  or tunnel required.

## Architecture

```text
   bitext/Bitext-customer-support-llm-chatbot-training-dataset (Hugging Face Hub)
                                  |
                    datasets.load_dataset(...)["train"]
                                  |
                  build_documents()  -->  1 Document per sampled
                  (instruction, category, intent, response) row
                                  |
              HuggingFaceEmbeddings (all-MiniLM-L6-v2)  -->  FAISS index
                                  |
                    Retriever (MMR, k=4, fetch_k=15)
                                  ^
                                  |
  User question + chat history --> rewritten standalone question
                                  |
                    Retrieved context (top-k support entries)
                                  |
              Grounding prompt (context + history + question) --> LLM
                                  |
                Answer + cited intent/category + updated memory
```

The LLM automatically switches between `Qwen/Qwen2.5-1.5B-Instruct` (GPU) and
`google/flan-t5-base` (CPU-only), so the notebook runs on both Colab and Kaggle regardless of
whether a GPU accelerator is enabled.

## Dataset

| | |
|---|---|
| Source | [`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) |
| Rows | 26,872 |
| Categories | 11 (`ACCOUNT`, `CANCEL`, `CONTACT`, `DELIVERY`, `FEEDBACK`, `INVOICE`, `ORDER`, `PAYMENT`, `REFUND`, `SHIPPING`, `SUBSCRIPTION`) |
| Intents | 27 |
| Fields used | `instruction`, `category`, `intent`, `response` |

The dataset's responses are **templated** (placeholders such as `{{Order Number}}`), not live
account data — this project is a knowledge-retrieval demo, not a connection to a real order
system.

## Project structure

```text
customer-support-rag-assistant/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── notebooks/
    └── customer_support_rag_assistant.ipynb
```

## Setup

### Option A — Run on Kaggle / Colab (recommended)

1. Upload `notebooks/customer_support_rag_assistant.ipynb`.
2. Make sure **Internet / Accelerator** is turned on in the notebook's settings (a GPU
   accelerator is optional but gives noticeably better answers).
3. Run the cells top to bottom. The first code cell installs and cleanly reinstalls the
   Hugging Face stack; if you see an `ImportError` right after it, use
   **Runtime → Restart session** (Colab) or **Restart Kernel** (Kaggle) once, then continue
   from the top without re-running the install cell twice in a row.

### Option B — Run locally

```bash
git clone https://github.com/ahmedsalah1461966-alt/customer-support-rag-assistant.git
cd customer-support-rag-assistant
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/customer_support_rag_assistant.ipynb
```

A GPU is optional; on CPU the notebook automatically falls back to a smaller model
(`google/flan-t5-base`).

## Usage

Once the notebook has run through the setup cells, ask the assistant questions directly:

```python
chat("How can I cancel my order?")
chat("How long will that take?")   # follow-up - resolved using conversation memory
```

Or use the Gradio chat widget in the last section of the notebook for an interactive UI.

## Testing & evaluation

The notebook defines 12 test cases and reports the results in a pass/review table rather than
a single accuracy score:

| Category | What it checks |
|---|---|
| Grounded answers (5 cases) | Order cancellation, refunds, tracking, account, payment |
| Follow-up / memory (2 cases) | Pronoun and reference resolution across turns |
| Out-of-scope refusal (4 cases) | Unrelated questions, invented policies, borderline requests |
| Determinism (1 case) | The same question, asked twice, must get the same answer |

Results vary slightly with the model and hardware in use (GPU vs. CPU model, library
versions), so exact pass counts are not hard-coded here — run the notebook's evaluation
section (`evaluation_table`, `capability_summary`, and the final checklist) to see a live
result for your environment.

## Limitations

- Knowledge is limited to the sampled Bitext dataset; nothing outside its 27 intents is
  available, and the assistant is designed to say so rather than guess.
- Responses are templated, not connected to a real order/account/payment backend.
- Retrieval is capped at `k = 4` entries per question.
- A small local LLM can still occasionally misjudge a borderline out-of-scope request; the
  similarity gate reduces this but does not eliminate it completely.
- Conversation memory is in-memory only and is cleared when the notebook session ends.

See the "Limitations" and "Final Project Checklist" sections inside the notebook for the full,
up-to-date list.

## License

Distributed under the MIT License — see [`LICENSE`](LICENSE) for details.

## Author

**[Ahmed Salah Hussein](https://github.com/ahmedsalah1461966-alt)**
