# Hybrid Support AI (hybrid-intent-agent)

A hybrid customer-support chatbot combining deterministic intent classification with LLM fallback and state management.

## Architecture

```
                    User
                      │
                      ▼
              ┌───────────────┐
              │   FastAPI     │
              └───────┬───────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Text Processing │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Intent Classifier│
             │   TensorFlow    │
             └────────┬────────┘
                      │
                 confidence?
                 /          \
               HIGH          LOW
                │             │
                ▼             ▼
         ┌────────────┐   ┌─────────┐
         │ Tool/Action│   │   LLM   │
         └─────┬──────┘   └────┬────┘
               │               │
               └───────┬───────┘
                       ▼
                Final Response
                       │
                       ▼
                 Conversation
                    State
```

## Project Structure

```
hybrid-support-ai/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── custom/
│
├── models/
│
├── src/
│   ├── preprocessing/
│   ├── classifier/
│   ├── ood/
│   ├── entities/
│   ├── conversation/
│   ├── tools/
│   ├── llm/
│   └── api/
│
├── tests/
│
├── evaluation/
│
├── scripts/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```
