LLM Test Set Generation Framework

This framework automates the creation of synthetic question-answer datasets for LLM evaluation from PDF sources. It extracts real-world domain text, ensures topic diversity, removes sensitive data (PII), and generates synthetic Q&A pairs.

✨ Features

PDF Ingestion: Extracts clean sentences from any PDF.
Embeddings: Uses Azure OpenAI embeddings to represent sentences numerically.
Topic Diversification: Clusters sentences into diverse topics using KMeans.
Privacy Filtering: Detects and removes sensitive information (PII) with Presidio.
Synthetic QA Generation: Generates rich, varied Q&A pairs via GPT-4 from sanitized text.
Modular Pipeline: Built as a flexible multi-agent graph using langgraph.
