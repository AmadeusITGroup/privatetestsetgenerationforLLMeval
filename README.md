# LLM Test Set Generation Framework

This framework automates the creation of synthetic question-answer datasets for LLM evaluation from PDF sources.  
It extracts real-world domain text, ensures topic diversity, removes sensitive data (PII), and generates synthetic Q&A pairs.

---

##  Features

- **PDF Ingestion**: Extracts clean sentences from any PDF.
- **Embeddings**: Uses Azure OpenAI embeddings to represent sentences numerically.
- **Topic Diversification**: Clusters sentences into diverse topics using KMeans.
- **Privacy Filtering**: Detects and removes sensitive information (PII) with Presidio.
- **Synthetic QA Generation**: Generates rich, varied Q&A pairs via GPT-4 from sanitized text.
- **Modular Pipeline**: Built as a flexible multi-agent graph using LangGraph.

---

##  How It Works

Each step is an autonomous "agent" in a pipeline:

- **Diversity Agent**: Clusters data into meaningful topics.
- **Privacy Agent**: Analyzes clusters and flags sensitive content.
- **Synthetic Data Generator**: Creates QA samples while explaining coverage strategies.

## Setup 

Clone the repository:
Install dependencies:
pip install -r requirements.txt

Create a .env file with the following variables:


AZURE_OPENAI_KEY=<your_azure_openai_api_key>

AZURE_OPENAI_API_VERSION=2023-12-01-preview

AZURE_ENDPOINT=https://<your_azure_resource>.openai.azure.com/

EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002

GPT_4_MODEL_DEPLOYMENT_NAME=gpt-4




## Usage 


To use the tool, follow these steps:

 ```bash
   pip install privatetestsetgenerationforLLMeval

   privatetestsetgeneration



```

  
   
## Contributing

Contributions to improve the tool are welcome! Feel free to open issues for bugs or feature requests, or submit pull requests for enhancements.



## Acknowledgements

This project utilizes various libraries, including LangChain for document processing and Presidio for PII detection and anonymization.




Example Output


Question	

1.What are the primary goals of digital transformation?

2.How can data privacy be ensured when migrating to the cloud?

Answers

1.To leverage technology to optimize business processes, enhance customer experiences, and create new revenue streams.

2.Through encryption, compliance audits, access control, and anonymization of sensitive data.
