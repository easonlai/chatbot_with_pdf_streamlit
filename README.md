# Chatbot with PDF for Semantic Search over Documents (Build with Streamlit, LangChain, Pinecone/Chroma/Azure Cognitive Search)

This repository contains a code example for how to build an interactive chatbot for semantic search over documents. The chatbot allows users to ask natural language questions and get relevant answers from a collection of documents. The chatbot uses [Streamlit](https://streamlit.io/) for web and chatbot interface, [LangChain](https://www.langchain.com/), and leverages various types of vector databases, such as [Pinecone](https://www.pinecone.io/), [Chroma](https://www.trychroma.com/), and [Azure Cognitive Search](https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search)’s [Vector Search](https://learn.microsoft.com/en-us/azure/search/vector-search-overview), to perform efficient and accurate similarity search. The code is written in Python and can be easily modified to suit different use cases and data sources.

Please also check out my story in [Medium](https://medium.com/@easonlai888) [(Streamlit and Vector Databases: A Guide to Creating Interactive Web Apps for Semantic Search over Documents)](https://easonlai888.medium.com/streamlit-and-vector-databases-a-guide-to-creating-interactive-web-apps-for-semantic-search-over-7a55d9b567ca) for more detail sharing.

* [preprocess_pinecone.ipynb](https://github.com/easonlai/chatbot_with_pdf_streamlit/blob/main/preprocess_pinecone.ipynb) <-- Example of using [Embedding Model](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models#embeddings-models) from [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview) to embed the content from the document and save it into Pinecone vector database.
* [preprocess_chroma.ipynb](https://github.com/easonlai/chatbot_with_pdf_streamlit/blob/main/preprocess_chroma.ipynb) <-- Example of using Embedding Model from Azure OpenAI Service to embed the content from the document and save it into Chroma vector database.
* [preprocess_acs.ipynb](https://github.com/easonlai/chatbot_with_pdf_streamlit/blob/main/preprocess_acs.ipynb) <-- Example of using Embedding Model from Azure OpenAI Service to embed the content from the document and save it into Azure Cognitive Search vector database.
* [consume_pinecone.ipynb](https://github.com/easonlai/chatbot_with_pdf_streamlit/blob/main/consume_pinecone.ipynb) <-- Example of using LangChain question-answering module to perform similarity search from the Pinecone vector database and use the [GPT-3.5 (text-davinci-003)](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/legacy-models#gpt-35) to summarize the result.
* [consume_chroma.ipynb](https://github.com/easonlai/chatbot_with_pdf_streamlit/blob/main/consume_chroma.ipynb) <-- Example of using LangChain question-answering module to perform similarity search from the Chroma vector database and use the GPT-3.5 (text-davinci-003) to summarize the result.
* [consume_acs.ipynb](https://github.com/easonlai/chatbot_with_pdf_streamlit/blob/main/consume_acs.ipynb) <-- Example of using LangChain question-answering module to perform similarity search from the Azure Cognitive Search vector database and use the GPT-3.5 (text-davinci-003) to summarize the result.
* [app_pinecone.py](https://github.com/easonlai/chatbot_with_pdf_streamlit/blob/main/app_pinecone.py) <-- Example of using Streamlit, LangChain, and Pinecone vector database to build an interactive chatbot to facilitate the semantic search over documents. It uses the [GPT-3.5-Turbo model](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models#gpt-35) from Azure OpenAI Service for result summarization and chat.
* [app_chroma.py](https://github.com/easonlai/chatbot_with_pdf_streamlit/blob/main/app_chroma.py) <-- Example of using Streamlit, LangChain, and Chroma vector database to build an interactive chatbot to facilitate the semantic search over documents. It uses the GPT-3.5-Turbo model from Azure OpenAI Service for result summarization and chat.
* [app_acs.py](https://github.com/easonlai/chatbot_with_pdf_streamlit/blob/main/app_acs.py) <-- Example of using Streamlit, LangChain, and Azure Cognitive Search vector database to build an interactive chatbot to facilitate the semantic search over documents. It uses the GPT-3.5-Turbo model from Azure OpenAI Service for result summarization and chat.

To run this Streamlit web app
```
streamlit run app_pinecone.py
```

**High-level architecture and flow of this Semantic Search over Documents demo**
![alt text](https://github.com/easonlai/chatbot_with_pdf_streamlit/blob/main/git-images/git-image-1.png)

Enjoy!