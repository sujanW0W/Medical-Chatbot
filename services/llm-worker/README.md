# Medical Chatbot using Multi-Agentic Retrieval Augmented Generation (RAG) 🤖🩺

A sophisticated medical chatbot that provides accurate medical information through a multi-agentic RAG system. This project implements a multi-agentic paradigm orchestrated by an LLM-powered agent to intelligently process user queries, retrieve relevant information from a medical knowledge base, perform real-time web searches when necessary, and synthesize a coherent, medically relevant response.

![Architecture](/assets/Architecture.png)

### 🔄 System Workflow

The system’s workflow is non-deterministic and adapts dynamically to the complexity and nature of user queries. The key steps involved are:

1.  **Query Submission:** Users submit their medical queries through the interactive Streamlit-based web interface.
2.  **Orchestration:** The Orchestrator Agent, powered by an LLM, analyzes the submitted query to understand its intent and complexity. Based on this analysis, it intelligently selects the appropriate subsequent agents (e.g., Retrieval Agent for knowledge base queries, Web Search Agent for current or external information).
3.  **Information Retrieval:** The invoked agents execute their specific tasks. The Retrieval Agent searches the Pinecone vector database for relevant medical documents, while the Web Search Agent performs real-time searches using DuckDuckGo for information not present in the knowledge base.
4.  **Context Synthesis:** The Synthesis Agent receives the original query, along with any retrieved documents from the knowledge base and results from web searches. It consolidates and structures this diverse information into a coherent context, preparing it for the final response generation.
5.  **Response Generation:** The structured context is passed to the Google Gemini 1.5 Flash LLM. The LLM uses this augmented information to generate a final, accurate, and medically relevant response, which is then displayed back to the user in the Streamlit interface.

## 💡 Features

- **Multi-Agent Architecture:** Utilizes specialized agents for distinct tasks (retrieval, web search, synthesis) orchestrated by another LLM-powered agent.
- **Retrieval Augmented Generation (RAG):** Combines retrieval from the knowledge base and web search results with the LLM's capabilities to generate informed responses.
- **Medical Knowledge Base:** Integrates with a Pinecone vector store populated with medical documents for domain-specific information retrieval.
- **Real-time Web Search:** Incorporates DuckDuckGo Search for accessing up-to-date information beyond the internal knowledge base.
- **Advanced Language Model:** Powered by Google's Gemini 1.5 Flash for natural language understanding, generation, and synthesis.
- **Interactive Interface:** Provides a user-friendly web interface built with Streamlit.

## 🛠️ Technology Stack

| Component       | Technology                           |
| --------------- | ------------------------------------ |
| Language Model  | Google Gemini 1.5 Flash              |
| Vector Database | Pinecone                             |
| Embedding Model | `all-MiniLM-L6-v2` (via HuggingFace) |
| Web Search      | DuckDuckGo API                       |
| Orchestration   | LangChain + LangGraph                |
| UI              | Streamlit                            |

## ✅ Prerequisites

- Python 3.9+
- Pinecone API key
- Google AI Studio API key
- DuckDuckGo API access
- LangChain & LangGraph

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/sujanW0W/Medical-Chatbot
cd medical-chatbot
```

2. Create and activate virtual environment:

```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_google_api_key
```

## 📁 Project Structure

```
medical-chatbot/
├── assets/
├── data/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── retrieval.py
│   │   ├── synthesis.py
│   │   ├── web_search.py
│   │   └── state.py
│   ├── __init__.py
│   ├── helpers.py
│   ├── orchestrator.py
│   └── prompts.py
├── app.py
├── requirements.txt
├── setup.py
├── store_index.py
├── template.py
├── start.sh
└── README.md
```

## 🚀 Usage

1.  **Populate the Pinecone index:** Before running the application for the first time, you need to process the medical documents (extraction, chunking, and embeddings) and upload them to your Pinecone index.

    ```bash
    python store_index.py
    ```

    _Ensure your Pinecone environment and index name are correctly configured in `store_index.py` and your `.env` file._

2.  **Start the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

    Alternatively, you can use the provided start script:

    ```bash
    ./start.sh
    ```

3.  **Open your web browser and navigate to:**

    ```
    http://localhost:8501
    ```

4.  **Start chatting with the medical bot!**

## 📊 Result

The system was tested with 20 diverse medical queries related to diseases, diagnosis, and drugs.

- 🕒 **Avg. Response Time**: 3.7 seconds
- ✅ **Query Success Rate**: 94.8%
- 💡 **System Strengths**:
    - Handles real-time, up-to-date questions
    - Maintains conversational context
    - Generates informative, non-hallucinated answers

## 🖼️ Preview

Here is a screenshot of the chatbot interface:

![Chatbot Interface](/assets/Result.png)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- LangChain for providing the framework components.
- Google for the powerful Gemini language model.
- Pinecone for the efficient vector storage solution.
- DuckDuckGo Search for the web search capabilities.
- Streamlit for enabling rapid development of the web interface.

## 👨‍💻 Collaborators

[@sujanW0W](https://github.com/sujanW0W)

[@Akatzz12](https://github.com/Akatzz12)

[@Aayush-lamsal](https://github.com/aayush-lamsal)

## 📧 Contact

Email: [Sujan Maharjan](mailto:sujan.maharjan.1@ndsu.edu)
