# Medical Chatbot 🤖🩺

A sophisticated medical chatbot leveraging LangGraph, LangChain, and Google's Gemini model to provide accurate medical information through a multi-agent system.

## Features

-   Multi-agent architecture for intelligent query processing
-   Medical knowledge base integration via Pinecone vector store
-   Real-time web search capabilities
-   Natural language understanding using Google's Gemini 1.5 Flash
-   Interactive Streamlit web interface

## Prerequisites

-   Python 3.9+
-   Pinecone API key
-   Google AI Studio API key
-   DuckDuckGo API access

## Installation

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

## Project Structure

```
medical-chatbot/
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

## Usage

1. Start the Streamlit application:

```bash
streamlit run app.py
```

Another option to run the project is to simply start script.

```bash
# On Windows
.\start.sh

# On Unix-like systems (Linux/Mac)
./start.sh
```

2. Open your web browser and navigate to:

```
http://localhost:8501
```

3. Start chatting with the medical bot!

## Agent System

The chatbot uses three specialized agents:

-   **Retrieval Agent**: Searches through medical knowledge base
-   **Web Search Agent**: Performs real-time internet searches
-   **Synthesis Agent**: Combines information and generates responses

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

-   LangChain for the agent framework
-   Google for the Gemini language model
-   Pinecone for vector storage
-   Streamlit for the web interface

## Collaborators

[@sujanW0W](https://github.com/sujanW0W)

[@Akatzz12](https://github.com/Akatzz12)

[@Aayush-lamsal](https://github.com/aayush-lamsal)

## Contact

Email: [Sujan Maharjan](mailto:sujan.maharjan.1@ndsu.edu)
