# 🏗️ BuildAtlas GenAI Copilot

An AI-powered Construction Planning and Project Analysis System built using **Streamlit**, **LangChain**, **FAISS**, **HuggingFace Embeddings**, and **Groq LLMs**.

BuildAtlas helps construction planners, engineers, and project managers estimate project costs, analyze risks, optimize timelines, evaluate sustainability practices, and perform intelligent scenario-based planning using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

### 🔐 Secure Login System

* Username and password authentication
* Session-based login management
* Logout functionality

### 📚 Retrieval-Augmented Generation (RAG)

* PDF-based knowledge base
* Automatic document loading
* Semantic search using FAISS vector database
* Context-aware responses

### 📊 Construction Project Analysis

Generate comprehensive project reports including:

#### Cost Estimation

* Total project cost calculation
* Material cost analysis
* Labour cost analysis
* Cost driver identification

#### Risk Analysis

* Delay risk assessment
* Cost overrun prediction
* Labour and material risk evaluation
* Risk severity classification

#### Timeline Planning

* Phase-wise schedule generation
* Foundation planning
* Structure planning
* Finishing schedule
* Total duration estimation

#### Cost Optimization

* Budget reduction recommendations
* Material trade-off analysis
* Labour optimization suggestions
* Timeline-cost balancing

#### Sustainability Analysis

* Eco-friendly recommendations
* Energy-efficient construction practices
* Green building suggestions

---

## 🔄 Scenario-Based Planning

The system automatically detects scenario-related queries such as:

* Weather impacts
* Rain delays
* Labour shortages
* Material shortages
* Inflation effects
* Material price fluctuations
* Project optimization requests

For detected scenarios, BuildAtlas generates:

### Scenario 1: Cost Optimized

Focuses on minimizing project expenses.

### Scenario 2: Time Optimized

Focuses on minimizing project delays.

### Scenario 3: Balanced

Provides the best trade-off between cost and time.

Each scenario includes:

* Cost impact
* Timeline impact
* Risk assessment
* Practical recommendations

---

## 🤖 AI Assistant

An integrated AI chatbot allows users to:

* Ask construction-related questions
* Query project information
* Retrieve knowledge from uploaded PDFs
* Get contextual engineering recommendations

---

## 🏗️ Technology Stack

### Frontend

* Streamlit

### LLM

* Groq
* Llama 3.3 70B Versatile

### RAG Components

* LangChain
* FAISS Vector Store
* HuggingFace Embeddings

### Document Processing

* PyPDFLoader
* DirectoryLoader
* RecursiveCharacterTextSplitter

### Environment Management

* python-dotenv

---

## 📂 Project Structure

```bash
BuildAtlas/
│
├── app.py
├── .env
├── requirements.txt
│
├── file/
│   ├── construction_data_1.pdf
│   ├── construction_data_2.pdf
│   └── ...
│
└── README.md
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/buildatlas.git

cd buildatlas
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

---

## 📄 Knowledge Base Setup

Place all construction-related PDF documents inside:

```bash
file/
```

The application will:

1. Load PDFs automatically
2. Split content into chunks
3. Generate embeddings
4. Create FAISS vector indexes
5. Enable semantic retrieval

---

## 🧠 Architecture

```text
PDF Documents
      │
      ▼
Document Loader
      │
      ▼
Text Splitter
      │
      ▼
Embeddings (BGE Base)
      │
      ▼
FAISS Vector Store
      │
      ▼
Retriever
      │
      ▼
Groq Llama 3.3 70B
      │
      ▼
Construction Analysis
```

---

## 📸 Application Workflow

### Login

Secure user authentication.

### Input Phase

User enters:

* Project Type
* Area
* Cost Per Sq Ft
* Material Quality
* Timeline
* Labour Availability

### AI Analysis

System retrieves relevant construction knowledge and generates:

* Cost Report
* Risk Report
* Timeline Plan
* Optimization Suggestions
* Sustainability Insights

### Output

Structured engineering recommendations are displayed.

---

## 🔮 Future Enhancements

* Voice-based project queries
* Construction cost prediction models
* BIM integration
* Real-time weather impact analysis
* Interactive project dashboards
* Multi-user authentication
* Report PDF generation
* Construction KPI monitoring

---

## 👨‍💻 Author

**Arpit Jain**

B.Tech (CSIT)
KIET Group of Institutions

Interests:
* Generative AI
* AI-Powered Engineering Solutions

---

## 📜 License

This project is licensed under the MIT License.

Feel free to fork, improve, and contribute.
