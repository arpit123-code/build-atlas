import streamlit as st
import speech_recognition as sr

# LangChain
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# ================= CONFIG =================

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.3,
    streaming=True,
)

# ================= PAGE CONFIG =================
st.set_page_config(layout="wide")

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.main { background-color: #0e1117; }

.title {
    text-align: center;
    font-size: 40px;
    color: #4CAF50;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ================= SESSION STATE =================
if "page" not in st.session_state:
    st.session_state.page = "input"

if "result" not in st.session_state:
    st.session_state.result = ""

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ================= LOGIN PAGE =================
if not st.session_state.logged_in:

    st.markdown("<div class='title'>🔐 BuildAtlas Login</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Secure Access to Construction AI</div>", unsafe_allow_html=True)

    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# ================= RAG =================
@st.cache_resource
def load_rag():
    loader = DirectoryLoader("file/", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore

vectorstore = load_rag()
retriever = vectorstore.as_retriever()

# ================= INTENT =================
def is_scenario(query):
    if not query:
        return False
    keywords = [
        "what if", "optimize", "compare",
        "increase", "decrease", "impact",
        "effect", "change", "delay",
        "rain", "weather", "temperature",
        "labour shortage", "material shortage",
        "price rise", "inflation"
    ]
    return any(k in query.lower() for k in keywords)

# ================= QUERY CLASSIFICATION =================
def classify_query(query):
    prompt = f"""
Classify the query into one or more of:
- cost
- material
- timeline
- risk

Also detect external factor if present:
- weather
- labour
- material_price
- logistics
- unknown

Return JSON:
{{
 "categories": [],
 "factor": ""
}}

Query: {query}
"""
    response = llm.invoke(prompt)
    return response.content

# ================= PROMPTS =================
def normal_prompt(context, project_type, area, material, timeline, labour,cost_per):
    return f"""
You are an expert civil engineer and construction planner with real-world experience in project estimation and management.

IMPORTANT:
Use ONLY the provided knowledge base for reasoning. Do not assume data outside it.

Knowledge Base:
{context}

Project Details:
- Type: {project_type}
- Area: {area} sq ft
- Material Quality: {material}
- Timeline Requirement: {timeline} months
- Labour Availability: {labour}
- cost_per:{cost_per}

NOTE:
We use abstracted categories like low, medium, and high to simplify complex construction parameters while still enabling accurate modeling and AI-based reasoning.

================ TASKS ================

1. COST ESTIMATION
- Estimate total project cost using formula:area*cost_per
- Provide breakdown (materials, labour, other)
- Explain key cost drivers

2. RISK ANALYSIS
- Identify delay risk
- Identify cost overrun risk
- Identify labour/material risks
- Give severity (Low / Medium / High) with reason

3. TIMELINE PLANNING
- Provide phase-wise construction schedule:
  (Planning, Foundation, Structure, Finishing)
- Give duration for each phase
- Total estimated duration
- Mention possible delays based on risks

4. COST OPTIMIZATION (VERY IMPORTANT)
- Suggest ways to reduce cost
- Mention trade-offs (quality vs cost vs time)
- Recommend changes in material/labour/timeline

5. SUSTAINABILITY
- Suggest eco-friendly improvements
- Suggest energy-efficient practices

6. FINAL INSIGHT
- Give a short expert recommendation

======================================

OUTPUT FORMAT (STRICT):

Cost:
Risk:
Timeline:
Optimization:
Sustainability:
Final Recommendation:
"""

def scenario_prompt(context, project_type, area, material, timeline, labour, user_query,cost_per):
    return f"""
You are an expert construction planner.

IMPORTANT:
- Use ONLY the provided knowledge base
- Do NOT hallucinate
- Apply real-world engineering logic

Knowledge Base:
{context}

Base Project:
- Type: {project_type}
- Area: {area}
- Material: {material}
- Timeline: {timeline}
- Labour: {labour}
- cost_per:{cost_per}

User Scenario:
{user_query}

TASK:

1. Identify:
   - What factor is changing (e.g., rain, labour, cost)
   - How it affects:
     - Cost
     - Timeline
     - Risk
     - Material usage

2. Apply realistic construction reasoning:
   Example:
   - Increased rain → delays curing → increases timeline
   - Rain → material damage → increases cost
   - Rain → site risk increases

3. Generate 3 scenarios:

### Scenario 1: Cost Optimized
- Adjust materials / labour to reduce cost impact

### Scenario 2: Time Optimized
- Reduce delay using extra labour / faster methods

### Scenario 3: Balanced
- Tradeoff between cost and time

For EACH scenario provide:
- Modified Parameters
- Cost Impact (↑ / ↓ with reason)
- Timeline Impact (↑ / ↓ with reason)
- Risk Level (Low / Medium / High)
- Insight

4. Recommend BEST scenario:
- Based on practicality

5. Suggest:
- Mitigation strategies (e.g., waterproofing, scheduling)

OUTPUT FORMAT:

Detected Factor:
Impact Analysis:

Scenario 1:
Scenario 2:
Scenario 3:

Best Scenario:
Reason:

Mitigation Tips:
""" 

# ================= PAGE 1: INPUT =================
if st.session_state.page == "input":

    # Logout button
    colA, colB = st.columns([8,1])
    with colB:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.page = "input"
            st.rerun()

    st.markdown("<div class='title'>🏗️ BuildAtlas GenAI Copilot</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>AI Construction Planning System</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        project_type = st.selectbox("🏢 Project Type", ["House", "Apartment", "Commercial"])
        area = st.number_input("📐 Area (sq ft)", 500, 5000, 1000)
        cost_per = st.number_input("💰 Cost per sqft")

    with col2:
        material = st.selectbox("🧱 Material", ["Low", "Medium", "High"])
        timeline = st.number_input("⏳ Timeline (months)", 1, 24, 6)

    labour = st.selectbox("👷 Labour", ["Low", "Medium", "High"])

    if st.button("🚀 Analyze Project"):

        base_query = f"""
        Type: {project_type}
        Area: {area}
        Material: {material}
        Timeline: {timeline}
        Labour: {labour}
        """

        docs = retriever.invoke(base_query)
        context = "\n".join([d.page_content for d in docs])

        with st.spinner("🤖 AI analyzing..."):

            if base_query and is_scenario(base_query):
                prompt = scenario_prompt(
                    context, project_type, area, material, timeline, labour, base_query, cost_per
                )
            else:
                prompt = normal_prompt(
                    context, project_type, area, material, timeline, labour, cost_per
                )

            response = llm.invoke(prompt)

        st.session_state.result = response.content
        st.session_state.page = "output"
        st.rerun()

# ================= PAGE 2: OUTPUT =================
elif st.session_state.page == "output":

    st.markdown("<div class='title'>📊 Analysis Result</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='card'>{st.session_state.result}</div>", unsafe_allow_html=True)

    if st.button("⬅️ Back"):
        st.session_state.page = "input"
        st.rerun()

# ================= CHAT =================
st.markdown("### 🤖 AI Assistant")

chat_query = st.text_input("Ask anything")

if st.button("Ask AI"):

    docs = retriever.invoke(chat_query)
    context = "\n".join([d.page_content for d in docs])

    category = classify_query(chat_query)

    with st.spinner("Thinking..."):
        response = llm.invoke(f"""
Use ONLY this knowledge:
{context}

Query Type: {category}

Question:
{chat_query}
""")

    st.markdown(f"<div class='card'>{response.content}</div>", unsafe_allow_html=True)