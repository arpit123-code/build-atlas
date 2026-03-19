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
#GROQ_API_KEY = "YOUR_GROQ_API_KEY"

llm = ChatGroq(
    #groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)

##################### Voice input #################
# def get_voice():
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         st.info("Speak...")
#         audio = r.listen(source)

#     try:
#         return r.recognize_google(audio)
#     except:
#         return ""

@st.cache_resource
def load_rag():
    loader = DirectoryLoader(
    "file/",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local("faiss_index")


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

#############UI#########################

st.set_page_config(layout="wide")
st.title("🏗️ BuildAtlas GenAI Copilot")

# Inputs
col1, col2 = st.columns(2)

with col1:
    project_type = st.selectbox("Project Type", ["House", "Apartment", "Commercial"])
    area = st.number_input("Area (sq ft)", 500, 5000, 1000)
    cost_per=st.number_input("cost_per_sqft")

with col2:
    material = st.selectbox("Material", ["Low", "Medium", "High"])
    timeline = st.number_input("Timeline (months)", 1, 24, 6)

labour = st.selectbox("Labour", ["Low", "Medium", "High"])

# Voice
# if st.button("🎤 Voice Input"):
#     voice_text = get_voice()
#     st.write("Detected:", voice_text)

# Query
#user_query = st.text_input("💬 Ask")

#######################Analysis#######################



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

    # Scenario Mode
    if base_query and is_scenario(base_query):
        st.header("🔮 Scenario Simulation")

        prompt = scenario_prompt(
            context, project_type, area, material, timeline, labour, base_query,cost_per
        )

        response = llm.invoke(prompt)
        st.write(response.content)

        # Normal Mode
    else:
        st.header("📊 Project Analysis")

        prompt = normal_prompt(
            context, project_type, area, material, timeline, labour,cost_per
        )

        response = llm.invoke(prompt)
        st.write(response.content)

#################### Chat ##########

st.header("🤖 AI Assistant")

chat_query = st.text_input("Ask anything")

if st.button("Ask AI"):

    docs = retriever.invoke(chat_query)
    context = "\n".join([d.page_content for d in docs])

    category = classify_query(chat_query)

    prompt = f"""
Use ONLY this knowledge:
{context}

Query Type: {category}

Question:
{chat_query}
"""

    response = llm.invoke(prompt)
    st.write(response.content)

