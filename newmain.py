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

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local("faiss_index")


    return vectorstore

vectorstore = load_rag()
retriever = vectorstore.as_retriever()

# ================= 🔥 ADDED: STRUCTURED DATA EXTRACTION =================
import re

def extract_numbers(context):
    data = {}

    material = re.search(r"material.*?(\d+)%", context.lower())
    labour = re.search(r"labour.*?(\d+)%", context.lower())
    saving = re.search(r"(save|saving).*?(\d+)%", context.lower())

    if material:
        data["material_percent"] = int(material.group(1))
    if labour:
        data["labour_percent"] = int(labour.group(1))
    if saving:
        data["saving_percent"] = int(saving.group(2))

    return data

# ================= 🔥 ADDED: RULE ENGINE =================

def calculate_project(area, cost_per_sqft, timeline, labour, material, rag_data):

    total_cost = area * cost_per_sqft

    material_pct = rag_data.get("material_percent", 50)
    labour_pct = rag_data.get("labour_percent", 30)

    material_cost = (material_pct / 100) * total_cost
    labour_cost = (labour_pct / 100) * total_cost
    other_cost = total_cost - material_cost - labour_cost

    labour_factor = {"Low": 1.3, "Medium": 1.0, "High": 0.8}
    material_factor = {"Low": 1.2, "Medium": 1.0, "High": 0.9}

    adjusted_time = timeline * labour_factor[labour] * material_factor[material]

    risk = 5
    if labour == "Low":
        risk += 2
    if material == "Low":
        risk += 2

    risk = min(10, risk)

    return {
        "total_cost": total_cost,
        "material_cost": material_cost,
        "labour_cost": labour_cost,
        "other_cost": other_cost,
        "timeline": adjusted_time,
        "risk": risk
    }

    # ================= 🔥 ADDED: OPTIMIZATION =================

def optimize_project(base, rag_data):

    saving_pct = rag_data.get("saving_percent", 10)

    new_cost = base["total_cost"] * (1 - saving_pct / 100)
    new_time = base["timeline"] * 0.9

    return {
        "optimized_cost": new_cost,
        "optimized_time": new_time,
        "saving": base["total_cost"] - new_cost
    }

    # ================= 🔥 ADDED: SUSTAINABILITY =================

def sustainability_insights(context):

    insights = []

    if "fly ash" in context.lower():
        insights.append("Fly ash bricks reduce cost by ~10%")

    if "solar" in context.lower():
        insights.append("Solar panels reduce long-term energy cost")

    if not insights:
        insights.append("INSUFFICIENT DATA")

    return insights


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
def normal_prompt(context, project_type, area, material, timeline, labour,cost_per_sqft):
    planning = timeline * 0.1
    foundation = timeline * 0.2
    structure = timeline * 0.4
    finishing = timeline * 0.3
    return f"""
You are a professional civil engineer.

CRITICAL RULES:
- Use knowledge base FIRST.
If required data is missing, use provided project inputs.
Do NOT hallucinate beyond these two sources.
- DO NOT hallucinate
- ALL outputs must be NUMERICAL and derived using formulas or explicit values from context
- If data is missing → say "INSUFFICIENT DATA"
- Do NOT guess
- show only the value

- In final recommendation give the recommendation according to knowledge base 

Knowledge Base:
{context}

Project Details:
- Type: {project_type}
- Area: {area} sq ft
- Material Quality: {material}
- Timeline: {timeline} months
- Labour: {labour}
-planning:{planning}
-foundation:{foundation}
-structure:{structure}
- finishing:{finishing}
-cost_per_sqft:{cost_per_sqft}

================ CALCULATION RULES ================

You MUST use engineering-style calculations:

1. COST FORMULA:
   Total Cost = Area × Cost_per_sqft

   If breakdown available:
   - Material Cost = % × Total Cost
   - Labour Cost = % × Total Cost
   - Other Cost = remaining %

2. TIMELINE FORMULA:
   Total Duration = Base Duration × Adjustment Factors

   Adjustment factors:
   - Labour ↓ → Time ↑
   - High Material → Time ↓
   - Risk factors → Time ↑

3. RISK SCORING:
   Assign numerical score (1–10):
   - 1–3 = Low
   - 4–7 = Medium
   - 8–10 = High

4. OPTIMIZATION:
   Must show NUMERICAL change:
   Example:
   - Cost reduced from ₹X → ₹Y
   - Time reduced from X → Y months

==================================================

TASKS:

1. COST ESTIMATION
- Calculate total cost (₹)
- Provide exact breakdown (₹ values)
- Show formula used

2. RISK ANALYSIS
- Provide risk score (1–10)
- Justify using context

3. TIMELINE
- Phase-wise duration (numbers only)
- Total duration (months)

4. OPTIMIZATION
- Show BEFORE vs AFTER numbers
- No vague suggestions

5. SUSTAINABILITY
- Only if data exists in context

6. FINAL RECOMMENDATION
- Based on calculated values only

==================================================

OUTPUT FORMAT (STRICT):

Cost:
- Total:
- Material:
- Labour:
- Other:
- Formula Used:

Risk:
- Score:
- Level:
- Reason:

Timeline:
- Planning:
- Foundation:
- Structure:
- Finishing:
- Total:

Optimization:
- Before:
- After:
- Savings:

Sustainability:

Final Recommendation:
"""

def scenario_prompt(context, project_type, area, material, timeline, labour, user_query,cost_per_sqft):
    return f"""
You are an expert construction engineer.

CRITICAL RULES:
- Use knowledge base FIRST.
If required data is missing, use provided project inputs.
Do NOT hallucinate beyond these two sources.
- DO NOT hallucinate
- ALL outputs must be NUMERICAL
- Use formulas for every prediction
- If missing data → say "INSUFFICIENT DATA"

Knowledge Base:
{context}

Base Project:
- Type: {project_type}
- Area: {area}
- Material: {material}
- Timeline: {timeline}
- Labour: {labour}
-cost_per_sqft:{cost_per_sqft}

User Scenario:
{user_query}

================ CALCULATION LOGIC ================

You MUST follow:

1. Identify factor change:
   Example:
   - Rain ↑ → Work efficiency ↓
   - Rain ↑ → Material wastage ↑

2. Apply NUMERICAL IMPACT:

   Example rules:
   - Rain increase → Timeline +10–30%
   - Rain increase → Cost +5–15%
   - Labour shortage → Time +20–40%

3. Use formulas:

   New Cost = Base Cost × (1 + Impact %)
   New Time = Base Time × (1 + Impact %)

4. Risk scoring:
   1–10 scale only

==================================================

TASK:

Step 1: Detect factor  
Step 2: Calculate base values  
Step 3: Apply % changes  
Step 4: Generate 3 scenarios  

==================================================

SCENARIOS:

### Scenario 1: Cost Optimized
- Reduce cost impact using numerical adjustment

### Scenario 2: Time Optimized
- Reduce delay using numerical adjustment

### Scenario 3: Balanced
- Moderate cost + time

==================================================

FOR EACH SCENARIO (STRICT):

- Modified Parameters
- Cost (₹ with formula)
- Timeline (months with formula)
- Risk Score (1–10)
- Explanation (based on calculation only)

==================================================

OUTPUT FORMAT:

Detected Factor:

Base Values:
- Cost:
- Time:

Impact Applied:
- Cost Change (%):
- Time Change (%):

Scenario 1:
Scenario 2:
Scenario 3:

Best Scenario:
Reason (based on numbers only):

Mitigation:
"""

#############UI#########################

st.set_page_config(layout="wide")
st.title("🏗️ BuildAtlas GenAI Copilot")

# Inputs
col1, col2 = st.columns(2)

with col1:
    project_type = st.selectbox("Project Type", ["House", "Apartment", "Commercial"])
    area = st.number_input("Area (sq ft)", 500, 5000, 1000)
    cost_per_sqft = st.number_input("cost (per_sqft)", 500, 5000, 1000)


with col2:
    material = st.selectbox("Material", ["Low", "Medium", "High"])
    timeline = st.number_input("Timeline (months)", 1, 24, 6)

labour = st.selectbox("Labour", ["Low", "Medium", "High"])

# Voice
# if st.button("🎤 Voice Input"):
#     voice_text = get_voice()
#     st.write("Detected:", voice_text)

# Query
user_query = st.text_input("💬 Ask")

#######################Analysis#######################



if st.button("🚀 Analyze Project"):

    docs = retriever.invoke(str(user_query))
    context = "\n".join([d.page_content for d in docs])

    # 🔥 ADDED: structured extraction
    rag_data = extract_numbers(context)

    # 🔥 ADDED: rule-based calculation
    calc = calculate_project(
        area, cost_per_sqft, timeline, labour, material, rag_data
    )

    # 🔥 ADDED: optimization
    opt = optimize_project(calc, rag_data)

    # 🔥 ADDED: sustainability
    sustain = sustainability_insights(context)

    st.header("📊 Project Analysis")

    # 🔥 MODIFIED: LLM now ONLY formats
    final_prompt = f"""
STRICTLY USE THESE VALUES (DO NOT CHANGE):

Cost:
Total: {calc['total_cost']}
Material: {calc['material_cost']}
Labour: {calc['labour_cost']}
Other: {calc['other_cost']}

Timeline:
{calc['timeline']}

Risk:
{calc['risk']}

Optimization:
New Cost: {opt['optimized_cost']}
Savings: {opt['saving']}

Sustainability:
{sustain}

TASK:
Format output professionally.
Do NOT modify numbers.
"""

    response = llm.invoke(final_prompt)
    st.write(response.content)
#################### Chat ##########

st.header("🤖 AI Assistant")

chat_query = st.text_input("Ask anything")

if st.button("Ask AI"):

    docs = retriever.invoke(chat_query)
    context = "\n".join([d.page_content for d in docs])

    rag_data = extract_numbers(context)

    calc = calculate_project(
        area, cost_per_sqft, timeline, labour, material, rag_data
    )

    response = llm.invoke(f"""
Use this data:
{context}

Calculated Values:
{calc}

Answer the query using BOTH.
Do NOT hallucinate.
""")

    st.write(response.content)

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

