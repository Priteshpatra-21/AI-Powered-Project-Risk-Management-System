import os
import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import boto3
from io import BytesIO
from typing import Annotated, TypedDict, List

# Core Agentic Imports
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# --- 1. LIGHT-THEME UI SETUP ---
st.set_page_config(page_title="Risk Intel Pro", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1e293b; }
    .main-header {
        background: #1e40af; 
        padding: 25px; border-radius: 12px; text-align: center;
        margin-bottom: 30px; border-bottom: 5px solid #3b82f6;
    }
    .main-header h1 { color: #ffffff !important; font-size: 2.5rem; margin: 0; }
    .main-header p { color: #dbeafe; font-size: 1.1rem; }
    .metric-container {
        background: #f8fafc; border: 2px solid #e2e8f0;
        padding: 20px; border-radius: 12px; text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    .metric-label { color: #64748b; font-weight: 600; font-size: 0.9rem; text-transform: uppercase; }
    .metric-value { color: #1e293b; font-size: 2rem; font-weight: 800; }
    .stChatMessage { background-color: #f1f5f9 !important; border: 1px solid #cbd5e1 !important; color: #000000 !important; }
    </style>
    <div class="main-header">
        <h1>🛡️ RISK COMMAND CENTER</h1>
        <p>AWS S3 Powered Multi-Agent Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

# --- 2. AUTH & MODEL DISCOVERY ---
# Ensure AWS credentials and Google API Key are in Streamlit Secrets
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    
    # Initialize S3 Client
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets["AWS_DEFAULT_REGION"]
    )
    BUCKET_NAME = st.secrets["S3_BUCKET"]
except Exception as e:
    st.error(f"🔑 Credentials Missing or S3 Connection Error: {e}")
    st.stop()

@st.cache_resource
def discover_stable_model():
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for m in models:
            if "gemini-1.5-flash" in m: return m
        return models[0] if models else "gemini-1.5-flash"
    except: return "gemini-1.5-flash"

working_model_id = discover_stable_model()

# --- 3. DATA ENGINE (AWS S3 INTEGRATED) ---
@st.cache_data
def load_data_from_s3(file_key):
    try:
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=file_key)
        df = pd.read_csv(obj['Body'])
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading {file_key} from S3: {e}")
        return pd.DataFrame()

p_df = load_data_from_s3('project_risk_raw_dataset.csv')
m_df = load_data_from_s3('market_trends.csv')
t_df = load_data_from_s3('transaction.csv')

def get_safe_col(df, options):
    for opt in options:
        if opt in df.columns: return opt
    return None

# --- 4. AGENTIC BRAIN ---
llm = ChatGoogleGenerativeAI(model=working_model_id, temperature=0.1)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "History"]

# Node Definitions for the 5 Agents
def manager_agent(state: AgentState):
    query = state['messages'][-1].content
    prompt = f"PROJECT DATA:\n{p_df.describe().to_string()}\n\nAnalyze strategic risk for: {query}"
    return {"messages": [AIMessage(content=llm.invoke(prompt).content, name="Project_Risk_Manager")]}

def market_agent(state: AgentState):
    query = state['messages'][-1].content
    prompt = f"MARKET DATA:\n{m_df.tail(10).to_string()}\n\nAnalyze market trends: {query}"
    return {"messages": [AIMessage(content=llm.invoke(prompt).content, name="Market_Analyst")]}

def scoring_agent(state: AgentState):
    query = state['messages'][-1].content
    txn_summary = t_df.groupby('Payment_Status')['Amount_USD'].sum().to_string()
    prompt = f"TRANSACTION SUMMARY:\n{txn_summary}\n\nAssess financial defaults: {query}"
    return {"messages": [AIMessage(content=llm.invoke(prompt).content, name="Risk_Scorer")]}

def status_agent(state: AgentState):
    query = state['messages'][-1].content
    status_summary = p_df[['Project_ID', 'Project_Phase', 'Team_Turnover_Rate']].head(10).to_string()
    prompt = f"STATUS DATA:\n{status_summary}\n\nTrack progress/delays: {query}"
    return {"messages": [AIMessage(content=llm.invoke(prompt).content, name="Status_Tracker")]}

def reporting_agent(state: AgentState):
    query = state['messages'][-1].content
    prompt = f"ROLE: Reporting Officer. Generate analytic summary for: {query}"
    return {"messages": [AIMessage(content=llm.invoke(prompt).content, name="Reporting_Officer")]}

def router(state: AgentState):
    msg = state['messages'][-1].content.lower()
    if any(k in msg for k in ["market", "trend", "inflation"]): return "market"
    if any(k in msg for k in ["payment", "overdue", "transaction"]): return "scoring"
    if any(k in msg for k in ["delay", "status", "turnover"]): return "status"
    if any(k in msg for k in ["report", "summary", "analytics"]): return "reporting"
    return "manager"

# Graph Build
builder = StateGraph(AgentState)
builder.add_node("manager", manager_agent); builder.add_node("market", market_agent)
builder.add_node("scoring", scoring_agent); builder.add_node("status", status_agent)
builder.add_node("reporting", reporting_agent)

builder.set_conditional_entry_point(router, {
    "manager": "manager", "market": "market", "scoring": "scoring", 
    "status": "status", "reporting": "reporting"
})

for node in ["manager", "market", "scoring", "status", "reporting"]:
    builder.add_edge(node, END)
agent_brain = builder.compile()

# --- 5. DASHBOARD & VISUALS ---
risk_col = get_safe_col(p_df, ['Risk_Level', 'Risk'])
complexity_col = get_safe_col(p_df, ['Complexity_Score', 'Complexity'])
sentiment_col = get_safe_col(m_df, ['Market_Sentiment', 'Sentiment'])

c1, c2, c3, c4 = st.columns(4)

with c1:
    high_count = len(p_df[p_df[risk_col] == 'High']) if risk_col else 0
    st.markdown(f'<div class="metric-container" style="border-top: 5px solid #ef4444;"><div class="metric-label">Critical Risks</div><div class="metric-value" style="color: #ef4444;">{high_count}</div></div>', unsafe_allow_html=True)

with c2:
    overdue_total = t_df[t_df['Payment_Status'] == 'Overdue']['Amount_USD'].sum() if 'Payment_Status' in t_df.columns else 0
    st.markdown(f'<div class="metric-container" style="border-top: 5px solid #f59e0b;"><div class="metric-label">Overdue Exposure</div><div class="metric-value" style="color: #f59e0b;">${overdue_total/1e6:.1f}M</div></div>', unsafe_allow_html=True)

with c3:
    avg_comp = p_df[complexity_col].mean() if complexity_col else 0
    st.markdown(f'<div class="metric-container" style="border-top: 5px solid #3b82f6;"><div class="metric-label">Avg Complexity</div><div class="metric-value" style="color: #3b82f6;">{avg_comp:.1f}</div></div>', unsafe_allow_html=True)

with c4:
    last_sent = m_df[sentiment_col].iloc[-1] if sentiment_col else 0
    st.markdown(f'<div class="metric-container" style="border-top: 5px solid #10b981;"><div class="metric-label">Market Index</div><div class="metric-value" style="color: #10b981;">{last_sent:.2f}</div></div>', unsafe_allow_html=True)

col_left, col_right = st.columns([3, 2])
with col_left:
    if not p_df.empty:
        fig = px.bar(p_df.head(20), x='Project_ID', y=complexity_col, color=risk_col, title="Project Complexity (S3 Real-time Data)", color_discrete_map={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'})
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    if not p_df.empty:
        fig2 = px.pie(p_df, names=risk_col, title="Portfolio Risk Breakdown", color_discrete_sequence=['#ef4444', '#f59e0b', '#10b981'])
        st.plotly_chart(fig2, use_container_width=True)

# --- 6. CHAT INTERFACE ---
st.markdown("<h3 style='color: #1e40af;'>💬 Intelligence Briefing</h3>", unsafe_allow_html=True)
if "history" not in st.session_state: st.session_state.history = []

for m in st.session_state.history:
    with st.chat_message(m["role"]): st.write(m["content"])

if prompt := st.chat_input("Ask about S3-hosted project risks..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)
    
    with st.spinner("🤖 Consulting Experts..."):
        try:
            result = agent_brain.invoke({"messages": [HumanMessage(content=prompt)]})
            ans = result["messages"][-1]
            full_msg = f"**{ans.name.replace('_', ' ')}**: {ans.content}"
            st.chat_message("assistant").write(full_msg)
            st.session_state.history.append({"role": "assistant", "content": full_msg})
        except Exception as e:
            st.error(f"Execution Error: {e}")
