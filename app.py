import os
import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import boto3
from typing import Annotated, TypedDict, List

# Core Agentic Imports
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# --- 1. UI & AUTH ---
st.set_page_config(page_title="Risk Intel Pro", layout="wide")

# Custom CSS for Light Mode
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1e293b; }
    .main-header { background: #1e40af; padding: 20px; border-radius: 12px; text-align: center; color: white; margin-bottom: 20px;}
    .metric-container { background: #f8fafc; border: 1px solid #e2e8f0; padding: 15px; border-radius: 10px; text-align: center; }
    </style>
    <div class="main-header"><h1>🛡️ RISK COMMAND CENTER</h1><p>AWS S3 + Multi-Agent Intelligence</p></div>
    """, unsafe_allow_html=True)

try:
    # Setup Clients
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    
    s3 = boto3.client(
        "s3",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets["AWS_DEFAULT_REGION"]
    )
    BUCKET = st.secrets["S3_BUCKET"]
except Exception as e:
    st.error("🔑 Configuration Error: Check your Streamlit Secrets.")
    st.stop()

# --- 2. S3 DATA LOADING ---
@st.cache_data
def load_s3_data(file_name):
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=file_name)
        df = pd.read_csv(obj['Body'])
        df.columns = df.columns.str.strip()
        return df
    except:
        return pd.DataFrame()

p_df = load_s3_data('project_risk_raw_dataset.csv')
m_df = load_s3_data('market_trends.csv')
t_df = load_s3_data('transaction.csv')

# --- 3. AGENT BRAIN ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "History"]

# Agent Nodes
def manager_agent(s: AgentState):
    res = llm.invoke(f"Risk Manager: Analyze this query using Project Data: {s['messages'][-1].content}")
    return {"messages": [AIMessage(content=res.content, name="Risk_Manager")]}

def market_agent(s: AgentState):
    res = llm.invoke(f"Market Analyst: Use Market Trends to answer: {s['messages'][-1].content}")
    return {"messages": [AIMessage(content=res.content, name="Market_Analyst")]}

def scoring_agent(s: AgentState):
    res = llm.invoke(f"Scorer: Analyze transaction/payment risk for: {s['messages'][-1].content}")
    return {"messages": [AIMessage(content=res.content, name="Risk_Scorer")]}

def status_agent(s: AgentState):
    res = llm.invoke(f"Status Tracker: Check timeline/delays for: {s['messages'][-1].content}")
    return {"messages": [AIMessage(content=res.content, name="Status_Tracker")]}

def reporting_agent(s: AgentState):
    res = llm.invoke(f"Reporting Officer: Create a summary for: {s['messages'][-1].content}")
    return {"messages": [AIMessage(content=res.content, name="Reporting_Officer")]}

def router(s: AgentState):
    q = s['messages'][-1].content.lower()
    if any(k in q for k in ["market", "trend", "inflation"]): return "market"
    if any(k in q for k in ["payment", "transaction", "overdue"]): return "scoring"
    if any(k in q for k in ["delay", "status", "phase"]): return "status"
    if any(k in q for k in ["report", "summary"]): return "reporting"
    return "manager"

# Graph
workflow = StateGraph(AgentState)
workflow.add_node("manager", manager_agent); workflow.add_node("market", market_agent)
workflow.add_node("scoring", scoring_agent); workflow.add_node("status", status_agent)
workflow.add_node("reporting", reporting_agent)

workflow.set_conditional_entry_point(router, {
    "manager": "manager", "market": "market", "scoring": "scoring", 
    "status": "status", "reporting": "reporting"
})

for node in ["manager", "market", "scoring", "status", "reporting"]:
    workflow.add_edge(node, END)
agent_brain = workflow.compile()

# --- 4. DASHBOARD ---
if not p_df.empty:
    c1, c2, c3 = st.columns(3)
    c1.metric("Critical Risks", len(p_df[p_df['Risk_Level']=='High']))
    c2.metric("Avg Complexity", round(p_df['Complexity_Score'].mean(), 2))
    c3.metric("Live Market Index", m_df['Value'].iloc[-1] if not m_df.empty else 0)

    st.plotly_chart(px.pie(p_df, names='Risk_Level', title="Risk Overview", color_discrete_sequence=['#ef4444', '#f59e0b', '#10b981']))

# --- 5. CHAT ---
if "history" not in st.session_state: st.session_state.history = []
for m in st.session_state.history:
    with st.chat_message(m["role"]): st.write(m["content"])

if prompt := st.chat_input("Ask about high risks or market trends..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.spinner("🤖 Consulting S3 Data & Agents..."):
        result = agent_brain.invoke({"messages": [HumanMessage(content=prompt)]})
        ans = result["messages"][-1]
        full_msg = f"**{ans.name}**: {ans.content}"
        st.chat_message("assistant").write(full_msg)
        st.session_state.history.append({"role": "assistant", "content": full_msg})
