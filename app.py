"""
Main Streamlit Chatbot
"""
import streamlit as st
import json
import os
from langchain_agent import LangChainAgent
from schema_extractor import SchemaExtractor
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="Dynamic SQL Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Dynamic SQL Chatbot with LangChain")
st.markdown("Ask questions → Get SQL → See results → Get insights → Visualize")

# Initialize components
@st.cache_resource
def init_agent():
    return LangChainAgent()

@st.cache_resource  
def init_schema():
    return SchemaExtractor()
 
# Sidebar
with st.sidebar:
    st.header("🔧 Setup")
    
    # Extract schema button
    if st.button("🔄 Extract Database Schema"):
        with st.spinner("Extracting schema..."):
            extractor = init_schema()
            schema = extractor.extract_schema()
            
            if "error" not in schema:
                st.success("✅ Schema extracted!")
                
                # Show schema file
                with open("schema_info.json", "r") as f:
                    schema_data = json.load(f)
                
                st.write(f"**Database:** {schema_data['database']}")
                st.write(f"**Tables:** {len(schema_data['tables'])}")
                
                # Let user view schema
                with st.expander("📋 View Schema Details"):
                    st.json(schema_data, expanded=False)
            else:
                st.error(f"❌ {schema['error']}")
    
    st.divider()
    
    # Example queries
    st.subheader("💡 Try These:")
    examples = [
        "Show total sales",
        "Sales by category last month",
        "Compare profit across regions",
        "Order distribution by quantity",
        "List all products with prices"
    ]
    
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.user_input = ex

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! I can help you query your database. What would you like to know?"
    })

# Show chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask about your data..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process with agent
    with st.chat_message("assistant"):
        with st.spinner("Processing with LangChain Agent..."):
            try:
                agent = init_agent()
                
                # Step 1: Show intent classification (optional - you can remove if you want only agent)
                try:
                    from intent_classifier import IntentClassifier
                    classifier = IntentClassifier()
                    intent_result = classifier.classify(prompt)
                    
                    with st.expander("🎯 View Intent Analysis"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Intent", intent_result["intent"])
                        with col2:
                            st.metric("Metrics", len(intent_result["metrics"]))
                        with col3:
                            st.metric("Dimensions", len(intent_result["dimensions"]))
                        with col4:
                            st.metric("Time Range", intent_result["time_range"] or "None")
                except Exception as e:
                    st.info("Note: Intent analysis skipped due to: " + str(e))
                
                # NEW: Show guardrail status
                try:
                    from query_guardrail import QueryGuardrail
                    guardrail = QueryGuardrail()
                    guardrail_result = guardrail.validate_query(prompt, intent_result)
                    
                    with st.expander("🛡️ View Guardrail Check"):
                        if guardrail_result["valid"]:
                            st.success("✅ Query passed all safety checks")
                            st.write(f"**Reason:** {guardrail_result['reason']}")
                        else:
                            st.error("❌ Query blocked by guardrail")
                            st.write(f"**Reason:** {guardrail_result['reason']}")
                            st.write(f"**Suggestion:** {guardrail_result['suggested_action']}")
                            # Don't proceed if guardrail fails
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"🛡️ **Query Validation Failed**\n\n**Reason:** {guardrail_result['reason']}\n\n**Suggested Action:** {guardrail_result['suggested_action']}"
                            })
                            st.stop()  # Skip further processing
                except Exception as e:
                    st.info(f"Note: Guardrail check skipped: {e}")
                
                # Step 2: Use LangChain agent for the main processing
                result = agent.process_query(prompt)
                
                if result["status"] == "success":
                    # Show SQL
                    st.subheader("📝 Generated SQL")
                    st.code(result["sql"], language="sql")
                    
                    # Show data summary
                    if "row_count" in result:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows Retrieved", result["row_count"])
                        with col2:
                            if "columns" in result:
                                st.metric("Columns", len(result["columns"]))
                        with col3:
                            if "data" in result and result["data"]:
                                # Show first value if available
                                if result["data"] and len(result["data"]) > 0:
                                    first_item = result["data"][0]
                                    if isinstance(first_item, dict):
                                        first_key = list(first_item.keys())[0] if first_item else ""
                                        first_val = first_item.get(first_key, "") if first_key else ""
                                        st.metric("Sample", f"{first_val}"[:20])
                    
                    # Show data table (first 10 rows)
                    if "data" in result and result["data"]:
                        st.subheader("📋 Query Results")
                        st.dataframe(result["data"], use_container_width=True)
                    
                    # Show insights
                    if "insights" in result and result["insights"]:
                        st.subheader("💡 Insights")
                        st.markdown(result["insights"])
                    
                    # Show chart if available
                    if "chart" in result and "image_base64" in result["chart"]:
                        st.subheader("📈 Visualization")
                        try:
                            # Try to display the chart image
                            st.image(f"data:image/png;base64,{result['chart']['image_base64']}")
                        except:
                            st.info("Chart available but could not be displayed")
                    
                    # Show the complete answer
                    st.subheader("📝 Complete Analysis")
                    st.markdown(result["response"])
                    
                    # Add to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["response"]
                    })
                    
                    # Optionally show intermediate steps (if available)
                    if result.get('intermediate_steps'):
                        with st.expander("🔧 View Agent Steps"):
                            st.write("**LangChain Agent Execution Steps:**")
                            for i, step in enumerate(result['intermediate_steps'], 1):
                                st.write(f"**Step {i}:** {step[0].tool}")
                                if len(str(step[1])) > 300:
                                    st.text(f"Result: {str(step[1])[:300]}...")
                                else:
                                    st.text(f"Result: {step[1]}")
                    
                else:
                    st.error(f"Error: {result.get('response', 'Unknown error')}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Sorry, I encountered an error: {result.get('response')}"
                    })
                    
            except Exception as e:
                st.error(f"System error: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"System error: {str(e)}"
                })
st.divider()
st.caption("Dynamic SQL Generator | LangChain | Ollama LLM | Auto-Schema Extraction | Data Analysis | Insights | Visualization")
















##worked just changed to used the updated schema.
#"""
#Main Streamlit Chatbot
#"""
#import streamlit as st
#import json
#import os
#from langchain_agent import LangChainAgent
#from schema_extractor import SchemaExtractor
#from dotenv import load_dotenv
#
#load_dotenv()
#
## Page config
#st.set_page_config(
#    page_title="Dynamic SQL Chatbot",
#    page_icon="🤖",
#    layout="wide"
#)
#
#st.title("🤖 Dynamic SQL Chatbot with LangChain")
#st.markdown("Ask questions → Get SQL → See results → Get insights → Visualize")
#
## Initialize components
#@st.cache_resource
#def init_agent():
#    return LangChainAgent()
#
#@st.cache_resource  
#def init_schema():
#    return SchemaExtractor()
#
## Sidebar
#with st.sidebar:
#    st.header("🔧 Setup")
#    
#    # Extract schema button
#    if st.button("🔄 Extract Database Schema"):
#        with st.spinner("Extracting schema..."):
#            extractor = init_schema()
#            schema = extractor.extract_schema()
#            
#            if "error" not in schema:
#                st.success("✅ Schema extracted!")
#                
#                # Show schema file
#                with open("schema_info.json", "r") as f:
#                    schema_data = json.load(f)
#                
#                st.write(f"**Database:** {schema_data['database']}")
#                st.write(f"**Tables:** {len(schema_data['tables'])}")
#                
#                # Let user view schema
#                with st.expander("📋 View Schema Details"):
#                    st.json(schema_data, expanded=False)
#            else:
#                st.error(f"❌ {schema['error']}")
#    
#    st.divider()
#    
#    # Example queries
#    st.subheader("💡 Try These:")
#    examples = [
#        "Show total sales",
#        "Sales by category last month",
#        "Compare profit across regions",
#        "Order distribution by quantity",
#        "List all products with prices"
#    ]
#    
#    for ex in examples:
#        if st.button(ex, use_container_width=True):
#            st.session_state.user_input = ex
#
## Chat interface
#if "messages" not in st.session_state:
#    st.session_state.messages = []
#    st.session_state.messages.append({
#        "role": "assistant",
#        "content": "Hi! I can help you query your database. What would you like to know?"
#    })
#
## Show chat
#for msg in st.session_state.messages:
#    with st.chat_message(msg["role"]):
#        st.markdown(msg["content"])
#
## User input
#if prompt := st.chat_input("Ask about your data..."):
#    # Add user message
#    st.session_state.messages.append({"role": "user", "content": prompt})
#    
#    # Show user message
#    with st.chat_message("user"):
#        st.markdown(prompt)
#    
#    # Process with agent
#    with st.chat_message("assistant"):
#        with st.spinner("Processing with LangChain Agent..."):
#            try:
#                agent = init_agent()
#                
#                # Step 1: Show intent classification (optional - you can remove if you want only agent)
#                try:
#                    from intent_classifier import IntentClassifier
#                    classifier = IntentClassifier()
#                    intent_result = classifier.classify(prompt)
#                    
#                    with st.expander("🎯 View Intent Analysis"):
#                        col1, col2, col3, col4 = st.columns(4)
#                        with col1:
#                            st.metric("Intent", intent_result["intent"])
#                        with col2:
#                            st.metric("Metrics", len(intent_result["metrics"]))
#                        with col3:
#                            st.metric("Dimensions", len(intent_result["dimensions"]))
#                        with col4:
#                            st.metric("Time Range", intent_result["time_range"] or "None")
#                except Exception as e:
#                    st.info("Note: Intent analysis skipped due to: " + str(e))
#                
#                # Step 2: Use LangChain agent for the main processing
#                result = agent.process_query(prompt)
#                
#                if result["status"] == "success":
#                    # Show SQL
#                    st.subheader("📝 Generated SQL")
#                    st.code(result["sql"], language="sql")
#                    
#                    # Show data summary
#                    if "row_count" in result:
#                        col1, col2, col3 = st.columns(3)
#                        with col1:
#                            st.metric("Rows Retrieved", result["row_count"])
#                        with col2:
#                            if "columns" in result:
#                                st.metric("Columns", len(result["columns"]))
#                        with col3:
#                            if "data" in result and result["data"]:
#                                # Show first value if available
#                                if result["data"] and len(result["data"]) > 0:
#                                    first_item = result["data"][0]
#                                    if isinstance(first_item, dict):
#                                        first_key = list(first_item.keys())[0] if first_item else ""
#                                        first_val = first_item.get(first_key, "") if first_key else ""
#                                        st.metric("Sample", f"{first_val}"[:20])
#                    
#                    # Show data table (first 10 rows)
#                    if "data" in result and result["data"]:
#                        st.subheader("📋 Query Results")
#                        st.dataframe(result["data"], use_container_width=True)
#                    
#                    # Show insights
#                    if "insights" in result and result["insights"]:
#                        st.subheader("💡 Insights")
#                        st.markdown(result["insights"])
#                    
#                    # Show chart if available
#                    if "chart" in result and "image_base64" in result["chart"]:
#                        st.subheader("📈 Visualization")
#                        try:
#                            # Try to display the chart image
#                            st.image(f"data:image/png;base64,{result['chart']['image_base64']}")
#                        except:
#                            st.info("Chart available but could not be displayed")
#                    
#                    # Show the complete answer
#                    st.subheader("📝 Complete Analysis")
#                    st.markdown(result["response"])
#                    
#                    # Add to chat
#                    st.session_state.messages.append({
#                        "role": "assistant",
#                        "content": result["response"]
#                    })
#                    
#                    # Optionally show intermediate steps (if available)
#                    if result.get('intermediate_steps'):
#                        with st.expander("🔧 View Agent Steps"):
#                            st.write("**LangChain Agent Execution Steps:**")
#                            for i, step in enumerate(result['intermediate_steps'], 1):
#                                st.write(f"**Step {i}:** {step[0].tool}")
#                                if len(str(step[1])) > 300:
#                                    st.text(f"Result: {str(step[1])[:300]}...")
#                                else:
#                                    st.text(f"Result: {step[1]}")
#                    
#                else:
#                    st.error(f"Error: {result.get('response', 'Unknown error')}")
#                    st.session_state.messages.append({
#                        "role": "assistant",
#                        "content": f"Sorry, I encountered an error: {result.get('response')}"
#                    })
#                    
#            except Exception as e:
#                st.error(f"System error: {str(e)}")
#                st.session_state.messages.append({
#                    "role": "assistant",
#                    "content": f"System error: {str(e)}"
#                })
#
## Footer
#st.divider()
#st.caption("Dynamic SQL Generator | LangChain | Ollama LLM | Auto-Schema Extraction | Data Analysis | Insights | Visualization")











## worked before we added data analyzer changed after adding analyzer part.
#"""
#Main Streamlit Chatbot
#"""
#import streamlit as st
#import json
#import os
#from langchain_agent import LangChainAgent
#from schema_extractor import SchemaExtractor
#from dotenv import load_dotenv
#
#load_dotenv()
#
## Page config
#st.set_page_config(
#    page_title="Dynamic SQL Chatbot",
#    page_icon="🤖",
#    layout="wide"
#)
#
#st.title("🤖 Dynamic SQL Chatbot with LangChain")
#st.markdown("Ask questions → Get SQL → See results")
#
## Initialize components
#@st.cache_resource
#def init_agent():
#    return LangChainAgent()
#
#@st.cache_resource  
#def init_schema():
#    return SchemaExtractor()
#
## Sidebar
#with st.sidebar:
#    st.header("🔧 Setup")
#    
#    # Extract schema button
#    if st.button("🔄 Extract Database Schema"):
#        with st.spinner("Extracting schema..."):
#            extractor = init_schema()
#            schema = extractor.extract_schema()
#            
#            if "error" not in schema:
#                st.success("✅ Schema extracted!")
#                
#                # Show schema file
#                with open("schema_info.json", "r") as f:
#                    schema_data = json.load(f)
#                
#                st.write(f"**Database:** {schema_data['database']}")
#                st.write(f"**Tables:** {len(schema_data['tables'])}")
#                
#                # Let user view schema
#                with st.expander("📋 View Schema Details"):
#                    st.json(schema_data, expanded=False)
#            else:
#                st.error(f"❌ {schema['error']}")
#    
#    st.divider()
#    
#    # Example queries
#    st.subheader("💡 Try These:")
#    examples = [
#        "Show total sales",
#        "Sales by category last month",
#        "Compare profit across regions",
#        "Order distribution by quantity",
#        "List all products with prices"
#    ]
#    
#    for ex in examples:
#        if st.button(ex, use_container_width=True):
#            st.session_state.user_input = ex
#
## Chat interface
#if "messages" not in st.session_state:
#    st.session_state.messages = []
#    st.session_state.messages.append({
#        "role": "assistant",
#        "content": "Hi! I can help you query your database. What would you like to know?"
#    })
#
## Show chat
#for msg in st.session_state.messages:
#    with st.chat_message(msg["role"]):
#        st.markdown(msg["content"])
#
## User input
#if prompt := st.chat_input("Ask about your data..."):
#    # Add user message
#    st.session_state.messages.append({"role": "user", "content": prompt})
#    
#    # Show user message
#    with st.chat_message("user"):
#        st.markdown(prompt)
#    
#    # Process with agent
#    with st.chat_message("assistant"):
#        with st.spinner("Processing with LangChain Agent..."):
#            try:
#                agent = init_agent()
#                
#                # Step 1: Show intent classification (optional - you can remove if you want only agent)
#                try:
#                    from intent_classifier import IntentClassifier
#                    classifier = IntentClassifier()
#                    intent_result = classifier.classify(prompt)
#                    
#                    with st.expander("🎯 View Intent Analysis"):
#                        col1, col2, col3, col4 = st.columns(4)
#                        with col1:
#                            st.metric("Intent", intent_result["intent"])
#                        with col2:
#                            st.metric("Metrics", len(intent_result["metrics"]))
#                        with col3:
#                            st.metric("Dimensions", len(intent_result["dimensions"]))
#                        with col4:
#                            st.metric("Time Range", intent_result["time_range"] or "None")
#                except Exception as e:
#                    st.info("Note: Intent analysis skipped due to: " + str(e))
#                
#                # Step 2: Use LangChain agent for the main processing
#                result = agent.process_query(prompt)
#                
#                if result["status"] == "success":
#                    # Show full response
#                    st.write(result["response"])
#                    
#                    # Optionally show intermediate steps
#                    if result.get('intermediate_steps'):
#                        with st.expander("🔧 View Agent Steps"):
#                            st.write("**LangChain Agent Execution Steps:**")
#                            for i, step in enumerate(result['intermediate_steps'], 1):
#                                st.write(f"**Step {i}:** {step[0].tool}")
#                                if len(str(step[1])) > 300:
#                                    st.text(f"Result: {str(step[1])[:300]}...")
#                                else:
#                                    st.text(f"Result: {step[1]}")
#                    
#                    # Add to chat
#                    st.session_state.messages.append({
#                        "role": "assistant",
#                        "content": result["response"]
#                    })
#                else:
#                    st.error(f"Error: {result.get('response', 'Unknown error')}")
#                    st.session_state.messages.append({
#                        "role": "assistant",
#                        "content": f"Sorry, I encountered an error: {result.get('response')}"
#                    })
#                    
#            except Exception as e:
#                st.error(f"System error: {str(e)}")
#                st.session_state.messages.append({
#                    "role": "assistant",
#                    "content": f"System error: {str(e)}"
#                })
#
## Footer
#st.divider()
#st.caption("Dynamic SQL Generator | LangChain | Ollama LLM | Auto-Schema Extraction")
















#"""
#Main Streamlit Chatbot
#"""
#import streamlit as st
#import json
#import os
#from langchain_agent import LangChainAgent
#from schema_extractor import SchemaExtractor
#from dotenv import load_dotenv
#
#load_dotenv()
#
## Page config
#st.set_page_config(
#    page_title="Dynamic SQL Chatbot",
#    page_icon="🤖",
#    layout="wide"
#)
#
#st.title("🤖 Dynamic SQL Chatbot with LangChain")
#st.markdown("Ask questions → Get SQL → See results")
#
## Initialize components
#@st.cache_resource
#def init_agent():
#    return LangChainAgent()
#
#@st.cache_resource  
#def init_schema():
#    return SchemaExtractor()
#
## Sidebar
#with st.sidebar:
#    st.header("🔧 Setup")
#    
#    # Extract schema button
#    if st.button("🔄 Extract Database Schema"):
#        with st.spinner("Extracting schema..."):
#            extractor = init_schema()
#            schema = extractor.extract_schema()
#            
#            if "error" not in schema:
#                st.success("✅ Schema extracted!")
#                
#                # Show schema file
#                with open("schema_info.json", "r") as f:
#                    schema_data = json.load(f)
#                
#                st.write(f"**Database:** {schema_data['database']}")
#                st.write(f"**Tables:** {len(schema_data['tables'])}")
#                
#                # Let user view schema
#                with st.expander("📋 View Schema Details"):
#                    st.json(schema_data, expanded=False)
#            else:
#                st.error(f"❌ {schema['error']}")
#    
#    st.divider()
#    
#    # Example queries
#    st.subheader("💡 Try These:")
#    examples = [
#        "Show total sales",
#        "Sales by category last month",
#        "Compare profit across regions",
#        "Order distribution by quantity",
#        "List all products with prices"
#    ]
#    
#    for ex in examples:
#        if st.button(ex, use_container_width=True):
#            st.session_state.user_input = ex
#
## Chat interface
#if "messages" not in st.session_state:
#    st.session_state.messages = []
#    st.session_state.messages.append({
#        "role": "assistant",
#        "content": "Hi! I can help you query your database. What would you like to know?"
#    })
#
## Show chat
#for msg in st.session_state.messages:
#    with st.chat_message(msg["role"]):
#        st.markdown(msg["content"])
#
## User input
#if prompt := st.chat_input("Ask about your data..."):
#    # Add user message
#    st.session_state.messages.append({"role": "user", "content": prompt})
#    
#    # Show user message
#    with st.chat_message("user"):
#        st.markdown(prompt)
#    
#    # Process with agent
#    with st.chat_message("assistant"):
#        with st.spinner("Processing..."):
#            try:
#                agent = init_agent()
#                
#                # Step 1: Show intent classification
#                from intent_classifier import IntentClassifier
#                classifier = IntentClassifier()
#                intent_result = classifier.classify(prompt)
#                
#                st.subheader("🎯 Intent Analysis")
#                col1, col2, col3, col4 = st.columns(4)
#                with col1:
#                    st.metric("Intent", intent_result["intent"])
#                with col2:
#                    st.metric("Metrics", len(intent_result["metrics"]))
#                with col3:
#                    st.metric("Dimensions", len(intent_result["dimensions"]))
#                with col4:
#                    st.metric("Time Range", intent_result["time_range"] or "None")
#                
#                # Step 2: Generate and show SQL
#                st.subheader("📝 Generated SQL")
#                result = agent.process_query(prompt)
#                
#                if result["status"] == "success":
#                    # Show response
#                    st.write(result["response"])
#                    
#                    # Add to chat
#                    st.session_state.messages.append({
#                        "role": "assistant",
#                        "content": result["response"][:500] + "..." if len(result["response"]) > 500 else result["response"]
#                    })
#                else:
#                    st.error(f"Error: {result.get('error', 'Unknown error')}")
#                    st.session_state.messages.append({
#                        "role": "assistant",
#                        "content": f"Sorry, I encountered an error: {result.get('error')}"
#                    })
#                    
#            except Exception as e:
#                st.error(f"System error: {str(e)}")
#                st.session_state.messages.append({
#                    "role": "assistant",
#                    "content": f"System error: {str(e)}"
#                })
#
## Footer
#st.divider()
#st.caption("Dynamic SQL Generator | LangChain | Ollama LLM | Auto-Schema Extraction")





#"""
#Main Chatbot - Uses AUTO schema extraction
#"""
#import streamlit as st
#import pandas as pd
#import plotly.express as px
#from database import Database
#from intent_classifier import classify_intent
#from ai_sql_generator import AISQLGenerator
#from schema_extractor import SchemaExtractor
#
## Initialize
#db = Database()
#sql_generator = AISQLGenerator(use_openai=False)  # Set True if you have OpenAI
#
## Page config
#st.set_page_config(
#    page_title="Auto SQL Generator",
#    page_icon="🤖",
#    layout="wide"
#)
#
#st.title("🤖 Auto SQL Chatbot")
#st.markdown("Ask questions → Get SQL → See results")
#
## Sidebar
#with st.sidebar:
#    st.header("Database Info")
#    
#    # Auto-extract schema button
#    if st.button("🔄 Extract Schema"):
#        extractor = SchemaExtractor()
#        schema = extractor.extract_schema()
#        
#        st.success(f"✅ Schema extracted!")
#        st.write(f"**Database:** {schema['database_name']}")
#        st.write(f"**Tables:** {len(schema['tables'])}")
#        
#        for table_name in schema['tables'].keys():
#            st.write(f"- {table_name}")
#
## Chat interface
#if "messages" not in st.session_state:
#    st.session_state.messages = []
#    st.session_state.messages.append({
#        "role": "assistant",
#        "content": "Hi! I can write SQL queries for you. Ask about your data!"
#    })
#
## Show chat
#for message in st.session_state.messages:
#    with st.chat_message(message["role"]):
#        st.markdown(message["content"])
#
## User input
#if prompt := st.chat_input("Ask about your data..."):
#    # Add user message
#    st.session_state.messages.append({"role": "user", "content": prompt})
#    
#    # Show user message
#    with st.chat_message("user"):
#        st.markdown(prompt)
#    
#    # Process
#    with st.chat_message("assistant"):
#        with st.spinner("Thinking..."):
#            
#            # 1. Classify intent
#            intent = classify_intent(prompt)
#            st.write(f"**Intent:** {intent}")
#            
#            # 2. Generate SQL
#            sql = sql_generator.generate_sql(prompt, intent)
#            
#            # Show SQL
#            with st.expander("📝 Generated SQL", expanded=True):
#                st.code(sql, language="sql")
#            
#            # 3. Execute
#            df = db.run_query(sql)
#            
#            if df is not None and not df.empty:
#                st.success(f"✅ Found {len(df)} rows")
#                
#                # Show data
#                with st.expander("📊 Data Preview"):
#                    st.dataframe(df)
#                
#                # Create chart
#                if len(df.columns) >= 2:
#                    fig = px.bar(df, x=df.columns[0], y=df.columns[1])
#                    st.plotly_chart(fig)
#                
#                response = f"Found {len(df)} results"
#            else:
#                response = "No data found"
#                st.warning(response)
#            
#            # Add to chat
#            st.session_state.messages.append({"role": "assistant", "content": response})









#"""
#Superstore Chatbot - SIMPLE VERSION
#"""
#import streamlit as st
#import pandas as pd
#import plotly.express as px
#from database import Database
#
## Initialize database (it will read from .env automatically)
#db = Database()
#
## Set page configuration
#st.set_page_config(
#    page_title="Superstore Chatbot",
#    page_icon="📊",
#    layout="wide"
#)
#
## Title
#st.title("📊 Superstore Data Chatbot")
#st.markdown("Ask questions about your Superstore data!")
#
## Sidebar
#with st.sidebar:
#    st.header("Database Connection")
#    
#    # Show current connection info
#    st.info(f"""
#    **Connection Details:**
#    - Host: {db.host}:{db.port}
#    - User: {db.user}
#    - Database: {db.database}
#    """)
#    
#    # Test connection button
#    if st.button("Test Connection"):
#        if db.connect():
#            st.success("✅ Connected successfully!")
#            columns = db.get_table_columns()
#            st.write(f"**Table 'store' has {len(columns)} columns**")
#        else:
#            st.error("❌ Connection failed")
#
## Initialize chat history
#if "messages" not in st.session_state:
#    st.session_state.messages = []
#    st.session_state.messages.append({
#        "role": "assistant",
#        "content": "Hello! I can help you analyze Superstore data. Try asking about sales, profits, or categories!"
#    })
#
## Display chat history
#for message in st.session_state.messages:
#    with st.chat_message(message["role"]):
#        st.markdown(message["content"])
#
## Chat input
#prompt = st.chat_input("Ask a question about the data...")
#
#if prompt:
#    # Add user message to chat
#    st.session_state.messages.append({"role": "user", "content": prompt})
#    
#    # Display user message
#    with st.chat_message("user"):
#        st.markdown(prompt)
#    
#    # Process the query
#    with st.chat_message("assistant"):
#        with st.spinner("Analyzing..."):
#            
#            # SIMPLE RULES (we'll improve this later)
#            user_query = prompt.lower()
#            sql = ""
#            
#            if "sales" in user_query and "category" in user_query:
#                sql = "SELECT Category, SUM(Sales) as Total_Sales FROM store GROUP BY Category ORDER BY Total_Sales DESC"
#                title = "Sales by Category"
#                chart_type = "bar"
#                
#            elif "profit" in user_query and "region" in user_query:
#                sql = "SELECT Region, SUM(Profit) as Total_Profit FROM store GROUP BY Region ORDER BY Total_Profit DESC"
#                title = "Profit by Region"
#                chart_type = "bar"
#                
#            elif "total sales" in user_query:
#                sql = "SELECT SUM(Sales) as Total_Sales FROM store"
#                title = "Total Sales"
#                chart_type = "number"
#                
#            elif "top" in user_query and "product" in user_query:
#                sql = "SELECT Product_Name, SUM(Sales) as Total_Sales FROM store GROUP BY Product_Name ORDER BY Total_Sales DESC LIMIT 10"
#                title = "Top Products by Sales"
#                chart_type = "bar"
#                
#            else:
#                # Default
#                sql = "SELECT Category, SUM(Sales) as Sales FROM store GROUP BY Category"
#                title = "Sales by Category"
#                chart_type = "bar"
#            
#            # Show SQL
#            with st.expander("🔍 Generated SQL"):
#                st.code(sql, language="sql")
#            
#            # Execute query
#            df = db.run_query(sql)
#            
#            if df is not None and not df.empty:
#                st.success(f"✅ Found {len(df)} results")
#                
#                # Show data
#                with st.expander("📊 View Data"):
#                    st.dataframe(df)
#                
#                # Create chart
#                if chart_type == "bar" and len(df.columns) >= 2:
#                    fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=title)
#                    st.plotly_chart(fig, use_container_width=True)
#                
#                elif chart_type == "number":
#                    value = df.iloc[0, 0]
#                    st.metric(label=title, value=f"${value:,.2f}")
#                
#                response = f"I analyzed your query and found {len(df)} results."
#                
#            else:
#                response = "No data found for that query."
#                st.warning(response)
#            
#            # Add to chat
#            st.session_state.messages.append({"role": "assistant", "content": response})
#
## Footer
#st.divider()
#st.caption(f"Connected to {db.database} database | Using {db.user} account")


















#"""
#Superstore Chatbot - SIMPLE VERSION FOR BEGINNERS
#"""
#import streamlit as st
#import pandas as pd
#import plotly.express as px
##from database import Database
#from database import DatabaseManager
#
#
#db = DatabaseManager()
#
#
## Page configuration
#st.set_page_config(
#    page_title="Superstore Chatbot",
#    page_icon="📊",
#    layout="wide"
#)
#
## Title
#st.title("🏪 Superstore Data Chatbot")
#st.markdown("Ask questions about your store data in plain English!")
#
## Sidebar for database info
#with st.sidebar:
#    st.header("🔧 Database Connection")
#    
#    # Test connection button
#    if st.button("Test Database Connection"):
#        db = Database()
#        if db.connect():
#            st.success("✅ Connected to database!")
#            
#            # Show table info
#            columns = db.get_table_info()
#            if columns:
#                st.write("**Table Columns:**")
#                for col in columns:
#                    st.write(f"- {col[0]} ({col[1]})")
#        else:
#            st.error("❌ Connection failed")
#    
#    st.divider()
#    
#    # Example questions
#    st.header("💡 Try These Questions:")
#    examples = [
#        "Show sales by category",
#        "What are total profits?",
#        "Sales by region",
#        "Top 10 products by sales",
#        "Profit trend over time"
#    ]
#    
#    for example in examples:
#        if st.button(example, use_container_width=True):
#            st.session_state.user_query = example
#
## Initialize chat history
#if "messages" not in st.session_state:
#    st.session_state.messages = []
#    st.session_state.messages.append({
#        "role": "assistant",
#        "content": "Hi! I'm your Superstore data assistant. Ask me anything about sales, profits, categories, or regions!"
#    })
#
## Display chat history
#for message in st.session_state.messages:
#    with st.chat_message(message["role"]):
#        st.markdown(message["content"])
#
## Chat input
#if prompt := st.chat_input("Type your question here..."):
#    # Add user message to chat
#    st.session_state.messages.append({"role": "user", "content": prompt})
#    
#    # Display user message
#    with st.chat_message("user"):
#        st.markdown(prompt)
#    
#    # Process the query
#    with st.chat_message("assistant"):
#        with st.spinner("Thinking..."):
#            
#            # SIMPLE RULE-BASED SQL GENERATION
#            # (We'll make this smarter with AI later)
#            query_lower = prompt.lower()
#            sql = ""
#            
#            if "sales" in query_lower and "category" in query_lower:
#                sql = """
#                SELECT Category, SUM(Sales) as Total_Sales 
#                FROM store 
#                GROUP BY Category 
#                ORDER BY Total_Sales DESC
#                """
#                chart_type = "bar"
#                
#            elif "profit" in query_lower and "region" in query_lower:
#                sql = """
#                SELECT Region, SUM(Profit) as Total_Profit 
#                FROM store 
#                GROUP BY Region 
#                ORDER BY Total_Profit DESC
#                """
#                chart_type = "bar"
#                
#            elif "total sales" in query_lower or "overall sales" in query_lower:
#                sql = "SELECT SUM(Sales) as Total_Sales FROM store"
#                chart_type = "number"
#                
#            elif "top" in query_lower and "product" in query_lower:
#                if "10" in query_lower:
#                    limit = 10
#                elif "5" in query_lower:
#                    limit = 5
#                else:
#                    limit = 10
#                    
#                sql = f"""
#                SELECT Product_Name, SUM(Sales) as Total_Sales 
#                FROM store 
#                GROUP BY Product_Name 
#                ORDER BY Total_Sales DESC 
#                LIMIT {limit}
#                """
#                chart_type = "bar"
#                
#            elif "trend" in query_lower or "over time" in query_lower:
#                sql = """
#                SELECT DATE(Order_Date) as Date, SUM(Sales) as Daily_Sales 
#                FROM store 
#                GROUP BY DATE(Order_Date) 
#                ORDER BY Date
#                """
#                chart_type = "line"
#                
#            else:
#                # Default query
#                sql = """
#                SELECT Category, Region, 
#                       SUM(Sales) as Sales, 
#                       SUM(Profit) as Profit 
#                FROM store 
#                GROUP BY Category, Region
#                """
#                chart_type = "bar"
#            
#            # Display the SQL
#            with st.expander("📝 Generated SQL"):
#                st.code(sql, language="sql")
#            
#            # Execute query
#            try:
#                db = Database()
#                db.connect()
#                df = db.run_query(sql)
#                
#                if df is not None and not df.empty:
#                    # Show data
#                    st.success(f"✅ Found {len(df)} results")
#                    
#                    with st.expander("📊 View Data", expanded=True):
#                        st.dataframe(df, use_container_width=True)
#                    
#                    # Create chart
#                    st.subheader("📈 Visualization")
#                    
#                    if chart_type == "bar" and len(df) > 0:
#                        # Find numeric column for y-axis
#                        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
#                        if len(numeric_cols) > 0:
#                            y_col = numeric_cols[0]
#                            x_col = df.columns[0]
#                            
#                            fig = px.bar(df, x=x_col, y=y_col, 
#                                       title=f"{y_col.replace('_', ' ').title()} by {x_col}")
#                            st.plotly_chart(fig, use_container_width=True)
#                    
#                    elif chart_type == "line" and len(df) > 0:
#                        if len(df.columns) >= 2:
#                            fig = px.line(df, x=df.columns[0], y=df.columns[1],
#                                        title=f"{df.columns[1].replace('_', ' ').title()} Trend")
#                            st.plotly_chart(fig, use_container_width=True)
#                    
#                    elif chart_type == "number":
#                        value = df.iloc[0, 0]
#                        st.metric(label="Total Sales", value=f"${value:,.2f}")
#                    
#                    # Simple insights
#                    st.subheader("💡 Quick Insights")
#                    if len(df) > 1:
#                        top_item = df.iloc[0]
#                        st.write(f"• **Top performer:** {top_item[df.columns[0]]} with ${top_item[df.columns[1]]:,.2f}")
#                    
#                    total = df.iloc[:, 1].sum() if len(df.columns) > 1 else df.iloc[0, 0]
#                    st.write(f"• **Total:** ${total:,.2f}")
#                    
#                    response = f"I found {len(df)} results for your query about '{prompt}'"
#                    
#                else:
#                    response = "I couldn't find any data for that query. Try asking differently."
#                    st.warning(response)
#                    
#            except Exception as e:
#                response = f"Sorry, I encountered an error: {str(e)}"
#                st.error(response)
#            
#            # Add assistant response
#            st.session_state.messages.append({"role": "assistant", "content": response})
#
## Footer
#st.divider()
#st.caption("Superstore Chatbot v1.0 | Connected to MySQL Database")