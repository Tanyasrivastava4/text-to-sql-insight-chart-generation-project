"""
COMPLETE LangChain SQL Agent with Data Analysis Pipeline
"""

import os
import mysql.connector
from typing import Any, Dict, List
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from schema_extractor import SchemaExtractor
from intent_classifier import IntentClassifier
from sql_validator import SQLValidator

# Add data analysis imports
try:
    from data_analyzer import DataAnalyzer
    from insight_generator import InsightGenerator
    from chart_generator import ChartGenerator
    DATA_ANALYSIS_AVAILABLE = True
    print("✅ Data analysis modules loaded")
except ImportError as e:
    print(f"⚠️ Data analysis modules not available: {e}")
    print("⚠️ Install: pip install matplotlib numpy pandas")
    DATA_ANALYSIS_AVAILABLE = False
    
    # Create dummy classes
    class DataAnalyzer:
        def analyze(self, data, intent, columns):
            return {
                "original_data": data,
                "columns": columns,
                "row_count": len(data),
                "analysis": {},
                "chart_data": {},
                "chart_type": "bar"
            }
    
    class InsightGenerator:
        def generate_insights(self, analysis_result, original_query, intent):
            return "## 📊 Key Insights\n\n*Install data_analyzer.py and insight_generator.py*"
    
    class ChartGenerator:
        def generate_chart(self, analysis_result):
            return {"error": "Install matplotlib: pip install matplotlib"}

load_dotenv()


class LangChainAgent:
    def __init__(self, refresh_schema: bool = False):
        print(f"🔧 Initializing LangChainAgent...")
        
        # Load schema dynamically with refresh option
        self.schema_extractor = SchemaExtractor()
        
        if refresh_schema:
            # Force refresh
            self.schema_extractor.refresh_schema(force=True)
            self.schema_text = self.schema_extractor.get_formatted_schema()
        else:
            # Use cached if recent, refresh if old
            self.schema_extractor.refresh_schema(force=False)
            self.schema_text = self.schema_extractor.get_formatted_schema()
        
        # Initialize intent classifier
        self.intent_classifier = IntentClassifier()
        
        # Initialize SQL validator
        self.sql_validator = SQLValidator()

        # Initialize Ollama LLM
        self.llm = ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://62.171.149.65:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            temperature=0,
            timeout=120
        )

        # DB config
        self.db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", 3307)),
            "user": os.getenv("DB_USER", "root"),
            "password": os.getenv("DB_PASSWORD", ""),
            "database": os.getenv("DB_NAME", "")
        }
        
        print(f"📡 Using Ollama: {os.getenv('OLLAMA_BASE_URL')}")
        print(f"🤖 Model: {os.getenv('OLLAMA_MODEL')}")
        print("✅ LangChainAgent initialized!")

    # -----------------------------
    # Database Connection
    # -----------------------------
    def connect_db(self):
        return mysql.connector.connect(**self.db_config)

    # -----------------------------
    # STEP 1: Generate SQL
    # -----------------------------
    def generate_sql(self, user_question: str, intent: Dict = None) -> str:
        system_prompt = f"""
You are a senior data analyst.

Database Schema:
{self.schema_text}

Rules:
- Generate ONLY valid MySQL SELECT queries
- Do NOT explain anything
- Do NOT add markdown or backticks
- Use proper table and column names from schema
- Include aggregations (SUM, COUNT, AVG) when appropriate
- Add GROUP BY for dimensional analysis
- Add ORDER BY for meaningful sorting
- Limit results if returning many rows

Important: Only use columns and tables that exist in the schema above.
"""

        # Include intent information if available
        if intent:
            intent_info = f"""
User Intent: {intent.get('intent', 'REPORT')}
Metrics requested: {', '.join(intent.get('metrics', []))}
Dimensions requested: {', '.join(intent.get('dimensions', []))}
Time range: {intent.get('time_range', 'None')}
"""
            system_prompt += f"\nAdditional Context:\n{intent_info}"

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_question)
            ])

            sql = response.content.strip()
            
            # Clean SQL (remove markdown if present)
            if sql.startswith("```sql"):
                sql = sql[6:]
            if sql.startswith("```"):
                sql = sql[3:]
            if sql.endswith("```"):
                sql = sql[:-3]
            
            # Ensure it's a SELECT query
            sql = sql.strip()
            if not sql.upper().startswith("SELECT"):
                print("⚠️ Generated non-SELECT query, adding SELECT * FROM store")
                return "SELECT * FROM store LIMIT 10"
            
            return sql
            
        except Exception as e:
            print(f"❌ SQL generation error: {e}")
            # Fallback to simple query
            return "SELECT * FROM store LIMIT 10"

    # -----------------------------
    # STEP 2: Execute SQL
    # -----------------------------
    def execute_sql(self, sql: str) -> Dict[str, Any]:
        try:
            conn = self.connect_db()
            cursor = conn.cursor(dictionary=True)
            
            print(f"📊 Executing SQL...")
            cursor.execute(sql)
            result = cursor.fetchall()
            
            # Get column names
            column_names = [desc[0] for desc in cursor.description] if cursor.description else []
            
            print(f"✅ Query executed successfully!")
            print(f"   Rows: {len(result)}, Columns: {len(column_names)}")
            
            return {
                "success": True,
                "row_count": len(result),
                "column_count": len(column_names),
                "columns": column_names,
                "data": result[:100],  # Limit to 100 rows for display
                "summary": f"Found {len(result)} rows with {len(column_names)} columns"
            }
        except mysql.connector.Error as e:
            print(f"❌ Database error: {e}")
            return {
                "success": False,
                "error": f"Database error: {str(e)}",
                "row_count": 0,
                "column_count": 0,
                "columns": [],
                "data": [],
                "summary": f"Error: {str(e)}"
            }
        except Exception as e:
            print(f"❌ Execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "row_count": 0,
                "column_count": 0,
                "columns": [],
                "data": [],
                "summary": f"Error: {str(e)}"
            }
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

    # -----------------------------
    # STEP 3: Generate Basic Answer (fallback)
    # -----------------------------
    def generate_answer(self, question: str, sql: str, result: Dict) -> str:
        if not result["success"]:
            return f"❌ Error: {result.get('error', 'Unknown error')}"
        
        if result["row_count"] == 0:
            return "📭 No data found for your query."
        
        prompt = f"""
User Question: {question}

SQL Used: {sql}

Query Results:
- Total rows: {result['row_count']}
- Columns: {', '.join(result['columns'])}
- Sample data (first 3 rows): {result['data'][:3]}

Provide a clear summary of what this data shows.
Highlight key numbers and patterns.
Keep it concise and business-friendly.
"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"❌ Answer generation error: {e}")
            return f"Query executed successfully. Found {result['row_count']} rows."

    # -----------------------------
    # MAIN CHAT FUNCTION - COMPLETE PIPELINE
    # -----------------------------
    def chat(self, user_question: str) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"🤖 Processing: {user_question}")
        print('='*60)
        
            # ✅ NEW: Check for database updates before processing
        print("🔍 Checking if database schema changed...")
        self.schema_extractor.refresh_schema(force=False)
        self.schema_text = self.schema_extractor.get_formatted_schema()
        # Step 1: Classify intent
        print("\n🎯 Classifying intent...")
        try:
            intent = self.intent_classifier.classify(user_question)
            print(f"✅ Intent: {intent['intent']}")
            print(f"   Metrics: {intent['metrics']}")
            print(f"   Dimensions: {intent['dimensions']}")
        except Exception as e:
            print(f"❌ Intent classification failed: {e}")
            intent = {"intent": "REPORT", "metrics": [], "dimensions": [], "time_range": None}
        
        # Step 2: Generate SQL
        print("\n🧠 Generating SQL...")
        sql = self.generate_sql(user_question, intent)
        print(f"✅ SQL: {sql}")
        
        # Step 3: Validate SQL
        print("\n🔒 Validating SQL...")
        validation_result = self.sql_validator.validate_sql(sql)
        print(f"✅ Validation: {validation_result}")
        
        if validation_result != "VALID_SQL":
            return {
                "question": user_question,
                "sql": sql,
                "result": {"success": False, "error": "SQL validation failed"},
                "answer": f"❌ SQL validation failed: {validation_result}",
                "intent": intent
            }
        
        # Step 4: Execute SQL
        print("\n📊 Executing SQL...")
        sql_result = self.execute_sql(sql)
        
        if not sql_result["success"]:
            return {
                "question": user_question,
                "sql": sql,
                "result": sql_result,
                "answer": f"❌ Query failed: {sql_result.get('error')}",
                "intent": intent
            }
        
        print(f"✅ Query successful! {sql_result['row_count']} rows retrieved")
        
        # Step 5-7: Data Analysis Pipeline (if available)
        insights = ""
        chart_result = {"error": "Chart generator not available"}
        analysis_result = None
        
        if DATA_ANALYSIS_AVAILABLE and sql_result["data"]:
            try:
                # Step 5: Data Analysis
                print("\n📈 Analyzing data...")
                data_analyzer = DataAnalyzer()
                analysis_result = data_analyzer.analyze(
                    data=sql_result["data"],
                    intent=intent,
                    columns=sql_result["columns"]
                )
                
                # Step 6: Generate Insights
                print("\n💡 Generating insights...")
                insight_generator = InsightGenerator()
                insights = insight_generator.generate_insights(
                    analysis_result=analysis_result,
                    original_query=user_question,
                    intent=intent
                )
                
                # Step 7: Generate Chart
                print("\n📊 Generating chart...")
                chart_generator = ChartGenerator()
                chart_result = chart_generator.generate_chart(analysis_result)
                
                if "image_base64" in chart_result:
                    print(f"✅ Chart generated: {chart_result.get('type', 'chart')}")
                else:
                    print(f"⚠️ Chart not generated: {chart_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"⚠️ Data analysis error: {e}")
                insights = f"## 📊 Key Insights\n\n*Analysis error: {str(e)}*"
        else:
            print("\n⚠️ Skipping data analysis (modules not available or no data)")
        
        # Step 8: Create final answer
        print("\n💬 Compiling final answer...")
        
        answer_parts = []
        
        # 1. SQL Query
        answer_parts.append(f"**📝 SQL Query Used:**\n```sql\n{sql}\n```")
        
        # 2. Data Summary
        answer_parts.append(f"**📊 Data Summary:**")
        answer_parts.append(f"- Rows retrieved: **{sql_result['row_count']}**")
        answer_parts.append(f"- Columns: {', '.join(sql_result['columns'])}")
        
        # 3. Show sample data if small
        if sql_result['row_count'] > 0 and sql_result['row_count'] <= 10:
            answer_parts.append(f"\n**Sample Data:**")
            for i, row in enumerate(sql_result['data'][:5], 1):
                row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
                answer_parts.append(f"{i}. {row_str}")
        
        # 4. Insights
        if insights:
            answer_parts.append(insights)
        
        # 5. Chart info
        if "image_base64" in chart_result:
            answer_parts.append(f"**📈 Visualization:** {chart_result.get('type', 'Chart')} generated")
        elif "error" in chart_result:
            answer_parts.append(f"**📈 Visualization:** {chart_result['error']}")
        
        final_answer = "\n\n".join(answer_parts)
        
        print(f"✅ Answer compiled ({len(final_answer)} characters)")
        
        return {
            "question": user_question,
            "sql": sql,
            "result": sql_result,
            "analysis": analysis_result,
            "insights": insights,
            "chart": chart_result,
            "answer": final_answer,
            "intent": intent
        }

    # -----------------------------
    # COMPATIBILITY METHOD FOR APP.PY
    # -----------------------------
    def process_query(self, user_question: str) -> Dict[str, Any]:
        """
        Wrapper method for backward compatibility
        Used by app.py to maintain the same interface
        """
        try:
            # Get the full response
            response = self.chat(user_question)
            
            # Format the response to match what app.py expects
            if response["result"]["success"]:
                return {
                    "status": "success",
                    "response": response["answer"],
                    "sql": response["sql"],
                    "data": response["result"]["data"],
                    "row_count": response["result"]["row_count"],
                    "columns": response["result"]["columns"],
                    "insights": response.get("insights", ""),
                    "chart": response.get("chart", {}),
                    "intermediate_steps": [],
                    "chat_history": ""
                }
            else:
                return {
                    "status": "error",
                    "response": response["answer"],
                    "sql": response["sql"],
                    "data": [],
                    "error": response["result"].get("error", "Unknown error"),
                    "intermediate_steps": [],
                    "chat_history": ""
                }
                
        except Exception as e:
            print(f"❌ Error in process_query: {e}")
            return {
                "status": "error",
                "response": f"System error: {str(e)}",
                "sql": "",
                "data": [],
                "error": str(e),
                "intermediate_steps": [],
                "chat_history": ""
            }

    # -----------------------------
    # SIMPLE QUERY METHOD
    # -----------------------------
    def simple_query(self, user_question: str) -> str:
        """Simple method that returns only the answer text"""
        result = self.chat(user_question)
        return result["answer"]


# -----------------------------
# TEST THE AGENT
# -----------------------------
if __name__ == "__main__":
    print("🤖 Testing LangChain Agent with Full Pipeline")
    print("=" * 60)
    
    # Initialize agent
    agent = LangChainAgent()
    
    # Test queries
    test_queries = [
        "Show total sales by region",
        "Top 5 products by profit",
        "Sales trend over time",
        "Count of orders by category"
    ]
    
    print("\n🧪 Running test queries...")
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: '{query}'")
        print('='*60)
        
        try:
            # Use the chat method directly
            response = agent.chat(query)
            
            print(f"\n✅ SQL Generated:")
            print(response["sql"])
            
            print(f"\n📊 Results: {response['result']['row_count']} rows")
            
            print(f"\n💡 Insights Generated: {'Yes' if response.get('insights') else 'No'}")
            
            if response.get('chart'):
                if "image_base64" in response["chart"]:
                    print(f"📈 Chart Generated: Yes ({response['chart'].get('type', 'chart')})")
                else:
                    print(f"📈 Chart Generated: No ({response['chart'].get('error', 'Unknown')})")
            
            print(f"\n🎯 Intent: {response['intent'].get('intent', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    # Interactive mode
    print("\n\n🎮 Interactive Mode (type 'exit' to quit)")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n📝 You: ")
            if user_input.lower() in ["exit", "quit", "bye", ""]:
                print("👋 Goodbye!")
                break
            
            print("\n" + "="*60)
            print(f"Processing: '{user_input}'")
            print("="*60)
            
            # Process query
            response = agent.chat(user_input)
            
            print(f"\n🤖 Assistant Response:")
            print("-" * 40)
            print(response["answer"])
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

















#"""
#COMPLETE LangChain SQL Agent with Data Analysis Pipeline
#"""
#
#import os
#import mysql.connector
#from typing import Any, Dict, List
#from dotenv import load_dotenv
#
#from langchain_community.chat_models import ChatOllama
#from langchain_core.messages import HumanMessage, SystemMessage
#
#from schema_extractor import SchemaExtractor
#from intent_classifier import IntentClassifier
#from sql_validator import SQLValidator
#
## Add data analysis imports
#try:
#    from data_analyzer import DataAnalyzer
#    from insight_generator import InsightGenerator
#    from chart_generator import ChartGenerator
#    DATA_ANALYSIS_AVAILABLE = True
#    print("✅ Data analysis modules loaded")
#except ImportError as e:
#    print(f"⚠️ Data analysis modules not available: {e}")
#    print("⚠️ Install: pip install matplotlib numpy pandas")
#    DATA_ANALYSIS_AVAILABLE = False
#    
#    # Create dummy classes
#    class DataAnalyzer:
#        def analyze(self, data, intent, columns):
#            return {
#                "original_data": data,
#                "columns": columns,
#                "row_count": len(data),
#                "analysis": {},
#                "chart_data": {},
#                "chart_type": "bar"
#            }
#    
#    class InsightGenerator:
#        def generate_insights(self, analysis_result, original_query, intent):
#            return "## 📊 Key Insights\n\n*Install data_analyzer.py and insight_generator.py*"
#    
#    class ChartGenerator:
#        def generate_chart(self, analysis_result):
#            return {"error": "Install matplotlib: pip install matplotlib"}
#
#load_dotenv()
#
#
#class LangChainAgent:
#    def __init__(self):
#        print(f"🔧 Initializing LangChainAgent...")
#        
#        # Load schema dynamically
#        self.schema_extractor = SchemaExtractor()
#        self.schema_text = self.schema_extractor.get_formatted_schema()
#        
#        # Initialize intent classifier
#        self.intent_classifier = IntentClassifier()
#        
#        # Initialize SQL validator
#        self.sql_validator = SQLValidator()
#
#        # Initialize Ollama LLM
#        self.llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://62.171.149.65:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
#            temperature=0,
#            timeout=120
#        )
#
#        # DB config
#        self.db_config = {
#            "host": os.getenv("DB_HOST", "localhost"),
#            "port": int(os.getenv("DB_PORT", 3307)),
#            "user": os.getenv("DB_USER", "root"),
#            "password": os.getenv("DB_PASSWORD", ""),
#            "database": os.getenv("DB_NAME", "")
#        }
#        
#        print(f"📡 Using Ollama: {os.getenv('OLLAMA_BASE_URL')}")
#        print(f"🤖 Model: {os.getenv('OLLAMA_MODEL')}")
#        print("✅ LangChainAgent initialized!")
#
#    # -----------------------------
#    # Database Connection
#    # -----------------------------
#    def connect_db(self):
#        return mysql.connector.connect(**self.db_config)
#
#    # -----------------------------
#    # STEP 1: Generate SQL
#    # -----------------------------
#    def generate_sql(self, user_question: str, intent: Dict = None) -> str:
#        system_prompt = f"""
#You are a senior data analyst.
#
#Database Schema:
#{self.schema_text}
#
#Rules:
#- Generate ONLY valid MySQL SELECT queries
#- Do NOT explain anything
#- Do NOT add markdown or backticks
#- Use proper table and column names from schema
#- Include aggregations (SUM, COUNT, AVG) when appropriate
#- Add GROUP BY for dimensional analysis
#- Add ORDER BY for meaningful sorting
#- Limit results if returning many rows
#
#Important: Only use columns and tables that exist in the schema above.
#"""
#
#        # Include intent information if available
#        if intent:
#            intent_info = f"""
#User Intent: {intent.get('intent', 'REPORT')}
#Metrics requested: {', '.join(intent.get('metrics', []))}
#Dimensions requested: {', '.join(intent.get('dimensions', []))}
#Time range: {intent.get('time_range', 'None')}
#"""
#            system_prompt += f"\nAdditional Context:\n{intent_info}"
#
#        try:
#            response = self.llm.invoke([
#                SystemMessage(content=system_prompt),
#                HumanMessage(content=user_question)
#            ])
#
#            sql = response.content.strip()
#            
#            # Clean SQL (remove markdown if present)
#            if sql.startswith("```sql"):
#                sql = sql[6:]
#            if sql.startswith("```"):
#                sql = sql[3:]
#            if sql.endswith("```"):
#                sql = sql[:-3]
#            
#            # Ensure it's a SELECT query
#            sql = sql.strip()
#            if not sql.upper().startswith("SELECT"):
#                print("⚠️ Generated non-SELECT query, adding SELECT * FROM store")
#                return "SELECT * FROM store LIMIT 10"
#            
#            return sql
#            
#        except Exception as e:
#            print(f"❌ SQL generation error: {e}")
#            # Fallback to simple query
#            return "SELECT * FROM store LIMIT 10"
#
#    # -----------------------------
#    # STEP 2: Execute SQL
#    # -----------------------------
#    def execute_sql(self, sql: str) -> Dict[str, Any]:
#        try:
#            conn = self.connect_db()
#            cursor = conn.cursor(dictionary=True)
#            
#            print(f"📊 Executing SQL...")
#            cursor.execute(sql)
#            result = cursor.fetchall()
#            
#            # Get column names
#            column_names = [desc[0] for desc in cursor.description] if cursor.description else []
#            
#            print(f"✅ Query executed successfully!")
#            print(f"   Rows: {len(result)}, Columns: {len(column_names)}")
#            
#            return {
#                "success": True,
#                "row_count": len(result),
#                "column_count": len(column_names),
#                "columns": column_names,
#                "data": result[:100],  # Limit to 100 rows for display
#                "summary": f"Found {len(result)} rows with {len(column_names)} columns"
#            }
#        except mysql.connector.Error as e:
#            print(f"❌ Database error: {e}")
#            return {
#                "success": False,
#                "error": f"Database error: {str(e)}",
#                "row_count": 0,
#                "column_count": 0,
#                "columns": [],
#                "data": [],
#                "summary": f"Error: {str(e)}"
#            }
#        except Exception as e:
#            print(f"❌ Execution error: {e}")
#            return {
#                "success": False,
#                "error": str(e),
#                "row_count": 0,
#                "column_count": 0,
#                "columns": [],
#                "data": [],
#                "summary": f"Error: {str(e)}"
#            }
#        finally:
#            if 'cursor' in locals():
#                cursor.close()
#            if 'conn' in locals():
#                conn.close()
#
#    # -----------------------------
#    # STEP 3: Generate Basic Answer (fallback)
#    # -----------------------------
#    def generate_answer(self, question: str, sql: str, result: Dict) -> str:
#        if not result["success"]:
#            return f"❌ Error: {result.get('error', 'Unknown error')}"
#        
#        if result["row_count"] == 0:
#            return "📭 No data found for your query."
#        
#        prompt = f"""
#User Question: {question}
#
#SQL Used: {sql}
#
#Query Results:
#- Total rows: {result['row_count']}
#- Columns: {', '.join(result['columns'])}
#- Sample data (first 3 rows): {result['data'][:3]}
#
#Provide a clear summary of what this data shows.
#Highlight key numbers and patterns.
#Keep it concise and business-friendly.
#"""
#
#        try:
#            response = self.llm.invoke([HumanMessage(content=prompt)])
#            return response.content.strip()
#        except Exception as e:
#            print(f"❌ Answer generation error: {e}")
#            return f"Query executed successfully. Found {result['row_count']} rows."
#
#    # -----------------------------
#    # MAIN CHAT FUNCTION - COMPLETE PIPELINE
#    # -----------------------------
#    def chat(self, user_question: str) -> Dict[str, Any]:
#        print(f"\n{'='*60}")
#        print(f"🤖 Processing: {user_question}")
#        print('='*60)
#    
#        # Step 1: Classify intent
#        print("\n🎯 Classifying intent...")
#        try:
#            intent = self.intent_classifier.classify(user_question)
#            print(f"✅ Intent: {intent['intent']}")
#            print(f"   Metrics: {intent['metrics']}")
#            print(f"   Dimensions: {intent['dimensions']}")
#        except Exception as e:
#            print(f"❌ Intent classification failed: {e}")
#            intent = {"intent": "REPORT", "metrics": [], "dimensions": [], "time_range": None}
#        
#        # Step 2: Generate SQL
#        print("\n🧠 Generating SQL...")
#        sql = self.generate_sql(user_question, intent)
#        print(f"✅ SQL: {sql}")
#        
#        # Step 3: Validate SQL
#        print("\n🔒 Validating SQL...")
#        validation_result = self.sql_validator.validate_sql(sql)
#        print(f"✅ Validation: {validation_result}")
#        
#        if validation_result != "VALID_SQL":
#            return {
#                "question": user_question,
#                "sql": sql,
#                "result": {"success": False, "error": "SQL validation failed"},
#                "answer": f"❌ SQL validation failed: {validation_result}",
#                "intent": intent
#            }
#        
#        # Step 4: Execute SQL
#        print("\n📊 Executing SQL...")
#        sql_result = self.execute_sql(sql)
#        
#        if not sql_result["success"]:
#            return {
#                "question": user_question,
#                "sql": sql,
#                "result": sql_result,
#                "answer": f"❌ Query failed: {sql_result.get('error')}",
#                "intent": intent
#            }
#        
#        print(f"✅ Query successful! {sql_result['row_count']} rows retrieved")
#        
#        # Step 5-7: Data Analysis Pipeline (if available)
#        insights = ""
#        chart_result = {"error": "Chart generator not available"}
#        analysis_result = None
#        
#        if DATA_ANALYSIS_AVAILABLE and sql_result["data"]:
#            try:
#                # Step 5: Data Analysis
#                print("\n📈 Analyzing data...")
#                data_analyzer = DataAnalyzer()
#                analysis_result = data_analyzer.analyze(
#                    data=sql_result["data"],
#                    intent=intent,
#                    columns=sql_result["columns"]
#                )
#                
#                # Step 6: Generate Insights
#                print("\n💡 Generating insights...")
#                insight_generator = InsightGenerator()
#                insights = insight_generator.generate_insights(
#                    analysis_result=analysis_result,
#                    original_query=user_question,
#                    intent=intent
#                )
#                
#                # Step 7: Generate Chart
#                print("\n📊 Generating chart...")
#                chart_generator = ChartGenerator()
#                chart_result = chart_generator.generate_chart(analysis_result)
#                
#                if "image_base64" in chart_result:
#                    print(f"✅ Chart generated: {chart_result.get('type', 'chart')}")
#                else:
#                    print(f"⚠️ Chart not generated: {chart_result.get('error', 'Unknown error')}")
#                    
#            except Exception as e:
#                print(f"⚠️ Data analysis error: {e}")
#                insights = f"## 📊 Key Insights\n\n*Analysis error: {str(e)}*"
#        else:
#            print("\n⚠️ Skipping data analysis (modules not available or no data)")
#        
#        # Step 8: Create final answer
#        print("\n💬 Compiling final answer...")
#        
#        answer_parts = []
#        
#        # 1. SQL Query
#        answer_parts.append(f"**📝 SQL Query Used:**\n```sql\n{sql}\n```")
#        
#        # 2. Data Summary
#        answer_parts.append(f"**📊 Data Summary:**")
#        answer_parts.append(f"- Rows retrieved: **{sql_result['row_count']}**")
#        answer_parts.append(f"- Columns: {', '.join(sql_result['columns'])}")
#        
#        # 3. Show sample data if small
#        if sql_result['row_count'] > 0 and sql_result['row_count'] <= 10:
#            answer_parts.append(f"\n**Sample Data:**")
#            for i, row in enumerate(sql_result['data'][:5], 1):
#                row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
#                answer_parts.append(f"{i}. {row_str}")
#        
#        # 4. Insights
#        if insights:
#            answer_parts.append(insights)
#        
#        # 5. Chart info
#        if "image_base64" in chart_result:
#            answer_parts.append(f"**📈 Visualization:** {chart_result.get('type', 'Chart')} generated")
#        elif "error" in chart_result:
#            answer_parts.append(f"**📈 Visualization:** {chart_result['error']}")
#        
#        final_answer = "\n\n".join(answer_parts)
#        
#        print(f"✅ Answer compiled ({len(final_answer)} characters)")
#        
#        return {
#            "question": user_question,
#            "sql": sql,
#            "result": sql_result,
#            "analysis": analysis_result,
#            "insights": insights,
#            "chart": chart_result,
#            "answer": final_answer,
#            "intent": intent
#        }
#
#    # -----------------------------
#    # COMPATIBILITY METHOD FOR APP.PY
#    # -----------------------------
#    def process_query(self, user_question: str) -> Dict[str, Any]:
#        """
#        Wrapper method for backward compatibility
#        Used by app.py to maintain the same interface
#        """
#        try:
#            # Get the full response
#            response = self.chat(user_question)
#            
#            # Format the response to match what app.py expects
#            if response["result"]["success"]:
#                return {
#                    "status": "success",
#                    "response": response["answer"],
#                    "sql": response["sql"],
#                    "data": response["result"]["data"],
#                    "row_count": response["result"]["row_count"],
#                    "columns": response["result"]["columns"],
#                    "insights": response.get("insights", ""),
#                    "chart": response.get("chart", {}),
#                    "intermediate_steps": [],
#                    "chat_history": ""
#                }
#            else:
#                return {
#                    "status": "error",
#                    "response": response["answer"],
#                    "sql": response["sql"],
#                    "data": [],
#                    "error": response["result"].get("error", "Unknown error"),
#                    "intermediate_steps": [],
#                    "chat_history": ""
#                }
#                
#        except Exception as e:
#            print(f"❌ Error in process_query: {e}")
#            return {
#                "status": "error",
#                "response": f"System error: {str(e)}",
#                "sql": "",
#                "data": [],
#                "error": str(e),
#                "intermediate_steps": [],
#                "chat_history": ""
#            }
#
#    # -----------------------------
#    # SIMPLE QUERY METHOD
#    # -----------------------------
#    def simple_query(self, user_question: str) -> str:
#        """Simple method that returns only the answer text"""
#        result = self.chat(user_question)
#        return result["answer"]
#
#
## -----------------------------
## TEST THE AGENT
## -----------------------------
#if __name__ == "__main__":
#    print("🤖 Testing LangChain Agent with Full Pipeline")
#    print("=" * 60)
#    
#    # Initialize agent
#    agent = LangChainAgent()
#    
#    # Test queries
#    test_queries = [
#        "Show total sales by region",
#        "Top 5 products by profit",
#        "Sales trend over time",
#        "Count of orders by category"
#    ]
#    
#    print("\n🧪 Running test queries...")
#    for i, query in enumerate(test_queries, 1):
#        print(f"\n{'='*60}")
#        print(f"Test {i}: '{query}'")
#        print('='*60)
#        
#        try:
#            # Use the chat method directly
#            response = agent.chat(query)
#            
#            print(f"\n✅ SQL Generated:")
#            print(response["sql"])
#            
#            print(f"\n📊 Results: {response['result']['row_count']} rows")
#            
#            print(f"\n💡 Insights Generated: {'Yes' if response.get('insights') else 'No'}")
#            
#            if response.get('chart'):
#                if "image_base64" in response["chart"]:
#                    print(f"📈 Chart Generated: Yes ({response['chart'].get('type', 'chart')})")
#                else:
#                    print(f"📈 Chart Generated: No ({response['chart'].get('error', 'Unknown')})")
#            
#            print(f"\n🎯 Intent: {response['intent'].get('intent', 'Unknown')}")
#            
#        except Exception as e:
#            print(f"❌ Test failed: {e}")
#    
#    # Interactive mode
#    print("\n\n🎮 Interactive Mode (type 'exit' to quit)")
#    print("=" * 60)
#    
#    while True:
#        try:
#            user_input = input("\n📝 You: ")
#            if user_input.lower() in ["exit", "quit", "bye", ""]:
#                print("👋 Goodbye!")
#                break
#            
#            print("\n" + "="*60)
#            print(f"Processing: '{user_input}'")
#            print("="*60)
#            
#            # Process query
#            response = agent.chat(user_input)
#            
#            print(f"\n🤖 Assistant Response:")
#            print("-" * 40)
#            print(response["answer"])
#            print("-" * 40)
#            
#        except KeyboardInterrupt:
#            print("\n\n👋 Interrupted. Goodbye!")
#            break
#        except Exception as e:
#            print(f"❌ Error: {e}")
#




#this worked we get the chart
#"""
#LangChain SQL Agent for Ollama Server with Data Analysis Pipeline
#"""
#
#import os
#import mysql.connector
#from typing import Any, Dict, List
#from dotenv import load_dotenv
#
#from langchain_community.chat_models import ChatOllama
#from langchain_core.messages import HumanMessage, SystemMessage
#
#from schema_extractor import SchemaExtractor
#from intent_classifier import IntentClassifier
#from sql_validator import SQLValidator
#
## Add data analysis imports with error handling
#try:
#    from data_analyzer import DataAnalyzer
#    from insight_generator import InsightGenerator
#    from chart_generator import ChartGenerator
#    DATA_ANALYSIS_AVAILABLE = True
#    print("✅ Data analysis modules loaded successfully")
#except ImportError as e:
#    print(f"⚠️ Data analysis modules not available: {e}")
#    print("⚠️ Running in basic mode (SQL only)")
#    DATA_ANALYSIS_AVAILABLE = False
#    
#    # Create dummy classes if modules aren't available
#    class DataAnalyzer:
#        def analyze(self, data, intent, columns):
#            return {
#                "original_data": data,
#                "columns": columns,
#                "row_count": len(data),
#                "analysis": {"basic": "Basic analysis mode"},
#                "chart_data": {},
#                "chart_type": "bar"
#            }
#    
#    class InsightGenerator:
#        def generate_insights(self, analysis_result, original_query, intent):
#            return "## 📊 Key Insights\n\n*Insights generation requires data_analyzer.py and insight_generator.py modules*"
#    
#    class ChartGenerator:
#        def generate_chart(self, analysis_result):
#            return {"error": "Chart generator not available. Install matplotlib for charts."}
#
#load_dotenv()
#
#
#class LangChainAgent:
#    def __init__(self):
#        print(f"🔧 Initializing LangChainAgent...")
#        print(f"📡 Connecting to Ollama: {os.getenv('OLLAMA_BASE_URL')}")
#        print(f"🤖 Using model: {os.getenv('OLLAMA_MODEL')}")
#        
#        # Load schema dynamically
#        self.schema_extractor = SchemaExtractor()
#        self.schema_text = self.schema_extractor.get_formatted_schema()
#        
#        # Initialize intent classifier
#        self.intent_classifier = IntentClassifier()
#        
#        # Initialize SQL validator
#        self.sql_validator = SQLValidator()
#
#        # Initialize Ollama LLM
#        self.llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://62.171.149.65:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
#            temperature=0,
#            timeout=120  # Increased timeout for slower connections
#        )
#
#        # DB config
#        self.db_config = {
#            "host": os.getenv("DB_HOST", "localhost"),
#            "port": int(os.getenv("DB_PORT", 3307)),
#            "user": os.getenv("DB_USER", "root"),
#            "password": os.getenv("DB_PASSWORD", ""),
#            "database": os.getenv("DB_NAME", "")
#        }
#        
#        print("✅ LangChainAgent initialized successfully!")
#        if not DATA_ANALYSIS_AVAILABLE:
#            print("⚠️ Running in BASIC mode. Install matplotlib, numpy, pandas for full features.")
#
#    # -----------------------------
#    # Database Connection
#    # -----------------------------
#    def connect_db(self):
#        return mysql.connector.connect(**self.db_config)
#
#    # -----------------------------
#    # STEP 1: Generate SQL
#    # -----------------------------
#    def generate_sql(self, user_question: str, intent: Dict = None) -> str:
#        system_prompt = f"""
#You are a senior data analyst.
#
#You are given the FULL database schema below.
#DO NOT guess column names.
#DO NOT hallucinate tables.
#ONLY use columns that exist in the schema.
#
#Schema:
#{self.schema_text}
#
#Rules:
#- Generate ONLY valid MySQL SQL
#- Do NOT explain anything
#- Do NOT add markdown
#- Do NOT add backticks
#- Prefer aggregation if question asks for totals, counts, trends
#"""
#
#        # Include intent information if available
#        if intent:
#            intent_info = f"""
#User Intent: {intent.get('intent', 'REPORT')}
#Metrics requested: {intent.get('metrics', [])}
#Dimensions requested: {intent.get('dimensions', [])}
#Time range: {intent.get('time_range', 'None')}
#"""
#            system_prompt += f"\nAdditional Context:\n{intent_info}"
#
#        try:
#            response = self.llm.invoke([
#                SystemMessage(content=system_prompt),
#                HumanMessage(content=user_question)
#            ])
#
#            sql = response.content.strip()
#            
#            # Clean SQL (remove markdown if present)
#            if sql.startswith("```sql"):
#                sql = sql[6:]
#            if sql.startswith("```"):
#                sql = sql[3:]
#            if sql.endswith("```"):
#                sql = sql[:-3]
#            
#            return sql.strip()
#        except Exception as e:
#            print(f"❌ SQL generation error: {e}")
#            # Fallback to simple query
#            return "SELECT * FROM store LIMIT 10"
#
#    # -----------------------------
#    # STEP 2: Execute SQL
#    # -----------------------------
#    def execute_sql(self, sql: str) -> Dict[str, Any]:
#        try:
#            conn = self.connect_db()
#            cursor = conn.cursor(dictionary=True)
#            
#            print(f"📊 Executing SQL: {sql[:100]}...")
#            cursor.execute(sql)
#            result = cursor.fetchall()
#            
#            # Get column names
#            column_names = [desc[0] for desc in cursor.description] if cursor.description else []
#            
#            return {
#                "success": True,
#                "row_count": len(result),
#                "column_count": len(column_names),
#                "columns": column_names,
#                "data": result[:100],  # Limit to 100 rows for display
#                "summary": f"Found {len(result)} rows with {len(column_names)} columns"
#            }
#        except Exception as e:
#            print(f"❌ SQL execution error: {e}")
#            return {
#                "success": False,
#                "error": str(e),
#                "row_count": 0,
#                "column_count": 0,
#                "columns": [],
#                "data": [],
#                "summary": f"Error: {str(e)}"
#            }
#        finally:
#            if 'cursor' in locals():
#                cursor.close()
#            if 'conn' in locals():
#                conn.close()
#
#    # -----------------------------
#    # STEP 3: Generate Basic Answer (for fallback)
#    # -----------------------------
#    def generate_answer(self, question: str, sql: str, result: Dict) -> str:
#        if not result["success"]:
#            return f"❌ Error executing query: {result.get('error', 'Unknown error')}"
#        
#        if result["row_count"] == 0:
#            return "📭 No data found for your query."
#        
#        prompt = f"""
#User Question:
#{question}
#
#SQL Used:
#{sql}
#
#SQL Result (first few rows):
#{result['data'][:5]}
#
#Total rows: {result['row_count']}
#Columns: {result['columns']}
#
#Explain the answer in simple business language.
#If numbers exist, summarize insights.
#Keep response concise and helpful.
#"""
#
#        try:
#            response = self.llm.invoke([HumanMessage(content=prompt)])
#            return response.content.strip()
#        except Exception as e:
#            print(f"❌ Answer generation error: {e}")
#            return f"Query executed successfully. Found {result['row_count']} rows."
#
#    # -----------------------------
#    # MAIN CHAT FUNCTION WITH FULL PIPELINE
#    # -----------------------------
#    def chat(self, user_question: str) -> Dict[str, Any]:
#        print(f"\n{'='*60}")
#        print(f"Processing query: {user_question}")
#        print('='*60)
#    
#        # Step 1: Classify intent
#        print("\n🎯 Classifying intent...")
#        try:
#            intent = self.intent_classifier.classify(user_question)
#            print(f"Intent: {intent}")
#        except Exception as e:
#            print(f"❌ Intent classification failed: {e}")
#            intent = {"intent": "REPORT", "metrics": [], "dimensions": [], "time_range": None}
#        
#        # Step 2: Generate SQL
#        print("\n🧠 Generating SQL...")
#        sql = self.generate_sql(user_question, intent)
#        print(f"Generated SQL: {sql}")
#        
#        # Step 3: Validate SQL
#        print("\n🔒 Validating SQL...")
#        validation_result = self.sql_validator.validate_sql(sql)
#        print(f"Validation: {validation_result}")
#        
#        if validation_result != "VALID_SQL":
#            return {
#                "question": user_question,
#                "sql": sql,
#                "result": {"success": False, "error": "SQL validation failed"},
#                "answer": f"❌ Generated SQL is not safe to execute: {validation_result}",
#                "intent": intent
#            }
#        
#        # Step 4: Execute SQL
#        print("\n📊 Executing SQL...")
#        sql_result = self.execute_sql(sql)
#        
#        if not sql_result["success"]:
#            return {
#                "question": user_question,
#                "sql": sql,
#                "result": sql_result,
#                "answer": f"❌ Query execution failed: {sql_result.get('error', 'Unknown error')}",
#                "intent": intent
#            }
#        
#        # Step 5: Data Analysis (if available)
#        analysis_result = None
#        insights = ""
#        chart_result = {"error": "Chart generation not available"}
#        
#        if DATA_ANALYSIS_AVAILABLE:
#            try:
#                print("\n📈 Analyzing data...")
#                data_analyzer = DataAnalyzer()
#                analysis_result = data_analyzer.analyze(
#                    data=sql_result["data"],
#                    intent=intent,
#                    columns=sql_result["columns"]
#                )
#                
#                # Step 6: Generate Insights
#                print("\n💡 Generating insights...")
#                insight_generator = InsightGenerator()
#                insights = insight_generator.generate_insights(
#                    analysis_result=analysis_result,
#                    original_query=user_question,
#                    intent=intent
#                )
#                
#                # Step 7: Generate Chart
#                print("\n📊 Generating chart...")
#                chart_generator = ChartGenerator()
#                chart_result = chart_generator.generate_chart(analysis_result)
#                
#            except Exception as e:
#                print(f"⚠️ Data analysis failed: {e}")
#                insights = "## 📊 Key Insights\n\n*Data analysis temporarily unavailable*"
#        else:
#            print("\n⚠️ Skipping data analysis (modules not available)")
#            insights = "## 📊 Key Insights\n\n*Install data_analyzer.py, insight_generator.py, and chart_generator.py for full analysis*"
#        
#        # Step 8: Combine Answer
#        print("\n💬 Compiling final answer...")
#        
#        # Create comprehensive answer
#        answer_parts = []
#        
#        # 1. SQL Used
#        answer_parts.append(f"**📝 SQL Query Used:**\n```sql\n{sql}\n```")
#        
#        # 2. Data Summary
#        answer_parts.append(f"**📊 Data Summary:**")
#        answer_parts.append(f"- Rows retrieved: **{sql_result['row_count']}**")
#        answer_parts.append(f"- Columns: {', '.join(sql_result['columns'])}")
#        
#        # Show sample data if small result set
#        if sql_result['row_count'] <= 5 and sql_result['data']:
#            answer_parts.append(f"\n**Sample Data:**")
#            for i, row in enumerate(sql_result['data'][:3], 1):
#                answer_parts.append(f"{i}. {row}")
#        
#        # 3. Insights
#        answer_parts.append(insights)
#        
#        # 4. Chart if available
#        if "error" not in chart_result and chart_result.get("image_base64"):
#            answer_parts.append(f"**📈 Visualization Generated:** ({chart_result.get('type', 'bar')} chart)")
#        elif DATA_ANALYSIS_AVAILABLE:
#            answer_parts.append(f"**📈 Visualization:** Chart generation not available for this query type.")
#        
#        final_answer = "\n\n".join(answer_parts)
#        
#        return {
#            "question": user_question,
#            "sql": sql,
#            "result": sql_result,
#            "analysis": analysis_result,
#            "insights": insights,
#            "chart": chart_result,
#            "answer": final_answer,
#            "intent": intent
#        }
#
#    # -----------------------------
#    # COMPATIBILITY METHOD FOR APP.PY (CRITICAL FIX)
#    # -----------------------------
#    def process_query(self, user_question: str) -> Dict[str, Any]:
#        """
#        Wrapper method for backward compatibility
#        Used by app.py to maintain the same interface
#        
#        Returns all fields needed by app.py including insights and chart
#        """
#        try:
#            # Get the full response
#            response = self.chat(user_question)
#            
#            # Format the response to match what app.py expects
#            if response["result"]["success"]:
#                return {
#                    "status": "success",
#                    "response": response["answer"],
#                    "sql": response["sql"],
#                    "data": response["result"]["data"],
#                    "row_count": response["result"]["row_count"],
#                    "columns": response["result"]["columns"],
#                    "insights": response.get("insights", ""),  # CRITICAL: Add this
#                    "chart": response.get("chart", {}),        # CRITICAL: Add this
#                    "intermediate_steps": [],
#                    "chat_history": ""
#                }
#            else:
#                return {
#                    "status": "error",
#                    "response": response["answer"],
#                    "sql": response["sql"],
#                    "data": [],
#                    "error": response["result"].get("error", "Unknown error"),
#                    "intermediate_steps": [],
#                    "chat_history": ""
#                }
#                
#        except Exception as e:
#            print(f"❌ Error in process_query: {e}")
#            return {
#                "status": "error",
#                "response": f"System error: {str(e)}",
#                "sql": "",
#                "data": [],
#                "error": str(e),
#                "intermediate_steps": [],
#                "chat_history": ""
#            }
#
#    # -----------------------------
#    # SIMPLE QUERY METHOD (Alternative)
#    # -----------------------------
#    def simple_query(self, user_question: str) -> str:
#        """
#        Simple method that returns only the answer text
#        """
#        result = self.chat(user_question)
#        return result["answer"]
#
#
## -----------------------------
## TEST RUN
## -----------------------------
#if __name__ == "__main__":
#    print("🤖 Testing LangChain Agent with Full Pipeline")
#    print("=" * 60)
#    
#    agent = LangChainAgent()
#    
#    # Test queries
#    test_queries = [
#        "Show total sales",
#        "Sales by region",
#        "Top 5 products by profit",
#        "Number of orders by category"
#    ]
#
#    for i, query in enumerate(test_queries, 1):
#        print(f"\n{'='*60}")
#        print(f"Test {i}: {query}")
#        print('='*60)
#        
#        try:
#            response = agent.chat(query)
#            
#            print(f"\n✅ SQL Generated:")
#            print(response["sql"])
#            
#            print(f"\n📊 Result Summary:")
#            print(f"Rows: {response['result']['row_count']}")
#            print(f"Columns: {response['result']['columns']}")
#            
#            print(f"\n💡 Insights Generated: {'Yes' if response.get('insights') else 'No'}")
#            
#            print(f"\n📈 Chart Generated: {'Yes' if response.get('chart') and 'error' not in response['chart'] else 'No'}")
#            
#            print(f"\n🎯 Intent:")
#            print(response["intent"])
#            
#        except Exception as e:
#            print(f"❌ Error: {e}")
#
#    # Interactive mode
#    print("\n\n🎮 Interactive Mode (type 'exit' to quit)")
#    print("=" * 60)
#    
#    while True:
#        user_input = input("\nYou: ")
#        if user_input.lower() in ["exit", "quit", "bye"]:
#            print("Goodbye!")
#            break
#        
#        try:
#            response = agent.chat(user_input)
#            
#            print(f"\n🤖 Assistant:")
#            print(response["answer"][:500] + "..." if len(response["answer"]) > 500 else response["answer"])
#            
#            show_details = input("\nShow full details? (y/n): ").lower()
#            if show_details == 'y':
#                print(f"\n📄 SQL Used:")
#                print(response["sql"])
#                
#                if response.get("insights"):
#                    print(f"\n💡 Insights:")
#                    print(response["insights"])
#                
#        except Exception as e:
#            print(f"❌ Error: {e}")




#"""
#LangChain SQL Agent for Ollama Server
#"""
#
#import os
#import mysql.connector
#from typing import Any, Dict, List
#from dotenv import load_dotenv
#
#from langchain_community.chat_models import ChatOllama
#from langchain_core.messages import HumanMessage, SystemMessage
#
#from schema_extractor import SchemaExtractor
#from intent_classifier import IntentClassifier
#from sql_validator import SQLValidator
## Add these imports at the top
#from data_analyzer import DataAnalyzer
#from insight_generator import InsightGenerator
#from chart_generator import ChartGenerator
#
#load_dotenv()
#
#
#class LangChainAgent:
#    def __init__(self):
#        print(f"🔧 Initializing LangChainAgent...")
#        print(f"📡 Connecting to Ollama: {os.getenv('OLLAMA_BASE_URL')}")
#        print(f"🤖 Using model: {os.getenv('OLLAMA_MODEL')}")
#        
#        # Load schema dynamically
#        self.schema_extractor = SchemaExtractor()
#        self.schema_text = self.schema_extractor.get_formatted_schema()
#        
#        # Initialize intent classifier
#        self.intent_classifier = IntentClassifier()
#        
#        # Initialize SQL validator
#        self.sql_validator = SQLValidator()
#
#        # Initialize Ollama LLM - THIS IS CORRECT FOR YOUR URL
#        self.llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://62.171.149.65:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
#            temperature=0,
#            timeout=120  # Increased timeout for slower connections
#        )
#
#        # DB config
#        self.db_config = {
#            "host": os.getenv("DB_HOST", "localhost"),
#            "port": int(os.getenv("DB_PORT", 3307)),
#            "user": os.getenv("DB_USER", "root"),
#            "password": os.getenv("DB_PASSWORD", ""),
#            "database": os.getenv("DB_NAME", "")
#        }
#        
#        print("✅ LangChainAgent initialized successfully!")
#
#    # ... [rest of your existing methods stay exactly the same]
#    # Database Connection, generate_sql, execute_sql, etc.
#
#
##"""
##LangChain SQL Agent using Dynamic Schema (NO manual columns)
##"""
##
##import os
##import mysql.connector
##from typing import Any, Dict, List
##from dotenv import load_dotenv
##
##from langchain_community.chat_models import ChatOllama
##from langchain_core.messages import HumanMessage, SystemMessage
##
##from schema_extractor import SchemaExtractor
##from intent_classifier import IntentClassifier
##from sql_validator import SQLValidator
##
##load_dotenv()
##
##
##class LangChainAgent:
##    def __init__(self):
##        # Load schema dynamically
##        self.schema_extractor = SchemaExtractor()
##        self.schema_text = self.schema_extractor.get_formatted_schema()
##        
##        # Initialize intent classifier
##        self.intent_classifier = IntentClassifier()
##        
##        # Initialize SQL validator
##        self.sql_validator = SQLValidator()
##
##        # LLM
##        self.llm = ChatOllama(
##            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
##            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
##            temperature=0
##        )
##
##        # DB config
##        self.db_config = {
##            "host": os.getenv("DB_HOST", "localhost"),
##            "port": int(os.getenv("DB_PORT", 3307)),
##            "user": os.getenv("DB_USER", "root"),
##            "password": os.getenv("DB_PASSWORD", ""),
##            "database": os.getenv("DB_NAME", "")
##        }
##
#    # -----------------------------
#    # Database Connection
#    # -----------------------------
#    def connect_db(self):
#        return mysql.connector.connect(**self.db_config)
#
#    # -----------------------------
#    # STEP 1: Generate SQL
#    # -----------------------------
#    def generate_sql(self, user_question: str, intent: Dict = None) -> str:
#        system_prompt = f"""
#You are a senior data analyst.
#
#You are given the FULL database schema below.
#DO NOT guess column names.
#DO NOT hallucinate tables.
#ONLY use columns that exist in the schema.
#
#Schema:
#{self.schema_text}
#
#Rules:
#- Generate ONLY valid MySQL SQL
#- Do NOT explain anything
#- Do NOT add markdown
#- Do NOT add backticks
#- Prefer aggregation if question asks for totals, counts, trends
#"""
#
#        # Include intent information if available
#        if intent:
#            intent_info = f"""
#User Intent: {intent.get('intent', 'REPORT')}
#Metrics requested: {intent.get('metrics', [])}
#Dimensions requested: {intent.get('dimensions', [])}
#Time range: {intent.get('time_range', 'None')}
#"""
#            system_prompt += f"\nAdditional Context:\n{intent_info}"
#
#        response = self.llm.invoke([
#            SystemMessage(content=system_prompt),
#            HumanMessage(content=user_question)
#        ])
#
#        sql = response.content.strip()
#        
#        # Clean SQL (remove markdown if present)
#        if sql.startswith("```sql"):
#            sql = sql[6:]
#        if sql.startswith("```"):
#            sql = sql[3:]
#        if sql.endswith("```"):
#            sql = sql[:-3]
#        
#        return sql.strip()
#
#    # -----------------------------
#    # STEP 2: Execute SQL
#    # -----------------------------
#    def execute_sql(self, sql: str) -> Dict[str, Any]:
#        conn = self.connect_db()
#        cursor = conn.cursor(dictionary=True)
#
#        try:
#            cursor.execute(sql)
#            result = cursor.fetchall()
#            
#            # Get column names
#            column_names = [desc[0] for desc in cursor.description] if cursor.description else []
#            
#            return {
#                "success": True,
#                "row_count": len(result),
#                "column_count": len(column_names),
#                "columns": column_names,
#                "data": result[:100],  # Limit to 100 rows for display
#                "summary": f"Found {len(result)} rows with {len(column_names)} columns"
#            }
#        except Exception as e:
#            return {
#                "success": False,
#                "error": str(e),
#                "row_count": 0,
#                "column_count": 0,
#                "columns": [],
#                "data": [],
#                "summary": f"Error: {str(e)}"
#            }
#        finally:
#            cursor.close()
#            conn.close()
#
#    # -----------------------------
#    # STEP 3: Generate Final Answer
#    # -----------------------------
#    def generate_answer(self, question: str, sql: str, result: Dict) -> str:
#        if not result["success"]:
#            return f"Error executing query: {result.get('error', 'Unknown error')}"
#        
#        if result["row_count"] == 0:
#            return "No data found for your query."
#        
#        prompt = f"""
#User Question:
#{question}
#
#SQL Used:
#{sql}
#
#SQL Result (first few rows):
#{result['data'][:5]}
#
#Total rows: {result['row_count']}
#Columns: {result['columns']}
#
#Explain the answer in simple business language.
#If numbers exist, summarize insights.
#Keep response concise and helpful.
#"""
#
#        response = self.llm.invoke([HumanMessage(content=prompt)])
#        return response.content.strip()
#
#    # -----------------------------
#    # MAIN CHAT FUNCTION
#    # ------------ran this fun before data analysis added changing when data analyzer.py is adding-----------------
#    
#    def chat(self, user_question: str) -> Dict[str, Any]:
#        print(f"\n{'='*60}")
#        print(f"Processing query: {user_question}")
#        print('='*60)
#    
#        # Step 1: Classify intent
#        print("\n🎯 Classifying intent...")
#        intent = self.intent_classifier.classify(user_question)
#        print(f"Intent: {intent}")
#        
#        # Step 2: Generate SQL
#        print("\n🧠 Generating SQL...")
#        sql = self.generate_sql(user_question, intent)
#        print(f"Generated SQL: {sql}")
#        
#        # Step 3: Validate SQL
#        print("\n🔒 Validating SQL...")
#        validation_result = self.sql_validator.validate_sql(sql)
#        print(f"Validation: {validation_result}")
#        
#        if validation_result != "VALID_SQL":
#            return {
#                "question": user_question,
#                "sql": sql,
#                "result": {"success": False, "error": "SQL validation failed"},
#                "answer": f"Generated SQL is not safe to execute: {validation_result}",
#                "intent": intent
#            }
#        
#        # Step 4: Execute SQL
#        print("\n📊 Executing SQL...")
#        sql_result = self.execute_sql(sql)
#        
#        if not sql_result["success"]:
#            return {
#                "question": user_question,
#                "sql": sql,
#                "result": sql_result,
#                "answer": f"Query execution failed: {sql_result.get('error', 'Unknown error')}",
#                "intent": intent
#            }
#        
#        # Step 5: Data Analysis
#        print("\n📈 Analyzing data...")
#        data_analyzer = DataAnalyzer()
#        analysis_result = data_analyzer.analyze(
#            data=sql_result["data"],
#            intent=intent,
#            columns=sql_result["columns"]
#        )
#        
#        # Step 6: Generate Insights
#        print("\n💡 Generating insights...")
#        insight_generator = InsightGenerator()
#        insights = insight_generator.generate_insights(
#            analysis_result=analysis_result,
#            original_query=user_question,
#            intent=intent
#        )
#        
#        # Step 7: Generate Chart
#        print("\n📊 Generating chart...")
#        chart_generator = ChartGenerator()
#        chart_result = chart_generator.generate_chart(analysis_result)
#        
#        # Step 8: Combine Answer
#        print("\n💬 Compiling final answer...")
#        
#        # Create comprehensive answer
#        answer_parts = []
#        
#        # 1. SQL Used
#        answer_parts.append(f"**📝 SQL Query Used:**\n```sql\n{sql}\n```")
#        
#        # 2. Data Summary
#        answer_parts.append(f"**📊 Data Summary:**\n- Rows: {sql_result['row_count']}")
#        if sql_result['row_count'] <= 10:
#            answer_parts.append(f"- Data: {sql_result['data']}")
#        
#        # 3. Insights
#        answer_parts.append(insights)
#        
#        # 4. Chart if available
#        if "error" not in chart_result:
#            answer_parts.append(f"**📈 Chart Generated:** ({chart_result.get('type', 'bar')} chart)")
#            # Include base64 image or ECharts config
#            if chart_result.get("image_base64"):
#                answer_parts.append(f"![Chart](data:image/png;base64,{chart_result['image_base64'][:100]}...)")
#        
#        final_answer = "\n\n".join(answer_parts)
#        
#        return {
#            "question": user_question,
#            "sql": sql,
#            "result": sql_result,
#            "analysis": analysis_result,
#            "insights": insights,
#            "chart": chart_result,
#            "answer": final_answer,
#            "intent": intent
#        }
#        
#   #def chat(self, user_question: str) -> Dict[str, Any]:
#   #    print(f"\n{'='*60}")
#   #    print(f"Processing query: {user_question}")
#   #    print('='*60)
#   #    
#   #    # Step 1: Classify intent
#   #    print("\n🎯 Classifying intent...")
#   #    intent = self.intent_classifier.classify(user_question)
#   #    print(f"Intent: {intent}")
#   #    
#   #    # Step 2: Generate SQL
#   #    print("\n🧠 Generating SQL...")
#   #    sql = self.generate_sql(user_question, intent)
#   #    print(f"Generated SQL: {sql}")
#   #    
#   #    # Step 3: Validate SQL
#   #    print("\n🔒 Validating SQL...")
#   #    validation_result = self.sql_validator.validate_sql(sql)
#   #    print(f"Validation: {validation_result}")
#   #    
#   #    if validation_result != "VALID_SQL":
#   #        return {
#   #            "question": user_question,
#   #            "sql": sql,
#   #            "result": {"success": False, "error": "SQL validation failed"},
#   #            "answer": f"Generated SQL is not safe to execute: {validation_result}"
#   #        }
#   #    
#   #    # Step 4: Execute SQL
#   #    print("\n📊 Executing SQL...")
#   #    result = self.execute_sql(sql)
#   #    
#   #    # Step 5: Generate answer
#   #    print("\n💬 Generating answer...")
#   #    answer = self.generate_answer(user_question, sql, result)
#   #    
#   #    return {
#   #        "question": user_question,
#   #        "sql": sql,
#   #        "result": result,
#   #        "answer": answer,
#   #        "intent": intent
#   #    }
#
#    # -----------------------------
#    # COMPATIBILITY METHOD FOR APP.PY
#    # -----------------------------
#    def process_query(self, user_question: str) -> Dict[str, Any]:
#        """
#        Wrapper method for backward compatibility
#        Used by app.py to maintain the same interface
#        
#        Returns: {
#            "status": "success/error",
#            "response": "answer text",
#            "sql": "generated sql",
#            "data": result data,
#            "intermediate_steps": [],
#            "chat_history": ""
#        }
#        """
#        try:
#            # Get the full response
#            response = self.chat(user_question)
#            
#            # Format the response to match what app.py expects
#            if response["result"]["success"]:
#                return {
#                    "status": "success",
#                    "response": response["answer"],
#                    "sql": response["sql"],
#                    "data": response["result"]["data"],
#                    "row_count": response["result"]["row_count"],
#                    "columns": response["result"]["columns"],
#                    "intermediate_steps": [],  # For compatibility
#                    "chat_history": ""  # For compatibility
#                }
#            else:
#                return {
#                    "status": "error",
#                    "response": response["answer"],
#                    "sql": response["sql"],
#                    "data": [],
#                    "error": response["result"].get("error", "Unknown error"),
#                    "intermediate_steps": [],
#                    "chat_history": ""
#                }
#                
#        except Exception as e:
#            print(f"❌ Error in process_query: {e}")
#            return {
#                "status": "error",
#                "response": f"System error: {str(e)}",
#                "sql": "",
#                "data": [],
#                "error": str(e),
#                "intermediate_steps": [],
#                "chat_history": ""
#            }
#
#    # -----------------------------
#    # SIMPLE QUERY METHOD (Alternative)
#    # -----------------------------
#    def simple_query(self, user_question: str) -> str:
#        """
#        Simple method that returns only the answer text
#        """
#        result = self.chat(user_question)
#        return result["answer"]
#
#
## -----------------------------
## TEST RUN
## -----------------------------
#if __name__ == "__main__":
#    agent = LangChainAgent()
#
#    print("\n🤖 Testing LangChain Agent")
#    print("Type 'exit' to quit\n")
#
#    test_queries = [
#        "Show total sales",
#        "Sales by region",
#        "Top 5 products by profit",
#        "Number of orders by category"
#    ]
#
#    for i, query in enumerate(test_queries, 1):
#        print(f"\n{'='*60}")
#        print(f"Test {i}: {query}")
#        print('='*60)
#        
#        try:
#            response = agent.chat(query)
#            
#            print(f"\n✅ SQL Generated:")
#            print(response["sql"])
#            
#            print(f"\n📊 Result Summary:")
#            print(f"Rows: {response['result']['row_count']}")
#            print(f"Columns: {response['result']['columns']}")
#            
#            print(f"\n💬 Answer:")
#            print(response["answer"][:200] + "..." if len(response["answer"]) > 200 else response["answer"])
#            
#            print(f"\n🎯 Intent:")
#            print(response["intent"])
#            
#        except Exception as e:
#            print(f"❌ Error: {e}")
#
#    # Interactive mode
#    print("\n\n🎮 Interactive Mode:")
#    print("You can now ask questions (type 'exit' to quit)")
#    
#    while True:
#        user_input = input("\nYou: ")
#        if user_input.lower() in ["exit", "quit", "bye"]:
#            print("Goodbye!")
#            break
#        
#        try:
#            response = agent.chat(user_input)
#            
#            print(f"\n🤖 Assistant:")
#            print(response["answer"])
#            
#            # Option to see SQL
#            show_sql = input("\nShow SQL? (y/n): ").lower()
#            if show_sql == 'y':
#                print(f"\n📄 SQL Used:")
#                print(response["sql"])
#                
#        except Exception as e:
#            print(f"❌ Error: {e}")
#










#"""
#LangChain SQL Agent using Dynamic Schema (NO manual columns)
#"""
#
#import os
#import mysql.connector
#from typing import Any, Dict, List
#from dotenv import load_dotenv
#
#from langchain_community.chat_models import ChatOllama
#from langchain_core.messages import HumanMessage, SystemMessage
#
#from schema_extractor import SchemaExtractor
#
#load_dotenv()
#
#
#class LangChainAgent:
#
#    def __init__(self):
#        # Load schema dynamically
#        self.schema_extractor = SchemaExtractor()
#        self.schema_text = self.schema_extractor.get_formatted_schema()
#
#        # LLM
#        self.llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
#            temperature=0
#        )
#
#        # DB config
#        self.db_config = {
#            "host": os.getenv("DB_HOST", "localhost"),
#            "port": int(os.getenv("DB_PORT", 3307)),
#            "user": os.getenv("DB_USER", "root"),
#            "password": os.getenv("DB_PASSWORD", ""),
#            "database": os.getenv("DB_NAME", "")
#        }
#
#    # -----------------------------
#    # Database Connection
#    # -----------------------------
#    def connect_db(self):
#        return mysql.connector.connect(**self.db_config)
#
#    # -----------------------------
#    # STEP 1: Generate SQL
#    # -----------------------------
#    def generate_sql(self, user_question: str) -> str:
#        system_prompt = f"""
#You are a senior data analyst.
#
#You are given the FULL database schema below.
#DO NOT guess column names.
#DO NOT hallucinate tables.
#ONLY use columns that exist in the schema.
#
#Schema:
#{self.schema_text}
#
#Rules:
#- Generate ONLY valid MySQL SQL
#- Do NOT explain anything
#- Do NOT add markdown
#- Do NOT add backticks
#- Prefer aggregation if question asks for totals, counts, trends
#"""
#
#        response = self.llm.invoke([
#            SystemMessage(content=system_prompt),
#            HumanMessage(content=user_question)
#        ])
#
#        return response.content.strip()
#
#    # -----------------------------
#    # STEP 2: Execute SQL
#    # -----------------------------
#    def execute_sql(self, sql: str) -> List[Dict[str, Any]]:
#        conn = self.connect_db()
#        cursor = conn.cursor(dictionary=True)
#
#        try:
#            cursor.execute(sql)
#            result = cursor.fetchall()
#            return result
#        finally:
#            cursor.close()
#            conn.close()
#
#    # -----------------------------
#    # STEP 3: Generate Final Answer
#    # -----------------------------
#    def generate_answer(self, question: str, sql: str, result: List[Dict]) -> str:
#        prompt = f"""
#User Question:
#{question}
#
#SQL Used:
#{sql}
#
#SQL Result:
#{result}
#
#Explain the answer in simple business language.
#If numbers exist, summarize insights.
#"""
#
#        response = self.llm.invoke([HumanMessage(content=prompt)])
#        return response.content.strip()
#
#    # -----------------------------
#    # MAIN CHAT FUNCTION
#    # -----------------------------
#    def chat(self, user_question: str) -> Dict[str, Any]:
#        print("\n🧠 Generating SQL...")
#        sql = self.generate_sql(user_question)
#        print("📄 SQL:", sql)
#
#        print("\n📊 Executing SQL...")
#        result = self.execute_sql(sql)
#
#        print("\n💬 Generating answer...")
#        answer = self.generate_answer(user_question, sql, result)
#
#        return {
#            "question": user_question,
#            "sql": sql,
#            "result": result,
#            "answer": answer
#        }
#
#
## -----------------------------
## TEST RUN
## -----------------------------
#if __name__ == "__main__":
#    agent = LangChainSQLAgent()
#
#    print("\n🤖 Ask questions about your database (type 'exit' to quit)\n")
#
#    while True:
#        user_input = input("You: ")
#        if user_input.lower() in ["exit", "quit"]:
#            break
#
#        try:
#            response = agent.chat(user_input)
#
#            print("\n✅ ANSWER:")
#            print(response["answer"])
#
#        except Exception as e:
#            print("❌ Error:", e)
#















#"""
#Complete LangChain Agent for SQL Generation
#Generates SQL queries using LLM
#"""
#import os
#import json
#from typing import Dict, Any, List
#
## LANGCHAIN 1.2.7 IMPORTS
#from langchain.agents import create_agent
##from langchain_community.chat_models import ChatOllama
#try:
#    from langchain_ollama import ChatOllama
#except ImportError:
#    from langchain_community.chat_models import ChatOllama
#from langchain_core.tools import Tool
#from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#
## MEMORY IMPORTS - moved to langchain_community
#from langchain_community.chat_message_histories import ChatMessageHistory
#from langchain_core.messages import HumanMessage, AIMessage
#from langchain_core.chat_history import BaseChatMessageHistory
#
#from schema_extractor import SchemaExtractor
#from intent_classifier import IntentClassifier
#from sql_validator import SQLValidator
#
#import mysql.connector
#import pandas as pd
#from dotenv import load_dotenv
#
#load_dotenv()
#
#class DatabaseTool:
#    """Database execution tool"""
#    
#    def __init__(self):
#        self.host = os.getenv("DB_HOST", "localhost")
#        self.port = int(os.getenv("DB_PORT", 3306))
#        self.user = os.getenv("DB_USER", "root")
#        self.password = os.getenv("DB_PASSWORD", "")
#        self.database = os.getenv("DB_NAME", "")
#    
#    def execute_query(self, sql: str) -> Dict[str, Any]:
#        """Execute SQL query and return structured results"""
#        try:
#            conn = mysql.connector.connect(
#                host=self.host,
#                port=self.port,
#                user=self.user,
#                password=self.password,
#                database=self.database
#            )
#            
#            df = pd.read_sql(sql, conn)
#            conn.close()
#            
#            return {
#                "success": True,
#                "row_count": len(df),
#                "column_count": len(df.columns),
#                "columns": list(df.columns),
#                "data": df.head(10).to_dict('records'),
#                "summary": f"Found {len(df)} rows with {len(df.columns)} columns"
#            }
#            
#        except Exception as e:
#            return {
#                "success": False,
#                "error": str(e)
#            }
#
#class SQLGenerator:
#    """Generates SQL queries using LLM"""
#    
#    def __init__(self):
#        self.llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama2"),
#            temperature=0.1
#        )
#    
#    def generate_sql(self, user_query: str, intent: Dict, schema: str) -> str:
#        """Generate SQL query based on user query, intent, and schema"""
#        
#        prompt = f"""You are a SQL expert. Generate a MySQL SELECT query.
#
#DATABASE SCHEMA:
#{schema[:4000]}
#
#USER QUERY: {user_query}
#INTENT: {intent['intent']}
#METRICS: {intent['metrics']}
#DIMENSIONS: {intent['dimensions']}
#TIME RANGE: {intent['time_range']}
#
#GENERATE A SQL QUERY THAT:
#1. Answers the user's question accurately
#2. Uses proper MySQL syntax
#3. Includes appropriate aggregations if needed
#4. Groups data when analyzing by dimensions
#5. Orders results meaningfully
#6. Limits results if returning many rows
#
#RULES:
#- Only SELECT queries, no modifications
#- Use table and column names exactly as in schema
#- Handle dates properly if time_range is specified
#
#Output ONLY the SQL query, nothing else. No explanations, no markdown, just SQL:"""
#        
#        try:
#            response = self.llm.invoke(prompt)
#            sql = response.content.strip()
#            
#            # Clean SQL (remove markdown if present)
#            if sql.startswith("```sql"):
#                sql = sql[6:]
#            if sql.startswith("```"):
#                sql = sql[3:]
#            if sql.endswith("```"):
#                sql = sql[:-3]
#            
#            return sql.strip()
#            
#        except Exception as e:
#            print(f"❌ SQL generation error: {e}")
#            return self._get_fallback_sql(intent)
#    
#    def _get_fallback_sql(self, intent: Dict) -> str:
#        """Simple fallback SQL if LLM fails"""
#        if intent['intent'] == 'KPI':
#            return "SELECT COUNT(*) as count FROM store"
#        elif intent['intent'] == 'COMPARISON':
#            return "SELECT Category, COUNT(*) as count FROM store GROUP BY Category"
#        else:
#            return "SELECT * FROM store LIMIT 10"
#
#class LangChainAgent:
#    def __init__(self):
#        # Initialize LLM for the agent
#        self.llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama2"),
#            temperature=0.1
#        )
#        
#        # Initialize all components
#        self.schema_extractor = SchemaExtractor()
#        self.intent_classifier = IntentClassifier()
#        self.sql_validator = SQLValidator()
#        self.sql_generator = SQLGenerator()
#        self.db_tool = DatabaseTool()
#        
#        # Create tools for LangChain
#        self.tools = self._create_tools()
#        
#        # Create memory for conversation
#        self.chat_history = ChatMessageHistory()
#        
#        # Create agent
#        self.agent = self._create_agent()
#    
#    def _create_tools(self) -> List[Tool]:
#        """Create LangChain tools"""
#        
#        def execute_sql_tool(sql_query: str) -> str:
#            """Execute SQL and return results"""
#            result = self.db_tool.execute_query(sql_query)
#            if result["success"]:
#                return json.dumps(result, indent=2)
#            else:
#                return f"Error: {result['error']}"
#        
#        def get_schema_tool() -> str:
#            """Get database schema"""
#            return self.schema_extractor.get_formatted_schema()
#        
#        def validate_sql_tool(sql_query: str) -> str:
#            """Validate SQL query"""
#            result = self.sql_validator.validate(sql_query)
#            if result == "VALID_SQL":
#                return "VALID_SQL: Query is safe to execute"
#            else:
#                return "INVALID_SQL: Query contains dangerous operations or syntax errors"
#        
#        def classify_intent_tool(user_query: str) -> str:
#            """Classify user intent"""
#            result = self.intent_classifier.classify(user_query)
#            return json.dumps(result, indent=2)
#        
#        def generate_sql_tool(user_query: str, intent_json: str) -> str:
#            """Generate SQL query"""
#            try:
#                intent = json.loads(intent_json)
#                schema = self.schema_extractor.get_formatted_schema()
#                sql = self.sql_generator.generate_sql(user_query, intent, schema)
#                return sql
#            except Exception as e:
#                return f"Error generating SQL: {e}"
#        
#        # Create Tool objects
#        return [
#            Tool(
#                name="get_database_schema",
#                func=get_schema_tool,
#                description="Get the complete database schema with column descriptions. Use this to understand the database structure before generating SQL."
#            ),
#            Tool(
#                name="classify_user_intent",
#                func=classify_intent_tool,
#                description="Classify user query intent (KPI, REPORT, TREND, COMPARISON, DISTRIBUTION). Use this first to understand what the user wants."
#            ),
#            Tool(
#                name="generate_sql_query",
#                func=generate_sql_tool,
#                description="Generate SQL query based on user query and intent. First get intent using classify_user_intent, then generate SQL."
#            ),
#            Tool(
#                name="validate_sql_query",
#                func=validate_sql_tool,
#                description="Validate if SQL query is safe. Returns VALID_SQL or INVALID_SQL. Use this before executing any SQL."
#            ),
#            Tool(
#                name="execute_sql_query",
#                func=execute_sql_tool,
#                description="Execute a SQL SELECT query and return results. Only use this after validating the SQL."
#            )
#        ]
#    
#    #def _create_agent(self):
#    #    """Create LangChain agent using create_agent"""
#    #    
#    #    # System prompt
#    #    system_prompt = """You are a SQL Assistant. Follow this EXACT workflow:
##
#    #    1. Use 'classify_user_intent' to understand what user wants
#    #    2. Use 'get_database_schema' to see database structure
#    #    3. Use 'generate_sql_query' to create SQL (pass user query and intent)
#    #    4. Use 'validate_sql_query' to check SQL safety
#    #    5. If valid, use 'execute_sql_query' to run it
#    #    6. Show results to user
##
#    #    Rules:
#    #    - Always show the generated SQL query to user
#    #    - Only proceed to execution if SQL is VALID
#    #    - If SQL is INVALID, explain why and suggest fix
#    #    - Only use SELECT queries"""
#    #    
#    #    prompt = ChatPromptTemplate.from_messages([
#    #        ("system", system_prompt),
#    #        MessagesPlaceholder(variable_name="chat_history"),
#    #        ("human", "{input}"),
#    #        MessagesPlaceholder(variable_name="agent_scratchpad"),
#    #    ])
#    #    
#    #    # Create agent using create_agent
#    #    agent = create_agent(
#    #        llm=self.llm,
#    #        tools=self.tools,
#    #        prompt=prompt,
#    #        verbose=True,
#    #        max_iterations=5,
#    #        handle_parsing_errors=True
#    #    )
#    #    
#    #    return agent
#    
#    def _create_agent(self):
#        """Simple direct agent without complex dependencies"""
#        
#        # Import what's available in your version
#        from langchain.agents import initialize_agent, AgentType
#        
#        # System prompt
#        system_prompt = """You are a SQL Assistant. Follow this EXACT workflow:
#    
#        1. Use 'classify_user_intent' to understand what user wants
#        2. Use 'get_database_schema' to see database structure
#        3. Use 'generate_sql_query' to create SQL (pass user query and intent)
#        4. Use 'validate_sql_query' to check SQL safety
#        5. If valid, use 'execute_sql_query' to run it
#        6. Show results to user
#    
#        Rules:
#        - Always show the generated SQL query to user
#        - Only proceed to execution if SQL is VALID
#        - If SQL is INVALID, explain why and suggest fix
#        - Only use SELECT queries"""
#        
#        # Try to create agent using initialize_agent (more common)
#        try:
#            agent = initialize_agent(
#                tools=self.tools,
#                llm=self.llm,
#                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#                verbose=True,
#                max_iterations=5,
#                handle_parsing_errors=True,
#                agent_kwargs={
#                    "prefix": system_prompt
#                }
#            )
#            return agent
#        except:
#            # Fallback: Use tools directly without agent
#            return None
#    
#    def process_query_with_agent(self, user_query: str) -> Dict[str, Any]:
#        """
#        Process query using LangChain agent
#        Returns: {
#            "status": "success/error",
#            "response": "...",
#            "intermediate_steps": [...]
#        }
#        """
#        try:
#            print(f"\n{'='*60}")
#            print(f"Processing query with LangChain Agent: {user_query}")
#            print('='*60)
#            
#            # Add user message to history
#            self.chat_history.add_user_message(user_query)
#            
#            # Get chat history
#            chat_history_messages = self.chat_history.messages
#            
#            # Use the LangChain agent
#            result = self.agent.invoke({
#                "input": user_query,
#                "chat_history": chat_history_messages
#            })
#            
#            print(f"\n✅ Agent execution complete")
#            
#            # Extract output
#            output = ""
#            if isinstance(result, dict):
#                output = result.get("output", str(result))
#            else:
#                output = str(result)
#            
#            # Add AI response to history
#            self.chat_history.add_ai_message(output)
#            
#            return {
#                "status": "success",
#                "response": output,
#                "intermediate_steps": result.get("intermediate_steps", []) if isinstance(result, dict) else [],
#                "chat_history": str(chat_history_messages)
#            }
#            
#        except Exception as e:
#            print(f"\n❌ Error in process_query_with_agent: {e}")
#            return {
#                "status": "error",
#                "response": f"Agent error: {str(e)}",
#                "intermediate_steps": [],
#                "chat_history": ""
#            }
#    
#    def process_query(self, user_query: str) -> Dict[str, Any]:
#        """
#        Main method for backward compatibility
#        Uses LangChain agent for processing
#        """
#        return self.process_query_with_agent(user_query)
#
## Simple standalone function for testing
#def simple_sql_generation(user_query: str) -> Dict[str, Any]:
#    """Simple function for app.py to use"""
#    agent = LangChainAgent()
#    return agent.process_query(user_query)
#
## Test
#if __name__ == "__main__":
#    print("Testing LangChain Agent with create_agent...")
#    
#    # Check Ollama is running
#    try:
#        llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama2")
#        )
#        test_response = llm.invoke("Test")
#        print("✅ Ollama is running")
#    except:
#        print("❌ Ollama is not running. Please start it with: ollama serve")
#        exit(1)
#    
#    # Test queries
#    agent = LangChainAgent()
#    
#    test_queries = [
#        "Show total sales",
#        "Sales by category",
#    ]
#    
#    for i, query in enumerate(test_queries, 1):
#        print(f"\n{'='*60}")
#        print(f"TEST {i}: {query}")
#        print('='*60)
#        
#        result = agent.process_query_with_agent(query)
#        
#        print(f"\n📋 RESULT SUMMARY:")
#        print(f"Status: {result['status']}")
#        print(f"Response: {result['response'][:100]}...")
#        
#        print(f"\n✅ Test {i} complete")
#


















#"""
#Complete LangChain Agent for SQL Generation
#Generates SQL queries using LLM
#"""
#import os
#import json
#from typing import Dict, Any, List
#
## LANGCHAIN 1.2.7 IMPORTS - CORRECT
##from langchain.agents import AgentExecutor, create_react_agent
##from langchain_community.chat_models import ChatOllama
##from langchain_core.tools import Tool
##from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
##from langchain.memory import ConversationBufferMemory
##from langchain import hub  # For pulling prompts
#
#from langchain.agents import create_agent
#from langchain_community.chat_models import ChatOllama
#from langchain_core.tools import Tool
#from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#from langchain.memory import ConversationBufferMemory
#
#from schema_extractor import SchemaExtractor
#from intent_classifier import IntentClassifier
#from sql_validator import SQLValidator
#
#import mysql.connector
#import pandas as pd
#from dotenv import load_dotenv
#
#load_dotenv()
#
#class DatabaseTool:
#    """Database execution tool"""
#    
#    def __init__(self):
#        self.host = os.getenv("DB_HOST", "localhost")
#        self.port = int(os.getenv("DB_PORT", 3306))
#        self.user = os.getenv("DB_USER", "root")
#        self.password = os.getenv("DB_PASSWORD", "")
#        self.database = os.getenv("DB_NAME", "")
#    
#    def execute_query(self, sql: str) -> Dict[str, Any]:
#        """Execute SQL query and return structured results"""
#        try:
#            conn = mysql.connector.connect(
#                host=self.host,
#                port=self.port,
#                user=self.user,
#                password=self.password,
#                database=self.database
#            )
#            
#            df = pd.read_sql(sql, conn)
#            conn.close()
#            
#            return {
#                "success": True,
#                "row_count": len(df),
#                "column_count": len(df.columns),
#                "columns": list(df.columns),
#                "data": df.head(10).to_dict('records'),
#                "summary": f"Found {len(df)} rows with {len(df.columns)} columns"
#            }
#            
#        except Exception as e:
#            return {
#                "success": False,
#                "error": str(e)
#            }
#
#class SQLGenerator:
#    """Generates SQL queries using LLM"""
#    
#    def __init__(self):
#        self.llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama2"),
#            temperature=0.1
#        )
#    
#    def generate_sql(self, user_query: str, intent: Dict, schema: str) -> str:
#        """Generate SQL query based on user query, intent, and schema"""
#        
#        prompt = f"""You are a SQL expert. Generate a MySQL SELECT query.
#
#DATABASE SCHEMA:
#{schema[:4000]}
#
#USER QUERY: {user_query}
#INTENT: {intent['intent']}
#METRICS: {intent['metrics']}
#DIMENSIONS: {intent['dimensions']}
#TIME RANGE: {intent['time_range']}
#
#GENERATE A SQL QUERY THAT:
#1. Answers the user's question accurately
#2. Uses proper MySQL syntax
#3. Includes appropriate aggregations if needed
#4. Groups data when analyzing by dimensions
#5. Orders results meaningfully
#6. Limits results if returning many rows
#
#RULES:
#- Only SELECT queries, no modifications
#- Use table and column names exactly as in schema
#- Handle dates properly if time_range is specified
#
#Output ONLY the SQL query, nothing else. No explanations, no markdown, just SQL:"""
#        
#        try:
#            response = self.llm.invoke(prompt)
#            sql = response.content.strip()
#            
#            # Clean SQL (remove markdown if present)
#            if sql.startswith("```sql"):
#                sql = sql[6:]
#            if sql.startswith("```"):
#                sql = sql[3:]
#            if sql.endswith("```"):
#                sql = sql[:-3]
#            
#            return sql.strip()
#            
#        except Exception as e:
#            print(f"❌ SQL generation error: {e}")
#            return self._get_fallback_sql(intent)
#    
#    def _get_fallback_sql(self, intent: Dict) -> str:
#        """Simple fallback SQL if LLM fails"""
#        if intent['intent'] == 'KPI':
#            return "SELECT COUNT(*) as count FROM store"
#        elif intent['intent'] == 'COMPARISON':
#            return "SELECT Category, COUNT(*) as count FROM store GROUP BY Category"
#        else:
#            return "SELECT * FROM store LIMIT 10"
#
#class LangChainAgent:
#    def __init__(self):
#        # Initialize LLM for the agent
#        self.llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama2"),
#            temperature=0.1
#        )
#        
#        # Initialize all components
#        self.schema_extractor = SchemaExtractor()
#        self.intent_classifier = IntentClassifier()
#        self.sql_validator = SQLValidator()
#        self.sql_generator = SQLGenerator()
#        self.db_tool = DatabaseTool()
#        
#        # Create tools for LangChain
#        self.tools = self._create_tools()
#        
#        # Create memory for conversation
#        self.memory = ConversationBufferMemory(
#            memory_key="chat_history",
#            return_messages=True
#        )
#        
#        # Create agent executor
#        self.agent_executor = self._create_agent_executor()
#    
#    def _create_tools(self) -> List[Tool]:
#        """Create LangChain tools"""
#        
#        def execute_sql_tool(sql_query: str) -> str:
#            """Execute SQL and return results"""
#            result = self.db_tool.execute_query(sql_query)
#            if result["success"]:
#                return json.dumps(result, indent=2)
#            else:
#                return f"Error: {result['error']}"
#        
#        def get_schema_tool() -> str:
#            """Get database schema"""
#            return self.schema_extractor.get_formatted_schema()
#        
#        def validate_sql_tool(sql_query: str) -> str:
#            """Validate SQL query"""
#            result = self.sql_validator.validate(sql_query)
#            if result == "VALID_SQL":
#                return "VALID_SQL: Query is safe to execute"
#            else:
#                return "INVALID_SQL: Query contains dangerous operations or syntax errors"
#        
#        def classify_intent_tool(user_query: str) -> str:
#            """Classify user intent"""
#            result = self.intent_classifier.classify(user_query)
#            return json.dumps(result, indent=2)
#        
#        def generate_sql_tool(user_query: str, intent_json: str) -> str:
#            """Generate SQL query"""
#            try:
#                intent = json.loads(intent_json)
#                schema = self.schema_extractor.get_formatted_schema()
#                sql = self.sql_generator.generate_sql(user_query, intent, schema)
#                return sql
#            except Exception as e:
#                return f"Error generating SQL: {e}"
#        
#        # Create Tool objects
#        return [
#            Tool(
#                name="get_database_schema",
#                func=get_schema_tool,
#                description="Get the complete database schema with column descriptions. Use this to understand the database structure before generating SQL."
#            ),
#            Tool(
#                name="classify_user_intent",
#                func=classify_intent_tool,
#                description="Classify user query intent (KPI, REPORT, TREND, COMPARISON, DISTRIBUTION). Use this first to understand what the user wants."
#            ),
#            Tool(
#                name="generate_sql_query",
#                func=generate_sql_tool,
#                description="Generate SQL query based on user query and intent. First get intent using classify_user_intent, then generate SQL."
#            ),
#            Tool(
#                name="validate_sql_query",
#                func=validate_sql_tool,
#                description="Validate if SQL query is safe. Returns VALID_SQL or INVALID_SQL. Use this before executing any SQL."
#            ),
#            Tool(
#                name="execute_sql_query",
#                func=execute_sql_tool,
#                description="Execute a SQL SELECT query and return results. Only use this after validating the SQL."
#            )
#        ]
#    
#    def _create_agent_executor(self) -> AgentExecutor:
#        """Create LangChain agent executor with proper workflow"""
#        
#        # Get prompt from hub (or use custom)
#        try:
#            prompt = hub.pull("hwchase17/react")
#        except:
#            # Fallback custom prompt
#            prompt = ChatPromptTemplate.from_messages([
#                ("system", """You are a SQL Assistant. Follow this EXACT workflow:
#
#                1. Use 'classify_user_intent' to understand what user wants
#                2. Use 'get_database_schema' to see database structure
#                3. Use 'generate_sql_query' to create SQL (pass user query and intent)
#                4. Use 'validate_sql_query' to check SQL safety
#                5. If valid, use 'execute_sql_query' to run it
#                6. Show results to user
#
#                Rules:
#                - Always show the generated SQL query to user
#                - Only proceed to execution if SQL is VALID
#                - If SQL is INVALID, explain why and suggest fix
#                - Only use SELECT queries"""),
#                ("human", "{input}"),
#                MessagesPlaceholder(variable_name="agent_scratchpad"),
#            ])
#        
#        # Create React agent
#        agent = create_react_agent(
#            llm=self.llm,
#            tools=self.tools,
#            prompt=prompt
#        )
#        
#        # Create executor
#        return AgentExecutor(
#            agent=agent,
#            tools=self.tools,
#            memory=self.memory,
#            verbose=True,
#            max_iterations=5,
#            handle_parsing_errors=True,
#            return_intermediate_steps=True
#        )
#    
#    def process_query_with_agent(self, user_query: str) -> Dict[str, Any]:
#        """
#        Process query using LangChain agent
#        Returns: {
#            "status": "success/error",
#            "response": "...",
#            "intermediate_steps": [...]
#        }
#        """
#        try:
#            print(f"\n{'='*60}")
#            print(f"Processing query with LangChain Agent: {user_query}")
#            print('='*60)
#            
#            # Use the LangChain agent to handle everything
#            result = self.agent_executor.invoke({"input": user_query})
#            
#            print(f"\n✅ Agent execution complete")
#            print(f"Output: {result['output'][:200]}...")
#            
#            return {
#                "status": "success",
#                "response": result['output'],
#                "intermediate_steps": result.get('intermediate_steps', []),
#                "chat_history": str(self.memory.chat_memory.messages)
#            }
#            
#        except Exception as e:
#            print(f"\n❌ Error in process_query_with_agent: {e}")
#            return {
#                "status": "error",
#                "response": f"Agent error: {str(e)}",
#                "intermediate_steps": [],
#                "chat_history": ""
#            }
#    
#    def process_query(self, user_query: str) -> Dict[str, Any]:
#        """
#        Main method for backward compatibility
#        Uses LangChain agent for processing
#        """
#        return self.process_query_with_agent(user_query)
#
## Test
#if __name__ == "__main__":
#    print("Testing LangChain Agent...")
#    
#    # Check Ollama is running
#    try:
#        llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama2")
#        )
#        test_response = llm.invoke("Test")
#        print("✅ Ollama is running")
#    except:
#        print("❌ Ollama is not running. Please start it with: ollama serve")
#        exit(1)
#    
#    # Test queries
#    agent = LangChainAgent()
#    
#    test_queries = [
#        "Show total sales",
#        "Sales by category",
#    ]
#    
#    for i, query in enumerate(test_queries, 1):
#        print(f"\n{'='*60}")
#        print(f"TEST {i}: {query}")
#        print('='*60)
#        
#        result = agent.process_query_with_agent(query)
#        
#        print(f"\n📋 RESULT SUMMARY:")
#        print(f"Status: {result['status']}")
#        print(f"Response length: {len(result['response'])} chars")
#        
#        if result['intermediate_steps']:
#            print(f"\n🔧 Intermediate Steps: {len(result['intermediate_steps'])}")
#        
#        print(f"\n✅ Test {i} complete")







#"""
#Complete LangChain Agent for SQL Generation
#Generates SQL queries using LLM
#"""
#import os
#import json
#from typing import Dict, Any, List
#
## LANGCHAIN 1.2.7 IMPORTS
##from langchain.agents import AgentExecutor, create_openai_tools_agent
##from langchain_community.chat_models import ChatOllama
##from langchain_core.tools import Tool
##from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
##from langchain.memory import ConversationBufferMemory
#
#
#from langchain.agents import AgentExecutor, create_react_agent
#from langchain_community.chat_models import ChatOllama
#from langchain_core.tools import Tool
#from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#from langchain.memory import ConversationBufferMemory
#from langchain import hub
#
#from schema_extractor import SchemaExtractor
#from intent_classifier import IntentClassifier
#from sql_validator import SQLValidator
#
#import mysql.connector
#import pandas as pd
#from dotenv import load_dotenv
#
#load_dotenv()
#
#class DatabaseTool:
#    """Database execution tool"""
#    
#    def __init__(self):
#        self.host = os.getenv("DB_HOST", "localhost")
#        self.port = int(os.getenv("DB_PORT", 3306))
#        self.user = os.getenv("DB_USER", "root")
#        self.password = os.getenv("DB_PASSWORD", "")
#        self.database = os.getenv("DB_NAME", "")
#    
#    def execute_query(self, sql: str) -> Dict[str, Any]:
#        """Execute SQL query and return structured results"""
#        try:
#            conn = mysql.connector.connect(
#                host=self.host,
#                port=self.port,
#                user=self.user,
#                password=self.password,
#                database=self.database
#            )
#            
#            df = pd.read_sql(sql, conn)
#            conn.close()
#            
#            return {
#                "success": True,
#                "row_count": len(df),
#                "column_count": len(df.columns),
#                "columns": list(df.columns),
#                "data": df.head(10).to_dict('records'),
#                "summary": f"Found {len(df)} rows with {len(df.columns)} columns"
#            }
#            
#        except Exception as e:
#            return {
#                "success": False,
#                "error": str(e)
#            }
#
#class SQLGenerator:
#    """Generates SQL queries using LLM"""
#    
#    def __init__(self):
#        self.llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama2"),
#            temperature=0.1
#        )
#    
#    def generate_sql(self, user_query: str, intent: Dict, schema: str) -> str:
#        """Generate SQL query based on user query, intent, and schema"""
#        
#        prompt = f"""You are a SQL expert. Generate a MySQL SELECT query.
#
#DATABASE SCHEMA:
#{schema[:4000]}
#
#USER QUERY: {user_query}
#INTENT: {intent['intent']}
#METRICS: {intent['metrics']}
#DIMENSIONS: {intent['dimensions']}
#TIME RANGE: {intent['time_range']}
#
#GENERATE A SQL QUERY THAT:
#1. Answers the user's question accurately
#2. Uses proper MySQL syntax
#3. Includes appropriate aggregations if needed
#4. Groups data when analyzing by dimensions
#5. Orders results meaningfully
#6. Limits results if returning many rows
#
#RULES:
#- Only SELECT queries, no modifications
#- Use table and column names exactly as in schema
#- Handle dates properly if time_range is specified
#
#Output ONLY the SQL query, nothing else. No explanations, no markdown, just SQL:"""
#        
#        try:
#            response = self.llm.invoke(prompt)
#            sql = response.content.strip()
#            
#            # Clean SQL (remove markdown if present)
#            if sql.startswith("```sql"):
#                sql = sql[6:]
#            if sql.startswith("```"):
#                sql = sql[3:]
#            if sql.endswith("```"):
#                sql = sql[:-3]
#            
#            return sql.strip()
#            
#        except Exception as e:
#            print(f"❌ SQL generation error: {e}")
#            return self._get_fallback_sql(intent)
#    
#    def _get_fallback_sql(self, intent: Dict) -> str:
#        """Simple fallback SQL if LLM fails"""
#        if intent['intent'] == 'KPI':
#            return "SELECT COUNT(*) as count FROM store"
#        elif intent['intent'] == 'COMPARISON':
#            return "SELECT Category, COUNT(*) as count FROM store GROUP BY Category"
#        else:
#            return "SELECT * FROM store LIMIT 10"
#
#class LangChainAgent:
#    def __init__(self):
#        # Initialize LLM for the agent
#        self.llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama2"),
#            temperature=0.1
#        )
#        
#        # Initialize all components
#        self.schema_extractor = SchemaExtractor()
#        self.intent_classifier = IntentClassifier()
#        self.sql_validator = SQLValidator()
#        self.sql_generator = SQLGenerator()
#        self.db_tool = DatabaseTool()
#        
#        # Create tools for LangChain
#        self.tools = self._create_tools()
#        
#        # Create memory for conversation
#        self.memory = ConversationBufferMemory(
#            memory_key="chat_history",
#            return_messages=True
#        )
#        
#        # Create agent executor
#        self.agent_executor = self._create_agent_executor()
#    
#    def _create_tools(self) -> List[Tool]:
#        """Create LangChain tools"""
#        
#        def execute_sql_tool(sql_query: str) -> str:
#            """Execute SQL and return results"""
#            result = self.db_tool.execute_query(sql_query)
#            if result["success"]:
#                return json.dumps(result, indent=2)
#            else:
#                return f"Error: {result['error']}"
#        
#        def get_schema_tool() -> str:
#            """Get database schema"""
#            return self.schema_extractor.get_formatted_schema()
#        
#        def validate_sql_tool(sql_query: str) -> str:
#            """Validate SQL query"""
#            result = self.sql_validator.validate(sql_query)
#            if result == "VALID_SQL":
#                return "VALID_SQL: Query is safe to execute"
#            else:
#                return "INVALID_SQL: Query contains dangerous operations or syntax errors"
#        
#        def classify_intent_tool(user_query: str) -> str:
#            """Classify user intent"""
#            result = self.intent_classifier.classify(user_query)
#            return json.dumps(result, indent=2)
#        
#        def generate_sql_tool(user_query: str, intent_json: str) -> str:
#            """Generate SQL query"""
#            try:
#                intent = json.loads(intent_json)
#                schema = self.schema_extractor.get_formatted_schema()
#                sql = self.sql_generator.generate_sql(user_query, intent, schema)
#                return sql
#            except Exception as e:
#                return f"Error generating SQL: {e}"
#        
#        # Create Tool objects
#        return [
#            Tool(
#                name="get_database_schema",
#                func=get_schema_tool,
#                description="Get the complete database schema with column descriptions. Use this to understand the database structure before generating SQL."
#            ),
#            Tool(
#                name="classify_user_intent",
#                func=classify_intent_tool,
#                description="Classify user query intent (KPI, REPORT, TREND, COMPARISON, DISTRIBUTION). Use this first to understand what the user wants."
#            ),
#            Tool(
#                name="generate_sql_query",
#                func=generate_sql_tool,
#                description="Generate SQL query based on user query and intent. First get intent using classify_user_intent, then generate SQL."
#            ),
#            Tool(
#                name="validate_sql_query",
#                func=validate_sql_tool,
#                description="Validate if SQL query is safe. Returns VALID_SQL or INVALID_SQL. Use this before executing any SQL."
#            ),
#            Tool(
#                name="execute_sql_query",
#                func=execute_sql_tool,
#                description="Execute a SQL SELECT query and return results. Only use this after validating the SQL."
#            )
#        ]
#    
#    def _create_agent_executor(self) -> AgentExecutor:
#        """Create LangChain agent executor with proper workflow"""
#        
#        # System prompt defining the exact workflow
#        system_prompt = """You are a SQL Assistant. Follow this EXACT workflow:
#
#        WORKFLOW STEPS:
#        1. ALWAYS use 'classify_user_intent' first to understand what user wants
#        2. Use 'get_database_schema' to see database structure
#        3. Use 'generate_sql_query' to create SQL (pass user query and intent)
#        4. Use 'validate_sql_query' to check SQL safety
#        5. If valid, use 'execute_sql_query' to run it
#        6. Show results to user in a clean, readable format
#
#        RULES:
#        - ALWAYS show the generated SQL query to user
#        - ONLY proceed to execution if SQL is VALID
#        - If SQL is INVALID, explain why and suggest fix
#        - Only use SELECT queries
#        - Use the tools in the correct order
#        - Always provide helpful explanations to the user
#        """
#        
#        prompt = ChatPromptTemplate.from_messages([
#            ("system", system_prompt),
#            MessagesPlaceholder(variable_name="chat_history"),
#            ("human", "{input}"),
#            MessagesPlaceholder(variable_name="agent_scratchpad"),
#        ])
#        
#        # Create agent
#        agent = create_openai_tools_agent(
#            llm=self.llm,
#            tools=self.tools,
#            prompt=prompt
#        )
#        
#        # Create executor
#        return AgentExecutor(
#            agent=agent,
#            tools=self.tools,
#            memory=self.memory,
#            verbose=True,
#            max_iterations=5,
#            handle_parsing_errors=True,
#            return_intermediate_steps=True
#        )
#    
#    def process_query_with_agent(self, user_query: str) -> Dict[str, Any]:
#        """
#        Process query using LangChain agent
#        Returns: {
#            "status": "success/error",
#            "response": "...",
#            "intermediate_steps": [...]
#        }
#        """
#        try:
#            print(f"\n{'='*60}")
#            print(f"Processing query with LangChain Agent: {user_query}")
#            print('='*60)
#            
#            # Use the LangChain agent to handle everything
#            result = self.agent_executor.invoke({"input": user_query})
#            
#            print(f"\n✅ Agent execution complete")
#            print(f"Output: {result['output'][:200]}...")
#            
#            return {
#                "status": "success",
#                "response": result['output'],
#                "intermediate_steps": result.get('intermediate_steps', []),
#                "chat_history": str(self.memory.chat_memory.messages)
#            }
#            
#        except Exception as e:
#            print(f"\n❌ Error in process_query_with_agent: {e}")
#            return {
#                "status": "error",
#                "response": f"Agent error: {str(e)}",
#                "intermediate_steps": [],
#                "chat_history": ""
#            }
#    
#    def process_query(self, user_query: str) -> Dict[str, Any]:
#        """
#        Main method for backward compatibility
#        Uses LangChain agent for processing
#        """
#        return self.process_query_with_agent(user_query)
#
## Simple standalone function for testing
#def simple_sql_generation(user_query: str) -> Dict[str, Any]:
#    """Simple function for app.py to use"""
#    agent = LangChainAgent()
#    return agent.process_query(user_query)
#
## Test
#if __name__ == "__main__":
#    print("Testing LangChain Agent...")
#    
#    # Check Ollama is running
#    try:
#        llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama2")
#        )
#        test_response = llm.invoke("Test")
#        print("✅ Ollama is running")
#    except:
#        print("❌ Ollama is not running. Please start it with: ollama serve")
#        exit(1)
#    
#    # Test queries with LangChain agent
#    print("\n" + "="*60)
#    print("Testing LangChain Agent Workflow...")
#    print("="*60)
#    
#    agent = LangChainAgent()
#    
#    test_queries = [
#        "Show total sales",
#        "Sales by category",
#        "Compare profit by region",
#        "Orders trend last month"
#    ]
#    
#    for i, query in enumerate(test_queries, 1):
#        print(f"\n{'='*60}")
#        print(f"TEST {i}: {query}")
#        print('='*60)
#        
#        result = agent.process_query_with_agent(query)
#        
#        print(f"\n📋 RESULT SUMMARY:")
#        print(f"Status: {result['status']}")
#        print(f"Response length: {len(result['response'])} chars")
#        
#        if result['intermediate_steps']:
#            print(f"\n🔧 Intermediate Steps ({len(result['intermediate_steps'])}):")
#            for j, step in enumerate(result['intermediate_steps'], 1):
#                action = step[0]
#                observation = step[1]
#                print(f"\n  Step {j}:")
#                print(f"    Action: {action.tool}")
#                if len(str(observation)) > 200:
#                    print(f"    Observation: {str(observation)[:200]}...")
#                else:
#                    print(f"    Observation: {observation}")
#        
#        print(f"\n✅ Test {i} complete")









#"""
#Complete LangChain Agent for SQL Generation
#Generates SQL queries using LLM
#"""
#import os
#import json
#from typing import Dict, Any, List
#
## LANGCHAIN 1.2.7 IMPORTS
#from langchain.agents import AgentExecutor, create_openai_tools_agent
#from langchain_community.chat_models import ChatOllama
#from langchain_core.tools import Tool
#from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#from langchain.memory import ConversationBufferMemory
##from langchain.agents import Tool, AgentExecutor
##from langchain.agents import create_openai_tools_agent
##from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
##from langchain.memory import ConversationBufferMemory
##from langchain_community.chat_models import ChatOllama
##from langchain.agents.agent import AgentExecutor
##from langchain.agents import create_openai_tools_agent
##from langchain_openai import ChatOpenAI
#
#
## FIXED IMPORTS:
##from langchain.agents import AgentExecutor, create_openai_tools_agent
##from langchain_core.tools import Tool
##from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
##from langchain.memory import ConversationBufferMemory
##from langchain_community.chat_models import ChatOllama
#
#
#from schema_extractor import SchemaExtractor
#from intent_classifier import IntentClassifier
#from sql_validator import SQLValidator
#
#import mysql.connector
#import pandas as pd
#from dotenv import load_dotenv
#
#load_dotenv()
#
#class DatabaseTool:
#    """Database execution tool"""
#    
#    def __init__(self):
#        self.host = os.getenv("DB_HOST", "localhost")
#        self.port = int(os.getenv("DB_PORT", 3306))
#        self.user = os.getenv("DB_USER", "root")
#        self.password = os.getenv("DB_PASSWORD", "")
#        self.database = os.getenv("DB_NAME", "")
#    
#    def execute_query(self, sql: str) -> Dict[str, Any]:
#        """Execute SQL query and return structured results"""
#        try:
#            conn = mysql.connector.connect(
#                host=self.host,
#                port=self.port,
#                user=self.user,
#                password=self.password,
#                database=self.database
#            )
#            
#            df = pd.read_sql(sql, conn)
#            conn.close()
#            
#            return {
#                "success": True,
#                "row_count": len(df),
#                "column_count": len(df.columns),
#                "columns": list(df.columns),
#                "data": df.head(10).to_dict('records'),
#                "summary": f"Found {len(df)} rows with {len(df.columns)} columns"
#            }
#            
#        except Exception as e:
#            return {
#                "success": False,
#                "error": str(e)
#            }
#
#class SQLGenerator:
#    """Generates SQL queries using LLM"""
#    
#    def __init__(self):
#        self.llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama2"),
#            temperature=0.1
#        )
#    
#    def generate_sql(self, user_query: str, intent: Dict, schema: str) -> str:
#        """Generate SQL query based on user query, intent, and schema"""
#        
#        prompt = f"""You are a SQL expert. Generate a MySQL SELECT query.
#
#DATABASE SCHEMA:
#{schema[:4000]}
#
#USER QUERY: {user_query}
#INTENT: {intent['intent']}
#METRICS: {intent['metrics']}
#DIMENSIONS: {intent['dimensions']}
#TIME RANGE: {intent['time_range']}
#
#GENERATE A SQL QUERY THAT:
#1. Answers the user's question accurately
#2. Uses proper MySQL syntax
#3. Includes appropriate aggregations if needed
#4. Groups data when analyzing by dimensions
#5. Orders results meaningfully
#6. Limits results if returning many rows
#
#RULES:
#- Only SELECT queries, no modifications
#- Use table and column names exactly as in schema
#- Handle dates properly if time_range is specified
#
#Output ONLY the SQL query, nothing else. No explanations, no markdown, just SQL:"""
#        
#        try:
#            response = self.llm.invoke(prompt)
#            sql = response.content.strip()
#            
#            # Clean SQL (remove markdown if present)
#            if sql.startswith("```sql"):
#                sql = sql[6:]
#            if sql.startswith("```"):
#                sql = sql[3:]
#            if sql.endswith("```"):
#                sql = sql[:-3]
#            
#            return sql.strip()
#            
#        except Exception as e:
#            print(f"❌ SQL generation error: {e}")
#            return self._get_fallback_sql(intent)
#    
#    def _get_fallback_sql(self, intent: Dict) -> str:
#        """Simple fallback SQL if LLM fails"""
#        if intent['intent'] == 'KPI':
#            return "SELECT COUNT(*) as count FROM store"
#        elif intent['intent'] == 'COMPARISON':
#            return "SELECT Category, COUNT(*) as count FROM store GROUP BY Category"
#        else:
#            return "SELECT * FROM store LIMIT 10"
#
#class CompleteLangChainAgent:
#    def __init__(self):
#        # Initialize LLM for the agent
#        self.llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama2"),
#            temperature=0.1
#        )
#        
#        # Initialize all components
#        self.schema_extractor = SchemaExtractor()
#        self.intent_classifier = IntentClassifier()
#        self.sql_validator = SQLValidator()
#        self.sql_generator = SQLGenerator()
#        self.db_tool = DatabaseTool()
#        
#        # Create tools for LangChain
#        self.tools = self._create_tools()
#        
#        # Create memory for conversation
#        self.memory = ConversationBufferMemory(
#            memory_key="chat_history",
#            return_messages=True
#        )
#        
#        # Create agent executor
#        self.agent_executor = self._create_agent_executor()
#    
#    def _create_tools(self) -> List[Tool]:
#        """Create LangChain tools"""
#        
#        def execute_sql_tool(sql_query: str) -> str:
#            """Execute SQL and return results"""
#            result = self.db_tool.execute_query(sql_query)
#            if result["success"]:
#                return json.dumps(result, indent=2)
#            else:
#                return f"Error: {result['error']}"
#        
#        def get_schema_tool() -> str:
#            """Get database schema"""
#            return self.schema_extractor.get_formatted_schema()
#        
#        def validate_sql_tool(sql_query: str) -> str:
#            """Validate SQL query"""
#            result = self.sql_validator.validate(sql_query)
#            if result == "VALID_SQL":
#                return "VALID_SQL: Query is safe to execute"
#            else:
#                return "INVALID_SQL: Query contains dangerous operations or syntax errors"
#        
#        def classify_intent_tool(user_query: str) -> str:
#            """Classify user intent"""
#            result = self.intent_classifier.classify(user_query)
#            return json.dumps(result, indent=2)
#        
#        def generate_sql_tool(user_query: str, intent_json: str) -> str:
#            """Generate SQL query"""
#            try:
#                intent = json.loads(intent_json)
#                schema = self.schema_extractor.get_formatted_schema()
#                sql = self.sql_generator.generate_sql(user_query, intent, schema)
#                return sql
#            except Exception as e:
#                return f"Error generating SQL: {e}"
#        
#        # Create Tool objects
#        return [
#            Tool(
#                name="get_database_schema",
#                func=get_schema_tool,
#                description="Get the complete database schema with column descriptions. Use this to understand the database structure before generating SQL."
#            ),
#            Tool(
#                name="classify_user_intent",
#                func=classify_intent_tool,
#                description="Classify user query intent (KPI, REPORT, TREND, COMPARISON, DISTRIBUTION). Use this first to understand what the user wants."
#            ),
#            Tool(
#                name="generate_sql_query",
#                func=generate_sql_tool,
#                description="Generate SQL query based on user query and intent. First get intent using classify_user_intent, then generate SQL."
#            ),
#            Tool(
#                name="validate_sql_query",
#                func=validate_sql_tool,
#                description="Validate if SQL query is safe. Returns VALID_SQL or INVALID_SQL. Use this before executing any SQL."
#            ),
#            Tool(
#                name="execute_sql_query",
#                func=execute_sql_tool,
#                description="Execute a SQL SELECT query and return results. Only use this after validating the SQL."
#            )
#        ]
#    
#    def _create_agent_executor(self) -> AgentExecutor:
#        """Create LangChain agent executor with proper workflow"""
#        
#        # System prompt defining the exact workflow
#        system_prompt = """You are a SQL Assistant. Follow this EXACT workflow:
#
#        WORKFLOW STEPS:
#        1. ALWAYS use 'classify_user_intent' first to understand what user wants
#        2. Use 'get_database_schema' to see database structure
#        3. Use 'generate_sql_query' to create SQL (pass user query and intent)
#        4. Use 'validate_sql_query' to check SQL safety
#        5. If valid, use 'execute_sql_query' to run it
#        6. Show results to user in a clean, readable format
#
#        RULES:
#        - ALWAYS show the generated SQL query to user
#        - ONLY proceed to execution if SQL is VALID
#        - If SQL is INVALID, explain why and suggest fix
#        - Only use SELECT queries
#        - Use the tools in the correct order
#        - Always provide helpful explanations to the user
#        """
#        
#        prompt = ChatPromptTemplate.from_messages([
#            ("system", system_prompt),
#            MessagesPlaceholder(variable_name="chat_history"),
#            ("human", "{input}"),
#            MessagesPlaceholder(variable_name="agent_scratchpad"),
#        ])
#        
#        # Create agent
#        agent = create_openai_tools_agent(
#            llm=self.llm,
#            tools=self.tools,
#            prompt=prompt
#        )
#        
#        # Create executor
#        return AgentExecutor(
#            agent=agent,
#            tools=self.tools,
#            memory=self.memory,
#            verbose=True,
#            max_iterations=5,
#            handle_parsing_errors=True,
#            return_intermediate_steps=True
#        )
#    
#    def process_query_with_agent(self, user_query: str) -> Dict[str, Any]:
#        """
#        Process query using LangChain agent
#        Returns: {
#            "status": "success/error",
#            "response": "...",
#            "intermediate_steps": [...]
#        }
#        """
#        try:
#            print(f"\n{'='*60}")
#            print(f"Processing query with LangChain Agent: {user_query}")
#            print('='*60)
#            
#            # Use the LangChain agent to handle everything
#            result = self.agent_executor.invoke({"input": user_query})
#            
#            print(f"\n✅ Agent execution complete")
#            print(f"Output: {result['output'][:200]}...")
#            
#            return {
#                "status": "success",
#                "response": result['output'],
#                "intermediate_steps": result.get('intermediate_steps', []),
#                "chat_history": str(self.memory.chat_memory.messages)
#            }
#            
#        except Exception as e:
#            print(f"\n❌ Error in process_query_with_agent: {e}")
#            return {
#                "status": "error",
#                "response": f"Agent error: {str(e)}",
#                "intermediate_steps": [],
#                "chat_history": ""
#            }
#    
#    def process_query(self, user_query: str) -> Dict[str, Any]:
#        """
#        Main method for backward compatibility
#        Uses LangChain agent for processing
#        """
#        return self.process_query_with_agent(user_query)
#
## Simple standalone function for testing
#def simple_sql_generation(user_query: str) -> Dict[str, Any]:
#    """Simple function for app.py to use"""
#    agent = CompleteLangChainAgent()
#    return agent.process_query(user_query)
#
## Test
#if __name__ == "__main__":
#    print("Testing Complete LangChain Agent...")
#    
#    # Check Ollama is running
#    try:
#        llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama2")
#        )
#        test_response = llm.invoke("Test")
#        print("✅ Ollama is running")
#    except:
#        print("❌ Ollama is not running. Please start it with: ollama serve")
#        exit(1)
#    
#    # Test queries with LangChain agent
#    print("\n" + "="*60)
#    print("Testing LangChain Agent Workflow...")
#    print("="*60)
#    
#    agent = CompleteLangChainAgent()
#    
#    test_queries = [
#        "Show total sales",
#        "Sales by category",
#        "Compare profit by region",
#        "Orders trend last month"
#    ]
#    
#    for i, query in enumerate(test_queries, 1):
#        print(f"\n{'='*60}")
#        print(f"TEST {i}: {query}")
#        print('='*60)
#        
#        result = agent.process_query_with_agent(query)
#        
#        print(f"\n📋 RESULT SUMMARY:")
#        print(f"Status: {result['status']}")
#        print(f"Response length: {len(result['response'])} chars")
#        
#        if result['intermediate_steps']:
#            print(f"\n🔧 Intermediate Steps ({len(result['intermediate_steps'])}):")
#            for j, step in enumerate(result['intermediate_steps'], 1):
#                action = step[0]
#                observation = step[1]
#                print(f"\n  Step {j}:")
#                print(f"    Action: {action.tool}")
#                if len(str(observation)) > 200:
#                    print(f"    Observation: {str(observation)[:200]}...")
#                else:
#                    print(f"    Observation: {observation}")
#        
#        print(f"\n✅ Test {i} complete")














#"""
#Complete LangChain Agent for SQL Generation
#Generates SQL queries using LLM
#"""
#import os
#import json
#from typing import Dict, Any, List
#from langchain.agents import Tool, AgentExecutor
#from langchain.agents import create_openai_tools_agent
#from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
#from langchain.memory import ConversationBufferMemory
#from langchain_community.chat_models import ChatOllama
#
#from schema_extractor import SchemaExtractor
#from intent_classifier import IntentClassifier
#from sql_validator import SQLValidator
#
#import mysql.connector
#import pandas as pd
#from dotenv import load_dotenv
#
#load_dotenv()
#
#class DatabaseTool:
#    """Database execution tool"""
#    
#    def __init__(self):
#        self.host = os.getenv("DB_HOST", "localhost")
#        self.port = int(os.getenv("DB_PORT", 3306))
#        self.user = os.getenv("DB_USER", "root")
#        self.password = os.getenv("DB_PASSWORD", "")
#        self.database = os.getenv("DB_NAME", "")
#    
#    def execute_query(self, sql: str) -> Dict[str, Any]:
#        """Execute SQL query and return structured results"""
#        try:
#            conn = mysql.connector.connect(
#                host=self.host,
#                port=self.port,
#                user=self.user,
#                password=self.password,
#                database=self.database
#            )
#            
#            df = pd.read_sql(sql, conn)
#            conn.close()
#            
#            return {
#                "success": True,
#                "row_count": len(df),
#                "column_count": len(df.columns),
#                "columns": list(df.columns),
#                "data": df.head(10).to_dict('records'),
#                "summary": f"Found {len(df)} rows with {len(df.columns)} columns"
#            }
#            
#        except Exception as e:
#            return {
#                "success": False,
#                "error": str(e)
#            }
#
#class SQLGenerator:
#    """Generates SQL queries using LLM"""
#    
#    def __init__(self):
#        self.llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama2"),
#            temperature=0.1
#        )
#    
#    def generate_sql(self, user_query: str, intent: Dict, schema: str) -> str:
#        """Generate SQL query based on user query, intent, and schema"""
#        
#        prompt = f"""You are a SQL expert. Generate a MySQL SELECT query.
#
#DATABASE SCHEMA:
#{schema[:4000]}
#
#USER QUERY: {user_query}
#INTENT: {intent['intent']}
#METRICS: {intent['metrics']}
#DIMENSIONS: {intent['dimensions']}
#TIME RANGE: {intent['time_range']}
#
#GENERATE A SQL QUERY THAT:
#1. Answers the user's question accurately
#2. Uses proper MySQL syntax
#3. Includes appropriate aggregations if needed
#4. Groups data when analyzing by dimensions
#5. Orders results meaningfully
#6. Limits results if returning many rows
#
#RULES:
#- Only SELECT queries, no modifications
#- Use table and column names exactly as in schema
#- Handle dates properly if time_range is specified
#
#Output ONLY the SQL query, nothing else. No explanations, no markdown, just SQL:"""
#        
#        try:
#            response = self.llm.invoke(prompt)
#            sql = response.content.strip()
#            
#            # Clean SQL (remove markdown if present)
#            if sql.startswith("```sql"):
#                sql = sql[6:]
#            if sql.startswith("```"):
#                sql = sql[3:]
#            if sql.endswith("```"):
#                sql = sql[:-3]
#            
#            return sql.strip()
#            
#        except Exception as e:
#            print(f"❌ SQL generation error: {e}")
#            return self._get_fallback_sql(intent)
#    
#    def _get_fallback_sql(self, intent: Dict) -> str:
#        """Simple fallback SQL if LLM fails"""
#        if intent['intent'] == 'KPI':
#            return "SELECT COUNT(*) as count FROM store"
#        elif intent['intent'] == 'COMPARISON':
#            return "SELECT Category, COUNT(*) as count FROM store GROUP BY Category"
#        else:
#            return "SELECT * FROM store LIMIT 10"
#
#class CompleteLangChainAgent:
#    def __init__(self):
#        # Initialize LLM
#        self.llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama2"),
#            temperature=0.1
#        )
#        
#        # Initialize all components
#        self.schema_extractor = SchemaExtractor()
#        self.intent_classifier = IntentClassifier()
#        self.sql_validator = SQLValidator()
#        self.sql_generator = SQLGenerator()
#        self.db_tool = DatabaseTool()
#        
#        # Create tools for LangChain
#        self.tools = self._create_tools()
#        
#        # Create agent executor
#        self.agent_executor = self._create_agent_executor()
#    
#    def _create_tools(self) -> List[Tool]:
#        """Create LangChain tools"""
#        
#        def execute_sql_tool(sql_query: str) -> str:
#            """Execute SQL and return results"""
#            result = self.db_tool.execute_query(sql_query)
#            if result["success"]:
#                return json.dumps(result, indent=2)
#            else:
#                return f"Error: {result['error']}"
#        
#        def get_schema_tool() -> str:
#            """Get database schema"""
#            return self.schema_extractor.get_formatted_schema()
#        
#        def validate_sql_tool(sql_query: str) -> str:
#            """Validate SQL query"""
#            return self.sql_validator.validate(sql_query)
#        
#        def classify_intent_tool(user_query: str) -> str:
#            """Classify user intent"""
#            result = self.intent_classifier.classify(user_query)
#            return json.dumps(result, indent=2)
#        
#        def generate_sql_tool(user_query: str, intent_json: str) -> str:
#            """Generate SQL query"""
#            try:
#                intent = json.loads(intent_json)
#                schema = self.schema_extractor.get_formatted_schema()
#                sql = self.sql_generator.generate_sql(user_query, intent, schema)
#                return sql
#            except Exception as e:
#                return f"Error generating SQL: {e}"
#        
#        # Create Tool objects
#        return [
#            Tool(
#                name="get_database_schema",
#                func=get_schema_tool,
#                description="Get the complete database schema with column descriptions"
#            ),
#            Tool(
#                name="classify_user_intent",
#                func=classify_intent_tool,
#                description="Classify user query intent (KPI, REPORT, TREND, COMPARISON, DISTRIBUTION)"
#            ),
#            Tool(
#                name="generate_sql_query",
#                func=generate_sql_tool,
#                description="Generate SQL query based on user query and intent"
#            ),
#            Tool(
#                name="validate_sql_query",
#                func=validate_sql_tool,
#                description="Validate if SQL query is safe. Returns VALID_SQL or INVALID_SQL"
#            ),
#            Tool(
#                name="execute_sql_query",
#                func=execute_sql_tool,
#                description="Execute a SQL SELECT query and return results"
#            )
#        ]
#    
#    def _create_agent_executor(self) -> AgentExecutor:
#        """Create LangChain agent executor with proper workflow"""
#        
#        # System prompt defining the exact workflow
#        system_prompt = """You are a SQL Assistant. Follow this EXACT workflow:
#
#        WORKFLOW STEPS:
#        1. Use 'classify_user_intent' to understand what user wants
#        2. Use 'get_database_schema' to see database structure
#        3. Use 'generate_sql_query' to create SQL (pass user query and intent)
#        4. Use 'validate_sql_query' to check SQL safety
#        5. If valid, use 'execute_sql_query' to run it
#        6. Show results to user
#
#        RULES:
#        - Always show the generated SQL query to user
#        - Only proceed to execution if SQL is VALID
#        - If SQL is INVALID, explain why and suggest fix
#        - Only use SELECT queries
#        """
#        
#        prompt = ChatPromptTemplate.from_messages([
#            ("system", system_prompt),
#            MessagesPlaceholder(variable_name="chat_history"),
#            ("human", "{input}"),
#            MessagesPlaceholder(variable_name="agent_scratchpad"),
#        ])
#        
#        # Create agent
#        agent = create_openai_tools_agent(
#            llm=self.llm,
#            tools=self.tools,
#            prompt=prompt
#        )
#        
#        # Create executor
#        return AgentExecutor(
#            agent=agent,
#            tools=self.tools,
#            verbose=True,
#            max_iterations=5,
#            handle_parsing_errors=True,
#            return_intermediate_steps=True
#        )
#    
#    def process_query(self, user_query: str) -> Dict[str, Any]:
#        """
#        Complete workflow processing
#        Returns: {
#            "status": "success/error",
#            "intent": {...},
#            "generated_sql": "...",
#            "validation_result": "...",
#            "execution_result": {...},
#            "error": "..."
#        }
#        """
#        try:
#            print(f"\n{'='*60}")
#            print(f"Processing query: {user_query}")
#            print('='*60)
#            
#            # Step 1: Classify intent
#            print("🔍 Step 1: Classifying intent...")
#            intent_result = self.intent_classifier.classify(user_query)
#            print(f"   Intent: {intent_result['intent']}")
#            print(f"   Metrics: {intent_result['metrics']}")
#            print(f"   Dimensions: {intent_result['dimensions']}")
#            
#            # Step 2: Get schema
#            print("\n📊 Step 2: Getting schema...")
#            schema = self.schema_extractor.get_formatted_schema()
#            print(f"   Schema loaded ({len(schema)} chars)")
#            
#            # Step 3: Generate SQL
#            print("\n🤖 Step 3: Generating SQL...")
#            generated_sql = self.sql_generator.generate_sql(
#                user_query, intent_result, schema
#            )
#            print(f"   Generated SQL: {generated_sql[:100]}...")
#            
#            # Step 4: Validate SQL
#            print("\n✅ Step 4: Validating SQL...")
#            validation_result = self.sql_validator.validate(generated_sql)
#            print(f"   Validation: {validation_result}")
#            
#            # Step 5: Execute if valid
#            execution_result = None
#            if validation_result == "VALID_SQL":
#                print("\n🚀 Step 5: Executing SQL...")
#                execution_result = self.db_tool.execute_query(generated_sql)
#                if execution_result["success"]:
#                    print(f"   Success: {execution_result['summary']}")
#                else:
#                    print(f"   Error: {execution_result['error']}")
#            else:
#                print("\n❌ Skipping execution - SQL is invalid")
#            
#            # Return complete result
#            return {
#                "status": "success",
#                "intent": intent_result,
#                "generated_sql": generated_sql,
#                "validation_result": validation_result,
#                "execution_result": execution_result,
#                "schema_preview": schema[:500] + "..." if len(schema) > 500 else schema
#            }
#            
#        except Exception as e:
#            print(f"\n❌ Error in process_query: {e}")
#            return {
#                "status": "error",
#                "error": str(e),
#                "intent": None,
#                "generated_sql": "",
#                "validation_result": "",
#                "execution_result": None
#            }
#
## Simple standalone function for testing
#def simple_sql_generation(user_query: str) -> Dict[str, Any]:
#    """Simple function for app.py to use"""
#    agent = CompleteLangChainAgent()
#    return agent.process_query(user_query)
#
## Test
#if __name__ == "__main__":
#    print("Testing Complete LangChain Agent...")
#    
#    # Check Ollama is running
#    try:
#        llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama2")
#        )
#        test_response = llm.invoke("Test")
#        print("✅ Ollama is running")
#    except:
#        print("❌ Ollama is not running. Please start it with: ollama serve")
#        exit(1)
#    
#    # Test queries
#    test_queries = [
#        "Show total sales",
#        "Sales by category",
#        "Compare profit by region",
#        "Orders trend last month"
#    ]
#    
#    agent = CompleteLangChainAgent()
#    
#    for i, query in enumerate(test_queries, 1):
#        print(f"\n{'='*60}")
#        print(f"TEST {i}: {query}")
#        print('='*60)
#        
#        result = agent.process_query(query)
#        
#        print(f"\n📋 RESULT SUMMARY:")
#        print(f"Status: {result['status']}")
#        print(f"Intent: {result['intent']['intent']}")
#        print(f"Generated SQL: {result['generated_sql'][:100]}...")
#        print(f"Validation: {result['validation_result']}")
#        
#        if result['execution_result']:
#            if result['execution_result']['success']:
#                print(f"Execution: {result['execution_result']['summary']}")
#            else:
#                print(f"Execution Error: {result['execution_result']['error']}")
#        
#        print(f"\n✅ Test {i} complete")