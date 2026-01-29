"""
Insight Generator - Dynamic, using actual data from database queries
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

class InsightGenerator:
    def __init__(self):
        self.llm = ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            temperature=0.3
        )
    
    def generate_insights(self, analysis_result: Dict, original_query: str, intent: Dict) -> str:
        """
        Generate insights dynamically based on actual query results
        """
        # Get ACTUAL data from the analysis result
        data = analysis_result.get("original_data", [])
        
        if not data:
            return "## 📊 Key Insights\n\nNo data available for insights."
        
        # Create a prompt with ACTUAL data (dynamic, from the query)
        data_str = self._format_data_for_prompt(data[:10])  # First 10 rows
        
        prompt = f"""
Based on this ACTUAL data from the database query, generate insights:

USER QUERY: "{original_query}"
QUERY INTENT: {intent.get('intent', 'ANALYSIS')}
METRICS MENTIONED: {intent.get('metrics', [])}
DIMENSIONS MENTIONED: {intent.get('dimensions', [])}

ACTUAL QUERY RESULTS (showing {min(10, len(data))} of {len(data)} rows):
{data_str}

Generate 2-3 key insights following these STRICT rules:
1. Base insights ONLY on the actual data provided above
2. Do NOT make up regions, categories, or numbers that aren't in the data
3. Do NOT use examples like "North region" if North is not in the data
4. Highlight actual patterns, comparisons, or trends visible in the data
5. Include specific numbers from the data when possible
6. Use bullet points with relevant emojis
7. Be concise and business-focused
8. If the data shows rankings (highest to lowest), mention that
9. If percentages can be calculated, include them
10. Focus on what the data ACTUALLY shows

IMPORTANT: If you cannot create insights from the data, say "Based on the data:" 
and list what you see instead of making up information.

Now generate insights:
"""
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a precise data analyst who ONLY uses actual data. Never hallucinate or make up data. Always base insights on the exact data provided."),
                HumanMessage(content=prompt)
            ])
            
            insights = response.content.strip()
            
            # Validate insights don't contain hallucinations
            if self._contains_hallucinations(insights, data):
                print("⚠️ Detected possible hallucinations in insights, using fallback")
                return self._generate_fallback_insights(analysis_result)
            
            return f"## 📊 Key Insights\n\n{insights}"
            
        except Exception as e:
            print(f"❌ Insight generation error: {e}")
            return self._generate_fallback_insights(analysis_result)
    
    def _format_data_for_prompt(self, data: list) -> str:
        """Format data in a readable way for the prompt"""
        if not data:
            return "No data"
        
        formatted = []
        for i, row in enumerate(data, 1):
            # Format each row nicely
            items = []
            for key, value in row.items():
                # Clean up decimal values
                try:
                    # Try to format as number
                    num = float(value)
                    if abs(num) >= 1000:
                        # Format large numbers with commas
                        formatted_num = f"${num:,.0f}" if num >= 0 else f"-${abs(num):,.0f}"
                    else:
                        formatted_num = f"${num:,.2f}" if num >= 0 else f"-${abs(num):,.2f}"
                    items.append(f"{key}: {formatted_num}")
                except (ValueError, TypeError):
                    # Not a number, use as-is
                    items.append(f"{key}: {value}")
            
            formatted.append(f"Row {i}: {', '.join(items)}")
        
        return "\n".join(formatted)
    
    def _contains_hallucinations(self, insights: str, actual_data: list) -> bool:
        """Check if insights contain made-up data not in actual_data"""
        if not actual_data:
            return False
        
        # Extract all unique values from actual data
        actual_values = set()
        for row in actual_data:
            for key, value in row.items():
                if isinstance(value, str):
                    # Add string values
                    actual_values.add(value.strip().lower())
                    # Also add column names
                    actual_values.add(key.lower())
                else:
                    # Add string representation of other values
                    actual_values.add(str(value).lower())
        
        insights_lower = insights.lower()
        
        # Check for common hallucination patterns
        hallucination_indicators = [
            "for example", "such as", "e.g.", "i.e.",  # Generic examples
            "assume", "assuming", "probably", "likely",  # Assumptions
            "might be", "could be", "maybe",  # Uncertainties
            "north", "south", "east", "west", "central",  # Regions (only if not in data)
            "example data", "sample data", "hypothetical",  # Example references
        ]
        
        # Only flag if the indicator is present AND the actual data doesn't contain it
        for indicator in hallucination_indicators:
            if indicator in insights_lower:
                # Check if this is actually in the data
                if indicator not in actual_values:
                    # Also check if it's being used as an example
                    context_words = ["example", "such as", "e.g."]
                    if any(context in insights_lower for context in context_words):
                        return True
        
        return False
    
    def _generate_fallback_insights(self, analysis_result: Dict) -> str:
        """Generate insights without LLM (no hallucinations)"""
        data = analysis_result.get("original_data", [])
        
        if not data:
            return "## 📊 Key Insights\n\nNo data available for insights."
        
        insights = ["## 📊 Key Insights\n"]
        
        # Simple analysis based on actual data
        if len(data) > 0 and isinstance(data[0], dict):
            # Find numeric columns
            first_row = data[0]
            numeric_cols = []
            text_cols = []
            
            for k, v in first_row.items():
                try:
                    float(v)
                    numeric_cols.append(k)
                except (ValueError, TypeError):
                    text_cols.append(k)
            
            if numeric_cols:
                numeric_col = numeric_cols[0]  # Use first numeric column
                
                try:
                    # Calculate basic statistics
                    values = []
                    for row in data:
                        try:
                            values.append(float(row.get(numeric_col, 0)))
                        except:
                            values.append(0)
                    
                    if values:
                        total = sum(values)
                        avg = total / len(values) if len(values) > 0 else 0
                        max_val = max(values)
                        min_val = min(values)
                        
                        # Find which rows have max and min
                        max_row = None
                        min_row = None
                        for row in data:
                            try:
                                val = float(row.get(numeric_col, 0))
                                if val == max_val:
                                    max_row = row
                                if val == min_val:
                                    min_row = row
                            except:
                                pass
                        
                        # Get category for max/min if available
                        if text_cols and max_row and min_row:
                            category_col = text_cols[0]
                            max_category = max_row.get(category_col, "Item")
                            min_category = min_row.get(category_col, "Item")
                            
                            insights.append(f"• **Highest {numeric_col}**: {max_category} with ${max_val:,.2f} 📈")
                            insights.append(f"• **Lowest {numeric_col}**: {min_category} with ${min_val:,.2f} 📉")
                        else:
                            insights.append(f"• **Highest {numeric_col}**: ${max_val:,.2f} 📈")
                            insights.append(f"• **Lowest {numeric_col}**: ${min_val:,.2f} 📉")
                        
                        # Add total and average if appropriate
                        insights.append(f"• **Total {numeric_col}**: ${total:,.2f} 💰")
                        insights.append(f"• **Average {numeric_col}**: ${avg:,.2f} ⚖️")
                
                except Exception as e:
                    print(f"⚠️ Fallback insight calculation error: {e}")
                    insights.append("• Basic analysis completed")
        
        insights.append(f"\n• **Records analyzed**: {len(data)}")
        insights.append("• **Note**: Insights are based on the actual query results")
        
        return "\n".join(insights)


# Test the insight generator WITHOUT manual data
if __name__ == "__main__":
    print("🧪 Testing Insight Generator (Dynamic Mode)...")
    
    # Initialize generator
    generator = InsightGenerator()
    
    print("✅ Insight Generator initialized successfully!")
    print("ℹ️ This generator works dynamically with any query results")
    print("ℹ️ No manual data - everything comes from the database")
    
    # Example of how it would be called in the actual workflow:
    print("\n📋 Example workflow:")
    print("1. User asks: 'show total sales by region'")
    print("2. SQL is generated and executed")
    print("3. DataAnalyzer processes the results")
    print("4. InsightGenerator creates insights from ACTUAL data")
    print("5. ChartGenerator creates visualization")
    
    print("\n✅ Insight Generator is ready for dynamic use!")






##this worked we get the chart commenting just for getting things dynamically.
#"""
#Insight Generator - Fixed to use actual data
#"""
#import os
#from typing import Dict, Any
#from dotenv import load_dotenv
#from langchain_community.chat_models import ChatOllama
#from langchain_core.messages import HumanMessage, SystemMessage
#
#load_dotenv()
#
#class InsightGenerator:
#    def __init__(self):
#        self.llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
#            temperature=0.3
#        )
#    
#    def generate_insights(self, analysis_result: Dict, original_query: str, intent: Dict) -> str:
#        # Prepare REAL data for LLM (not hallucinations)
#        data_summary = self._prepare_real_data_summary(analysis_result)
#        
#        prompt = f"""
#You are a data analyst. Generate insights based on ACTUAL DATA only.
#
#USER QUERY: {original_query}
#INTENT: {intent.get('intent', 'REPORT')}
#
#ACTUAL DATA FROM DATABASE:
#{data_summary}
#
#Generate 2-3 key insights:
#1. Base insights ONLY on the actual data provided above
#2. Do NOT make up regions or numbers that aren't in the data
#3. Highlight actual patterns you see in the data
#4. Use bullet points with relevant emojis
#5. Be concise
#
#Example based on actual data:
#• West region has the highest sales at $725K (38% of total) 📈
#• South region has the lowest sales at $391K (21% of total) 📉
#
#Now generate insights for the actual data above:
#"""
#        
#        try:
#            response = self.llm.invoke([
#                SystemMessage(content="You are a precise data analyst who only uses actual data."),
#                HumanMessage(content=prompt)
#            ])
#            
#            insights = response.content.strip()
#            return f"## 📊 Key Insights\n\n{insights}"
#            
#        except Exception as e:
#            print(f"❌ Insight generation error: {e}")
#            return self._generate_fallback_insights(analysis_result)
#    
#    def _prepare_real_data_summary(self, analysis_result: Dict) -> str:
#        """Prepare actual data summary to prevent hallucinations"""
#        data = analysis_result.get("original_data", [])
#        columns = analysis_result.get("columns", [])
#        
#        if not data:
#            return "No data available"
#        
#        summary = []
#        summary.append(f"Total rows: {len(data)}")
#        summary.append(f"Columns: {', '.join(columns)}")
#        summary.append("\nActual Data:")
#        
#        # Show actual data
#        for i, row in enumerate(data[:5]):  # Show first 5 rows
#            row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
#            summary.append(f"Row {i+1}: {row_str}")
#        
#        return "\n".join(summary)
#    
#    def _generate_fallback_insights(self, analysis_result: Dict) -> str:
#        """Generate insights without LLM (no hallucinations)"""
#        data = analysis_result.get("original_data", [])
#        
#        if not data:
#            return "## 📊 Key Insights\n\nNo data available for insights."
#        
#        insights = ["## 📊 Key Insights\n"]
#        
#        # Simple analysis based on actual data
#        if len(data) > 0 and isinstance(data[0], dict):
#            # Find numeric columns
#            first_row = data[0]
#            numeric_cols = [k for k, v in first_row.items() 
#                          if isinstance(v, (int, float)) or 
#                          (isinstance(v, str) and v.replace('.', '', 1).isdigit())]
#            
#            if numeric_cols:
#                # Find max and min values
#                max_val = max(data, key=lambda x: float(x.get(numeric_cols[0], 0)))
#                min_val = min(data, key=lambda x: float(x.get(numeric_cols[0], 0)))
#                
#                insights.append(f"• **Highest value**: {max_val}")
#                insights.append(f"• **Lowest value**: {min_val}")
#        
#        insights.append(f"\n• **Total records analyzed**: {len(data)}")
#        insights.append("• **Note**: For deeper insights, try more specific queries")
#        
#        return "\n".join(insights)





# worked changed bcz chart not get in o/p
#"""
#Insight Generator Agent
#Generates natural language insights from analyzed data
#"""
#import os
#from typing import Dict, Any, List
#from dotenv import load_dotenv
#from langchain_community.chat_models import ChatOllama
#from langchain_core.messages import HumanMessage, SystemMessage
#
#load_dotenv()
#
#class InsightGenerator:
#    def __init__(self):
#        self.llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
#            temperature=0.3  # Slightly creative for insights
#        )
#    
#    def generate_insights(self, analysis_result: Dict, original_query: str, intent: Dict) -> str:
#        """
#        Generate natural language insights from analyzed data
#        """
#        # Prepare data for LLM
#        data_summary = self._prepare_data_summary(analysis_result)
#        
#        prompt = f"""
#You are a data analyst expert. Generate insightful observations from this data analysis.
#
#USER QUERY: {original_query}
#INTENT: {intent.get('intent', 'REPORT')}
#
#DATA ANALYSIS RESULTS:
#{data_summary}
#
#Generate 3-5 key insights in clear, business-friendly language:
#1. Start with the most important finding
#2. Include percentages and comparisons where relevant
#3. Highlight anomalies or interesting patterns
#4. Suggest possible actions or implications
#5. Keep each insight concise (1-2 sentences)
#
#Format the response as bullet points with emojis to make it engaging.
#
#Example format:
#• West region leads with $725K sales (42% of total) 📈
#• East shows steady growth at 15% month-over-month ↗️
#• Consider reallocating resources to high-performing regions 💡
#
#Now generate insights for the data above:
#"""
#        
#        try:
#            response = self.llm.invoke([
#                SystemMessage(content="You are an insightful data analyst."),
#                HumanMessage(content=prompt)
#            ])
#            
#            insights = response.content.strip()
#            
#            # Add header
#            formatted_insights = f"## 📊 Key Insights\n\n{insights}"
#            
#            return formatted_insights
#            
#        except Exception as e:
#            print(f"❌ Error generating insights: {e}")
#            return self._generate_fallback_insights(analysis_result)
#    
#    def _prepare_data_summary(self, analysis_result: Dict) -> str:
#        """Prepare data summary for LLM"""
#        summary_parts = []
#        
#        # Add basic info
#        summary_parts.append(f"Total Rows: {analysis_result.get('row_count', 0)}")
#        summary_parts.append(f"Columns: {', '.join(analysis_result.get('columns', []))}")
#        
#        # Add analysis results
#        analysis = analysis_result.get("analysis", {})
#        for analysis_type, analysis_data in analysis.items():
#            summary_parts.append(f"\n{analysis_type.upper()} Analysis:")
#            
#            if analysis_type == "comparison" and "data" in analysis_data:
#                data = analysis_data["data"]
#                for item in data[:5]:  # Show first 5 items
#                    if 'percentage' in item:
#                        summary_parts.append(f"  • {item.get(list(item.keys())[0], 'Item')}: {item.get('percentage', 0)}%")
#            
#            elif analysis_type == "kpi_metrics":
#                for metric, values in analysis_data.items():
#                    summary_parts.append(f"  • {metric}: Sum={values.get('sum', 0):,.2f}, Avg={values.get('average', 0):,.2f}")
#            
#            elif analysis_type == "trend":
#                if "growth_rates" in analysis_data:
#                    growth = analysis_data["growth_rates"]
#                    if growth:
#                        summary_parts.append(f"  • Growth rates: {growth}")
#                    if "total_growth" in analysis_data:
#                        summary_parts.append(f"  • Total growth: {analysis_data['total_growth']}%")
#            
#            elif analysis_type == "distribution":
#                if "percentages" in analysis_data:
#                    percentages = analysis_data["percentages"]
#                    for key, value in list(percentages.items())[:5]:
#                        summary_parts.append(f"  • {key}: {value}%")
#        
#        # Add chart data if available
#        chart_data = analysis_result.get("chart_data", {})
#        if chart_data.get("values"):
#            summary_parts.append(f"\nChart Data: {len(chart_data.get('values', []))} data points")
#            if chart_data.get("percentages"):
#                summary_parts.append(f"Percentages: {chart_data.get('percentages', [])}")
#        
#        return "\n".join(summary_parts)
#    
#    def _generate_fallback_insights(self, analysis_result: Dict) -> str:
#        """Generate simple insights if LLM fails"""
#        insights = ["## 📊 Key Insights\n"]
#        
#        analysis = analysis_result.get("analysis", {})
#        
#        if "comparison" in analysis:
#            comp_data = analysis["comparison"]
#            if "data" in comp_data and comp_data["data"]:
#                # Find max and min
#                data_items = comp_data["data"]
#                max_item = max(data_items, key=lambda x: x.get('percentage', 0))
#                min_item = min(data_items, key=lambda x: x.get('percentage', 0))
#                
#                insights.append(f"• **{list(max_item.keys())[0]}** has the highest share at {max_item.get('percentage', 0)}% 📈")
#                insights.append(f"• **{list(min_item.keys())[0]}** has the lowest share at {min_item.get('percentage', 0)}% 📉")
#        
#        elif "kpi_metrics" in analysis:
#            kpi_data = analysis["kpi_metrics"]
#            for metric, values in kpi_data.items():
#                insights.append(f"• **{metric}**: Total = {values.get('sum', 0):,.2f}, Average = {values.get('average', 0):,.2f}")
#        
#        elif "trend" in analysis:
#            trend_data = analysis["trend"]
#            if "total_growth" in trend_data:
#                growth = trend_data["total_growth"]
#                trend_emoji = "📈" if growth > 0 else "📉" if growth < 0 else "➡️"
#                insights.append(f"• Overall trend shows {growth}% growth {trend_emoji}")
#        
#        insights.append("\n💡 **Recommendation**: Consider focusing on top performers for better results.")
#        
#        return "\n".join(insights)