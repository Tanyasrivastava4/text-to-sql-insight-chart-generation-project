"""
Dynamic Intent Classifier using LLM
Returns JSON as specified
"""
import json
import os
from langchain_community.chat_models import ChatOllama
#from langchain.schema import HumanMessage
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

class IntentClassifier:
    def __init__(self):
        self.llm = ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama2"),
            temperature=0
        )
    
    def classify(self, user_query: str) -> dict:
        """
        Classify user query using LLM
        Returns JSON with: intent, metrics, dimensions, time_range
        """
        prompt = f"""You are an intent classification agent.

Classify the user query into exactly ONE of the following intents:
- KPI: Asking for specific metrics (totals, averages, counts)
- REPORT: General reporting or listing data
- TREND: Time-based analysis, patterns over time
- COMPARISON: Comparing items, groups, or categories
- DISTRIBUTION: Spread, frequency, or distribution analysis

Also extract:
- metric(s) mentioned (e.g., sales, profit, quantity)
- dimension(s) mentioned (e.g., category, region, product)
- time reference (if any, e.g., "last month", "2023", "Q4")

Rules:
- Respond with ONLY valid JSON
- Do NOT include explanations
- Do NOT include extra fields

User Query: "{user_query}"

Required output format (JSON only):
{{
  "intent": "KPI | REPORT | TREND | COMPARISON | DISTRIBUTION",
  "metrics": [ ],
  "dimensions": [ ],
  "time_range": null
}}"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Extract JSON from response
            content = response.content.strip()
            
            # Find JSON in response
            if '{' in content and '}' in content:
                json_str = content[content.find('{'):content.rfind('}')+1]
                result = json.loads(json_str)
                
                # Validate required fields
                required = ["intent", "metrics", "dimensions", "time_range"]
                if all(field in result for field in required):
                    return result
            
            # Fallback to default
            return {
                "intent": "REPORT",
                "metrics": [],
                "dimensions": [],
                "time_range": None
            }
            
        except Exception as e:
            print(f"❌ Intent classification error: {e}")
            return {
                "intent": "REPORT",
                "metrics": [],
                "dimensions": [],
                "time_range": None
            }

# Test
if __name__ == "__main__":
    print("Testing intent classifier...")
    classifier = IntentClassifier()
    
    test_queries = [
        "Show total sales last month",
        "Compare profit by category",
        "Sales trend over time",
        "Distribution of order quantities",
        "List all products"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = classifier.classify(query)
        print(f"Result: {json.dumps(result, indent=2)}")









#"""
#Classify user queries into categories
#"""
#def classify_intent(user_query):
#    """
#    Classify the user's query into categories:
#    - KPI: Asking for specific metrics
#    - TREND: Time-based analysis
#    - COMPARISON: Comparing items
#    - DISTRIBUTION: Spread of data
#    - REPORT: General reporting
#    """
#    query_lower = user_query.lower()
#    
#    # KPI Queries (specific metrics)
#    kpi_keywords = ['total', 'sum', 'average', 'avg', 'count', 'maximum', 'minimum', 
#                    'max', 'min', 'how many', 'what is the', 'kpi', 'metric']
#    
#    # Trend Queries (time-based)
#    trend_keywords = ['trend', 'over time', 'over the years', 'monthly', 'weekly', 
#                     'daily', 'yearly', 'quarterly', 'growth', 'decline', 'forecast']
#    
#    # Comparison Queries
#    comparison_keywords = ['compare', 'vs', 'versus', 'difference', 'better', 'worse',
#                          'highest', 'lowest', 'top', 'bottom', 'rank']
#    
#    # Distribution Queries
#    distribution_keywords = ['distribution', 'spread', 'range', 'histogram', 'frequency',
#                           'percentile', 'quartile', 'standard deviation', 'variance']
#    
#    # Check each category
#    if any(keyword in query_lower for keyword in kpi_keywords):
#        return "KPI"
#    elif any(keyword in query_lower for keyword in trend_keywords):
#        return "TREND"
#    elif any(keyword in query_lower for keyword in comparison_keywords):
#        return "COMPARISON"
#    elif any(keyword in query_lower for keyword in distribution_keywords):
#        return "DISTRIBUTION"
#    else:
#        return "REPORT"  # Default for general queries
#
## Test function
#if __name__ == "__main__":
#    test_queries = [
#        "What is the total sales?",
#        "Show sales trend over time",
#        "Compare sales by region",
#        "Show distribution of profits",
#        "List all products"
#    ]
#    
#    print("Intent Classification Test:")
#    print("-" * 40)
#    for query in test_queries:
#        intent = classify_intent(query)
#        print(f"Query: '{query}'")
#        print(f"Intent: {intent}")
#        print()