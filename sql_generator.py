#Not used this code
"""
AI SQL Generator using AUTO-EXTRACTED schema
"""
import os
from schema_extractor import SchemaExtractor
from intent_classifier import classify_intent

class AISQLGenerator:
    def __init__(self, use_openai=True):
        self.schema_extractor = SchemaExtractor()
        self.schema_context = self.schema_extractor.get_schema_text()
        
        if use_openai and os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = "gpt-3.5-turbo"
            self.use_ai = True
        else:
            self.use_ai = False
            print("⚠️ Using rule-based SQL generation (no OpenAI)")
    
    def generate_sql(self, user_query: str, intent: str) -> str:
        """
        Generate SQL using AI with auto-extracted schema
        """
        if self.use_ai:
            return self._generate_with_ai(user_query, intent)
        else:
            return self._generate_with_rules(user_query, intent)
    
    def _generate_with_ai(self, user_query: str, intent: str) -> str:
        """Generate SQL using OpenAI"""
        prompt = f"""
        You are a SQL expert. Generate a MySQL SELECT query.
        
        DATABASE SCHEMA:
        {self.schema_context}
        
        USER QUERY: {user_query}
        QUERY TYPE: {intent}
        
        Rules:
        1. Use ONLY SELECT queries
        2. Include proper GROUP BY, ORDER BY when needed
        3. Use LIMIT for large results
        4. Only output SQL, no explanations
        
        SQL Query:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            sql = response.choices[0].message.content.strip()
            
            # Clean SQL
            if sql.startswith("```sql"):
                sql = sql[6:]
            if sql.startswith("```"):
                sql = sql[3:]
            if sql.endswith("```"):
                sql = sql[:-3]
            
            return sql.strip()
            
        except Exception as e:
            print(f"❌ AI generation failed: {e}")
            return self._generate_with_rules(user_query, intent)
    
    def _generate_with_rules(self, user_query: str, intent: str) -> str:
        """Simple rule-based SQL generation"""
        query_lower = user_query.lower()
        
        # Get table names from schema
        schema = self.schema_extractor.extract_schema()
        table_names = list(schema['tables'].keys())
        main_table = table_names[0] if table_names else "store"
        
        # Get column names from first table
        columns = []
        if table_names:
            columns = list(schema['tables'][main_table]['columns'].keys())
        
        # Find relevant columns based on query
        numeric_cols = []
        category_cols = []
        date_cols = []
        
        for col in columns:
            if any(word in col.lower() for word in ['sales', 'profit', 'amount', 'price', 'cost', 'quantity']):
                numeric_cols.append(col)
            elif any(word in col.lower() for word in ['date', 'time', 'year', 'month']):
                date_cols.append(col)
            elif any(word in col.lower() for word in ['category', 'region', 'type', 'name', 'segment']):
                category_cols.append(col)
        
        # Generate SQL based on intent
        if intent == "KPI":
            if numeric_cols:
                col = numeric_cols[0]
                return f"SELECT SUM({col}) as Total_{col} FROM {main_table}"
            else:
                return f"SELECT COUNT(*) as Total_Records FROM {main_table}"
        
        elif intent == "TREND":
            if date_cols and numeric_cols:
                date_col = date_cols[0]
                num_col = numeric_cols[0]
                return f"SELECT {date_col}, SUM({num_col}) as Total FROM {main_table} GROUP BY {date_col} ORDER BY {date_col}"
            else:
                return f"SELECT * FROM {main_table} LIMIT 10"
        
        elif intent == "COMPARISON":
            if category_cols and numeric_cols:
                cat_col = category_cols[0]
                num_col = numeric_cols[0]
                return f"SELECT {cat_col}, SUM({num_col}) as Total FROM {main_table} GROUP BY {cat_col} ORDER BY Total DESC"
            else:
                return f"SELECT * FROM {main_table} LIMIT 10"
        
        else:
            return f"SELECT * FROM {main_table} LIMIT 10"

# Test
if __name__ == "__main__":
    print("Testing AI SQL Generator with auto schema...")
    
    # First, extract schema
    extractor = SchemaExtractor()
    schema_text = extractor.get_schema_text()
    print(f"Schema extracted: {len(schema_text)} chars")
    
    # Test SQL generation
    generator = AISQLGenerator(use_openai=False)
    
    test_queries = [
        "Show total sales",
        "Sales trend over time",
        "Compare by category"
    ]
    
    for query in test_queries:
        intent = classify_intent(query)
        sql = generator.generate_sql(query, intent)
        print(f"\nQuery: {query}")
        print(f"Intent: {intent}")
        print(f"SQL: {sql}")