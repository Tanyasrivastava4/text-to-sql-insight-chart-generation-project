"""
Database connection for Superstore - USING .ENV FILE
"""
import mysql.connector
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Database:
    def __init__(self):
        """Get connection details from .env file"""
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = int(os.getenv("DB_PORT", 3307))
        self.user = os.getenv("DB_USER", "root")
        self.password = os.getenv("DB_PASSWORD", "root")
        self.database = os.getenv("DB_NAME", "superstore")
        self.connection = None
    
    def connect(self):
        """Create database connection using .env values"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
            print(f"✅ Connected to {self.database} as {self.user}")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def run_query(self, sql_query):
        """Run SQL query and return results"""
        try:
            if not self.connection:
                self.connect()
            
            df = pd.read_sql(sql_query, self.connection)
            return df
            
        except Exception as e:
            print(f"❌ Query error: {e}")
            return None
    
    def get_table_columns(self):
        """Get column names from store table"""
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor()
            cursor.execute("DESCRIBE store")
            columns = [col[0] for col in cursor.fetchall()]
            return columns
            
        except Exception as e:
            print(f"❌ Error getting columns: {e}")
            return []

# Test the connection
if __name__ == "__main__":
    print("Testing database connection from .env file...")
    db = Database()
    
    if db.connect():
        print("✅ Success! Connection works.")
        
        # Show current user
        cursor = db.connection.cursor()
        cursor.execute("SELECT USER()")
        current_user = cursor.fetchone()[0]
        print(f"📋 Connected as: {current_user}")
        
        # Show tables
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print(f"📊 Tables in database: {[t[0] for t in tables]}")
    else:
        print("❌ Connection failed. Check .env file.")


































#"""
#Database connection and basic operations
#"""
#import mysql.connector
#import pandas as pd
#import os
#from dotenv import load_dotenv
#from typing import Optional, List, Dict, Any
#import json
#
#load_dotenv()
#
#class DatabaseManager:
#    def __init__(self):
#        self.connection = None
#        self.connect()
#    
#    def connect(self) -> bool:
#        """Establish database connection"""
#        try:
#            self.connection = mysql.connector.connect(
#                host=os.getenv("DB_HOST"),
#                user=os.getenv("DB_USER"),
#                password=os.getenv("DB_PASSWORD"),
#                database=os.getenv("DB_NAME"),
#                charset='utf8mb4'
#            )
#            print("✅ Database connected successfully")
#            return True
#        except Exception as e:
#            print(f"❌ Database connection failed: {e}")
#            return False
#    
#    def get_schema(self) -> Dict[str, Any]:
#        """
#        Get complete database schema including:
#        - Table names
#        - Column names and types
#        - Sample data
#        - Foreign keys
#        """
#        if not self.connection:
#            if not self.connect():
#                return {}
#        
#        schema = {
#            "database": os.getenv("DB_NAME"),
#            "tables": {}
#        }
#        
#        try:
#            cursor = self.connection.cursor(dictionary=True)
#            
#            # Get all tables
#            cursor.execute("SHOW TABLES")
#            tables = cursor.fetchall()
#            table_names = [list(table.values())[0] for table in tables]
#            
#            for table_name in table_names:
#                # Get columns
#                cursor.execute(f"DESCRIBE {table_name}")
#                columns = cursor.fetchall()
#                
#                # Get sample data (5 rows)
#                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
#                sample_data = cursor.fetchall()
#                
#                # Get row count
#                cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
#                row_count = cursor.fetchone()['count']
#                
#                # Get column statistics
#                column_stats = {}
#                for column in columns:
#                    col_name = column['Field']
#                    col_type = column['Type']
#                    
#                    # Get sample values for this column
#                    cursor.execute(f"SELECT DISTINCT {col_name} FROM {table_name} LIMIT 10")
#                    sample_values = [row[col_name] for row in cursor.fetchall()]
#                    
#                    column_stats[col_name] = {
#                        "type": col_type,
#                        "sample_values": sample_values[:5]  # Limit to 5
#                    }
#                
#                schema["tables"][table_name] = {
#                    "columns": columns,
#                    "row_count": row_count,
#                    "sample_data": sample_data,
#                    "column_stats": column_stats
#                }
#            
#            cursor.close()
#            return schema
#            
#        except Exception as e:
#            print(f"❌ Error getting schema: {e}")
#            return {}
#    
#    def execute_safe_query(self, query: str) -> Optional[pd.DataFrame]:
#        """
#        Execute SQL query with safety checks
#        - No DELETE, UPDATE, DROP, TRUNCATE, ALTER
#        - Only SELECT queries allowed
#        """
#        # Safety check: Only allow SELECT queries
#        query_upper = query.strip().upper()
#        dangerous_keywords = ['DELETE', 'UPDATE', 'DROP', 'TRUNCATE', 'ALTER', 'INSERT']
#        
#        for keyword in dangerous_keywords:
#            if keyword in query_upper and 'SELECT' not in query_upper:
#                raise ValueError(f"❌ Query contains dangerous operation: {keyword}")
#        
#        # Remove trailing semicolon if present
#        query = query.strip().rstrip(';')
#        
#        try:
#            if not self.connection:
#                self.connect()
#            
#            print(f"📊 Executing query: {query[:100]}...")
#            df = pd.read_sql(query, self.connection)
#            print(f"✅ Query returned {len(df)} rows")
#            return df
#            
#        except Exception as e:
#            print(f"❌ Query execution failed: {e}")
#            # Try to get error details
#            if "doesn't exist" in str(e):
#                # Get correct table names
#                schema = self.get_schema()
#                table_names = list(schema.get("tables", {}).keys())
#                raise ValueError(f"Table not found. Available tables: {', '.join(table_names)}")
#            raise e
#    
#    def get_table_info(self, table_name: str = None) -> str:
#        """
#        Get formatted table information for LLM context
#        """
#        schema = self.get_schema()
#        
#        if table_name and table_name in schema["tables"]:
#            table_info = schema["tables"][table_name]
#            info_str = f"Table: {table_name}\n"
#            info_str += f"Rows: {table_info['row_count']}\n"
#            info_str += "Columns:\n"
#            
#            for col in table_info["columns"]:
#                info_str += f"  - {col['Field']} ({col['Type']})\n"
#            
#            # Add sample values for important columns
#            info_str += "\nSample values for key columns:\n"
#            for col_name, stats in table_info["column_stats"].items():
#                if stats["sample_values"]:
#                    info_str += f"  - {col_name}: {stats['sample_values']}\n"
#            
#            return info_str
#        
#        # Return all tables info
#        info_str = f"Database: {schema['database']}\n\n"
#        for table_name, table_info in schema["tables"].items():
#            info_str += f"Table: {table_name} ({table_info['row_count']} rows)\n"
#            info_str += "Columns: " + ", ".join([col['Field'] for col in table_info["columns"]]) + "\n\n"
#        
#        return info_str
#    
#    def close(self):
#        """Close database connection"""
#        if self.connection:
#            self.connection.close()
#            print("🔌 Database connection closed")