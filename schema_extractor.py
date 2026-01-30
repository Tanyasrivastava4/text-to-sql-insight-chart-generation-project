"""
Auto-extracts schema from ANY MySQL database
Generates detailed descriptions automatically
"""
import mysql.connector
import pandas as pd
import json
import os
from typing import Dict, Any, List
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

class SchemaExtractor:
    def __init__(self):
        # Initialize Ollama LLM for schema description
        self.llm = ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            temperature=0.1
        )
        
        # Database connection
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = int(os.getenv("DB_PORT", 3307))
        self.user = os.getenv("DB_USER", "root")
        self.password = os.getenv("DB_PASSWORD", "")
        self.database = os.getenv("DB_NAME", "")
    
    def connect_db(self):
        """Connect to MySQL database"""
        try:
            return mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return None
    
    def extract_schema(self) -> Dict[str, Any]:
        """
        Extract COMPLETE schema with AI-generated descriptions
        """
        print("🔍 Extracting database schema...")
        
        conn = self.connect_db()
        if not conn:
            return {"error": "Cannot connect to database"}
        
        schema = {
            "database": self.database,
            "tables": {},
            "extracted_at": pd.Timestamp.now().isoformat()
        }
        
        try:
            cursor = conn.cursor(dictionary=True)
            
            # Get all tables
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = list(table.values())[0]
                print(f"📊 Analyzing table: {table_name}")
                
                # Get column information
                cursor.execute(f"DESCRIBE {table_name}")
                columns = cursor.fetchall()
                
                # Get sample data
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                sample_data = cursor.fetchall()
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) as cnt FROM {table_name}")
                row_count = cursor.fetchone()['cnt']
                
                # Analyze each column with AI
                column_details = {}
                for col in columns:
                    col_name = col['Field']
                    col_type = col['Type']
                    
                    # Get column statistics
                    stats = self._get_column_stats(cursor, table_name, col_name, col_type)
                    
                    # Generate AI description
                    description = self._generate_column_description(
                        table_name, col_name, col_type, stats
                    )
                    
                    column_details[col_name] = {
                        "type": col_type,
                        "nullable": col['Null'] == 'YES',
                        "key": col['Key'] or '',
                        "default": col['Default'],
                        "description": description,
                        "sample_values": stats.get('sample_values', []),
                        "is_primary": 'PRI' in col['Key']
                    }
                
                # Generate table description
                table_description = self._generate_table_description(
                    table_name, column_details, row_count
                )
                
                schema["tables"][table_name] = {
                    "columns": column_details,
                    "row_count": row_count,
                    "sample_data": sample_data,
                    "description": table_description
                }
            
            cursor.close()
            conn.close()
            
            # Save schema
            self._save_schema(schema)
            
            print(f"✅ Schema extracted! Found {len(schema['tables'])} tables")
            return schema
            
        except Exception as e:
            print(f"❌ Error extracting schema: {e}")
            return {"error": str(e)}
    
    def refresh_schema(self, force: bool = False) -> Dict[str, Any]:
        """
        Check if database changed and refresh schema if needed
        - force: True to always refresh
        - False: Check actual database structure vs cached
        """
        schema_file = "schema_info.json"
        
        if force:
            print("🔄 Forced schema refresh")
            return self.extract_schema()
        
        try:
            # Check if cached schema exists
            if not os.path.exists(schema_file):
                print("🔄 No cached schema found, extracting...")
                return self.extract_schema()
            
            # Load cached schema
            with open(schema_file, "r") as f:
                cached_schema = json.load(f)
            
            # Connect to database to check current state
            conn = self.connect_db()
            if not conn:
                print("❌ Cannot connect to database, using cached schema")
                return cached_schema
            
            #cursor = conn.cursor()
            cursor = conn.cursor(dictionary=True)  # ✅ CORRECT - matches extract_schema()
            
            # 1. Check if tables changed
            cursor.execute("SHOW TABLES")
            #current_tables = set([row[0] for row in cursor.fetchall()])
            current_tables = set([row[next(iter(row))] for row in cursor.fetchall()])
            
            # Get cached tables from schema
            cached_tables = set(cached_schema.get("tables", {}).keys())
            
            # Check for table differences
            tables_added = current_tables - cached_tables
            tables_removed = cached_tables - current_tables
            
            if tables_added or tables_removed:
                print(f"🔄 Tables changed. Added: {tables_added}, Removed: {tables_removed}")
                cursor.close()
                conn.close()
                return self.extract_schema()
            
            # 2. Check if columns changed in existing tables (INCLUDING TYPE CHANGES)
            schema_changed = False
            
            for table_name in current_tables:
                # Get current columns WITH TYPES from database
                cursor.execute(f"DESCRIBE {table_name}")
                current_columns_info = cursor.fetchall()
                current_columns = {}
                for col in current_columns_info:
                    current_columns[col['Field']] = {  # Changed from col[0]
                       "type": col['Type'],           # Changed from col[1]
                       "nullable": col['Null']        # Changed from col[2]
                    }
                #for col in current_columns_info:
                  #  current_columns[col[0]] = {
                   #     "type": col[1],      # Data type (INT, VARCHAR, etc.)
                   #     "nullable": col[2]   # Nullable status (YES/NO)
                   # }
                
                # Get cached columns WITH TYPES from schema
                cached_table_info = cached_schema.get("tables", {}).get(table_name, {})
                cached_columns = {}
                for col_name, col_info in cached_table_info.get("columns", {}).items():
                    cached_columns[col_name] = {
                        "type": col_info.get("type", ""),
                        "nullable": "YES" if col_info.get("nullable") else "NO"
                    }
                
                # Check for column name differences
                columns_added = set(current_columns.keys()) - set(cached_columns.keys())
                columns_removed = set(cached_columns.keys()) - set(current_columns.keys())
                
                # Check for TYPE and NULLABLE changes in existing columns
                type_changed = False
                common_columns = set(current_columns.keys()) & set(cached_columns.keys())
                #for col_name in common_columns:
                #    # Check data type change (INT → DECIMAL, VARCHAR → TEXT, etc.)
                #    if current_columns[col_name]["type"] != cached_columns[col_name]["type"]:
                #        print(f"🔄 Column '{col_name}' data type changed: "
                #              f"{cached_columns[col_name]['type']} → {current_columns[col_name]['type']}")
                #        type_changed = True
                #        break
                #    
                #    # Check nullable status change (YES → NO or NO → YES)
                #    if current_columns[col_name]["nullable"] != cached_columns[col_name]["nullable"]:
                #        print(f"🔄 Column '{col_name}' nullable status changed: "
                #              f"{cached_columns[col_name]['nullable']} → {current_columns[col_name]['nullable']}")
                #        type_changed = True
                #        break

                for col_name in common_columns:
                # Check data type change (INT → DECIMAL, VARCHAR → TEXT, etc.)
                # Convert both to strings for comparison (handles bytes vs string issue)
                    current_type = str(current_columns[col_name]["type"])
                    cached_type = str(cached_columns[col_name]["type"])

                    if current_type != cached_type:
                        print(f"🔄 Column '{col_name}' data type changed: "
                              f"{cached_type} → {current_type}")
                        type_changed = True
                        break
    
                # Check nullable status change (YES → NO or NO → YES)
                # Convert both to strings for comparison
                    current_nullable = str(current_columns[col_name]["nullable"])
                    cached_nullable = str(cached_columns[col_name]["nullable"])

                    if current_nullable != cached_nullable:
                        print(f"🔄 Column '{col_name}' nullable status changed: "
                        f"{cached_nullable} → {current_nullable}")
                        type_changed = True
                        break
                
                if columns_added or columns_removed or type_changed:
                    print(f"🔄 Schema changed in table '{table_name}'")
                    schema_changed = True
                    break
            
            cursor.close()
            conn.close()
            
            if schema_changed:
                return self.extract_schema()
            else:
                print("✅ Database unchanged, using cached schema")
                return cached_schema
                
        except Exception as e:
            print(f"❌ Error checking database changes: {e}")
            # Fallback to cached schema
            try:
                with open(schema_file, "r") as f:
                    return json.load(f)
            except:
                return self.extract_schema()
    
    def _get_column_stats(self, cursor, table: str, column: str, col_type: str) -> Dict:
        """Get statistics and sample values for a column"""
        stats = {}
        
        try:
            # Check if column is numeric
            if any(t in col_type.lower() for t in ['int', 'decimal', 'float', 'double']):
                cursor.execute(f"""
                    SELECT 
                        MIN({column}) as min_val,
                        MAX({column}) as max_val,
                        AVG({column}) as avg_val,
                        COUNT(DISTINCT {column}) as distinct_count
                    FROM {table}
                    WHERE {column} IS NOT NULL
                """)
                stats.update(cursor.fetchone())
            
            # Get sample values
            cursor.execute(f"""
                SELECT DISTINCT {column} 
                FROM {table} 
                WHERE {column} IS NOT NULL 
                LIMIT 5
            """)
            stats['sample_values'] = [row[column] for row in cursor.fetchall()]
            
        except:
            stats['sample_values'] = []
        
        return stats
    
    def _generate_column_description(self, table: str, column: str, col_type: str, stats: Dict) -> str:
        """Generate column description using LLM"""
        prompt = f"""
        Based on this database context, generate a one-line description for this column:
        
        Table: {table}
        Column: {column}
        Data Type: {col_type}
        Sample Values: {stats.get('sample_values', [])[:3]}
        Statistics: {stats}
        
        Description format: "[Column Name] : [Data Type] : [Brief description of what this column contains and how to use it in analysis]"
        
        Example: "Order_Date : DATE : Date when order was placed - Use for time-based analysis"
        
        Generate ONLY the description line, nothing else:
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except:
            # Fallback description
            return f"{column} : {col_type} : Column in {table} table"
    
    def _generate_table_description(self, table: str, columns: Dict, row_count: int) -> str:
        """Generate table description using LLM"""
        column_list = "\n".join([f"- {name}: {info['type']}" for name, info in columns.items()])
        
        prompt = f"""
        Generate a brief description for this database table:
        
        Table Name: {table}
        Row Count: {row_count:,}
        Columns:
        {column_list}
        
        Describe what this table contains and its purpose in one paragraph:
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except:
            return f"Table {table} with {row_count} rows and {len(columns)} columns"
    
    def _save_schema(self, schema: Dict):
        """Save schema to JSON file"""
        try:
            with open("schema_info.json", "w") as f:
                json.dump(schema, f, indent=2, default=str)
            print("💾 Schema saved to: schema_info.json")
        except Exception as e:
            print(f"❌ Error saving schema: {e}")
    
    def load_schema(self) -> Dict:
        """Load schema from file or extract fresh"""
        try:
            with open("schema_info.json", "r") as f:
                return json.load(f)
        except:
            return self.extract_schema()
    
    def get_formatted_schema(self) -> str:
        """Get schema in formatted text for LLM context"""
        schema = self.load_schema()
        
        if "error" in schema:
            return "Error: Could not load schema"
        
        formatted = f"Database: {schema['database']}\n\n"
        
        for table_name, table_info in schema["tables"].items():
            formatted += f"TABLE: {table_name}\n"
            formatted += f"Description: {table_info['description']}\n"
            formatted += f"Rows: {table_info['row_count']:,}\n\n"
            
            formatted += "COLUMNS:\n"
            for col_name, col_info in table_info["columns"].items():
                formatted += f"  • {col_info['description']}\n"
                
                if col_info.get('sample_values'):
                    samples = col_info['sample_values'][:3]
                    formatted += f"    Sample values: {samples}\n"
                
                if col_info['is_primary']:
                    formatted += f"    ⚠️ Primary Key - Do not use in analysis\n"
            
            formatted += "\n" + "="*60 + "\n\n"
        
        return formatted

# Test
if __name__ == "__main__":
    print("Testing schema extractor...")
    extractor = SchemaExtractor()
    schema = extractor.extract_schema()
    
    if "error" not in schema:
        print(f"\n✅ Schema Summary:")
        print(f"Database: {schema['database']}")
        print(f"Tables: {len(schema['tables'])}")
        
        # Show first table details
        for table_name in list(schema['tables'].keys())[:1]:
            table = schema['tables'][table_name]
            print(f"\n📊 Table: {table_name}")
            print(f"Rows: {table['row_count']:,}")
            print(f"Description: {table['description'][:100]}...")
            
            # Show first 3 columns
            cols = list(table['columns'].items())[:3]
            for col_name, col_info in cols:
                print(f"\n  Column: {col_name}")
                print(f"  Desc: {col_info['description']}")
    else:
        print(f"❌ Error: {schema['error']}")






##worked just changed for adding datatype change and database change detected .
#"""
#Auto-extracts schema from ANY MySQL database
#Generates detailed descriptions automatically
#"""
#import mysql.connector
#import pandas as pd
#import json
#import os
#from typing import Dict, Any, List
#from langchain_community.chat_models import ChatOllama
#from langchain_core.messages import HumanMessage
#from dotenv import load_dotenv
#
#load_dotenv()
#
#class SchemaExtractor:
#    def __init__(self):
#        # Initialize Ollama LLM for schema description
#        self.llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
#            temperature=0.1
#        )
#        
#        # Database connection
#        self.host = os.getenv("DB_HOST", "localhost")
#        self.port = int(os.getenv("DB_PORT", 3307))
#        self.user = os.getenv("DB_USER", "root")
#        self.password = os.getenv("DB_PASSWORD", "")
#        self.database = os.getenv("DB_NAME", "")
#    
#    def connect_db(self):
#        """Connect to MySQL database"""
#        try:
#            return mysql.connector.connect(
#                host=self.host,
#                port=self.port,
#                user=self.user,
#                password=self.password,
#                database=self.database
#            )
#        except Exception as e:
#            print(f"❌ Database connection failed: {e}")
#            return None
#    
#    def extract_schema(self) -> Dict[str, Any]:
#        """
#        Extract COMPLETE schema with AI-generated descriptions
#        """
#        print("🔍 Extracting database schema...")
#        
#        conn = self.connect_db()
#        if not conn:
#            return {"error": "Cannot connect to database"}
#        
#        schema = {
#            "database": self.database,
#            "tables": {},
#            "extracted_at": pd.Timestamp.now().isoformat()
#        }
#        
#        try:
#            cursor = conn.cursor(dictionary=True)
#            
#            # Get all tables
#            cursor.execute("SHOW TABLES")
#            tables = cursor.fetchall()
#            
#            for table in tables:
#                table_name = list(table.values())[0]
#                print(f"📊 Analyzing table: {table_name}")
#                
#                # Get column information
#                cursor.execute(f"DESCRIBE {table_name}")
#                columns = cursor.fetchall()
#                
#                # Get sample data
#                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
#                sample_data = cursor.fetchall()
#                
#                # Get row count
#                cursor.execute(f"SELECT COUNT(*) as cnt FROM {table_name}")
#                row_count = cursor.fetchone()['cnt']
#                
#                # Analyze each column with AI
#                column_details = {}
#                for col in columns:
#                    col_name = col['Field']
#                    col_type = col['Type']
#                    
#                    # Get column statistics
#                    stats = self._get_column_stats(cursor, table_name, col_name, col_type)
#                    
#                    # Generate AI description
#                    description = self._generate_column_description(
#                        table_name, col_name, col_type, stats
#                    )
#                    
#                    column_details[col_name] = {
#                        "type": col_type,
#                        "nullable": col['Null'] == 'YES',
#                        "key": col['Key'] or '',
#                        "default": col['Default'],
#                        "description": description,
#                        "sample_values": stats.get('sample_values', []),
#                        "is_primary": 'PRI' in col['Key']
#                    }
#                
#                # Generate table description
#                table_description = self._generate_table_description(
#                    table_name, column_details, row_count
#                )
#                
#                schema["tables"][table_name] = {
#                    "columns": column_details,
#                    "row_count": row_count,
#                    "sample_data": sample_data,
#                    "description": table_description
#                }
#            
#            cursor.close()
#            conn.close()
#            
#            # Save schema
#            self._save_schema(schema)
#            
#            print(f"✅ Schema extracted! Found {len(schema['tables'])} tables")
#            return schema
#            
#        except Exception as e:
#            print(f"❌ Error extracting schema: {e}")
#            return {"error": str(e)}
#    
#    def refresh_schema(self, force: bool = False) -> Dict[str, Any]:
#        """
#        Check if database changed and refresh schema if needed
#        - force: True to always refresh
#        - False: Check actual database structure vs cached
#        """
#        schema_file = "schema_info.json"
#        
#        if force:
#            print("🔄 Forced schema refresh")
#            return self.extract_schema()
#        
#        try:
#            # Check if cached schema exists
#            if not os.path.exists(schema_file):
#                print("🔄 No cached schema found, extracting...")
#                return self.extract_schema()
#            
#            # Load cached schema
#            with open(schema_file, "r") as f:
#                cached_schema = json.load(f)
#            
#            # Connect to database to check current state
#            conn = self.connect_db()
#            if not conn:
#                print("❌ Cannot connect to database, using cached schema")
#                return cached_schema
#            
#            cursor = conn.cursor()
#            
#            # 1. Check if tables changed
#            cursor.execute("SHOW TABLES")
#            current_tables = set([row[0] for row in cursor.fetchall()])
#            
#            # Get cached tables from schema
#            cached_tables = set(cached_schema.get("tables", {}).keys())
#            
#            # Check for table differences
#            tables_added = current_tables - cached_tables
#            tables_removed = cached_tables - current_tables
#            
#            if tables_added or tables_removed:
#                print(f"🔄 Tables changed. Added: {tables_added}, Removed: {tables_removed}")
#                cursor.close()
#                conn.close()
#                return self.extract_schema()
#            
#            # 2. Check if columns changed in existing tables
#            schema_changed = False
#            
#            for table_name in current_tables:
#                # Get current columns from database
#                cursor.execute(f"DESCRIBE {table_name}")
#                current_columns = set([row[0] for row in cursor.fetchall()])
#                
#                # Get cached columns from schema
#                cached_table_info = cached_schema.get("tables", {}).get(table_name, {})
#                cached_columns = set(cached_table_info.get("columns", {}).keys())
#                
#                # Check for column differences
#                columns_added = current_columns - cached_columns
#                columns_removed = cached_columns - current_columns
#                
#                if columns_added or columns_removed:
#                    print(f"🔄 Columns changed in table '{table_name}'. "
#                          f"Added: {columns_added}, Removed: {columns_removed}")
#                    schema_changed = True
#                    break
#            
#            cursor.close()
#            conn.close()
#            
#            if schema_changed:
#                return self.extract_schema()
#            else:
#                print("✅ Database unchanged, using cached schema")
#                return cached_schema
#                
#        except Exception as e:
#            print(f"❌ Error checking database changes: {e}")
#            # Fallback to cached schema
#            try:
#                with open(schema_file, "r") as f:
#                    return json.load(f)
#            except:
#                return self.extract_schema()
#    
#    def _get_column_stats(self, cursor, table: str, column: str, col_type: str) -> Dict:
#        """Get statistics and sample values for a column"""
#        stats = {}
#        
#        try:
#            # Check if column is numeric
#            if any(t in col_type.lower() for t in ['int', 'decimal', 'float', 'double']):
#                cursor.execute(f"""
#                    SELECT 
#                        MIN({column}) as min_val,
#                        MAX({column}) as max_val,
#                        AVG({column}) as avg_val,
#                        COUNT(DISTINCT {column}) as distinct_count
#                    FROM {table}
#                    WHERE {column} IS NOT NULL
#                """)
#                stats.update(cursor.fetchone())
#            
#            # Get sample values
#            cursor.execute(f"""
#                SELECT DISTINCT {column} 
#                FROM {table} 
#                WHERE {column} IS NOT NULL 
#                LIMIT 5
#            """)
#            stats['sample_values'] = [row[column] for row in cursor.fetchall()]
#            
#        except:
#            stats['sample_values'] = []
#        
#        return stats
#    
#    def _generate_column_description(self, table: str, column: str, col_type: str, stats: Dict) -> str:
#        """Generate column description using LLM"""
#        prompt = f"""
#        Based on this database context, generate a one-line description for this column:
#        
#        Table: {table}
#        Column: {column}
#        Data Type: {col_type}
#        Sample Values: {stats.get('sample_values', [])[:3]}
#        Statistics: {stats}
#        
#        Description format: "[Column Name] : [Data Type] : [Brief description of what this column contains and how to use it in analysis]"
#        
#        Example: "Order_Date : DATE : Date when order was placed - Use for time-based analysis"
#        
#        Generate ONLY the description line, nothing else:
#        """
#        
#        try:
#            response = self.llm.invoke([HumanMessage(content=prompt)])
#            return response.content.strip()
#        except:
#            # Fallback description
#            return f"{column} : {col_type} : Column in {table} table"
#    
#    def _generate_table_description(self, table: str, columns: Dict, row_count: int) -> str:
#        """Generate table description using LLM"""
#        column_list = "\n".join([f"- {name}: {info['type']}" for name, info in columns.items()])
#        
#        prompt = f"""
#        Generate a brief description for this database table:
#        
#        Table Name: {table}
#        Row Count: {row_count:,}
#        Columns:
#        {column_list}
#        
#        Describe what this table contains and its purpose in one paragraph:
#        """
#        
#        try:
#            response = self.llm.invoke([HumanMessage(content=prompt)])
#            return response.content.strip()
#        except:
#            return f"Table {table} with {row_count} rows and {len(columns)} columns"
#    
#    def _save_schema(self, schema: Dict):
#        """Save schema to JSON file"""
#        try:
#            with open("schema_info.json", "w") as f:
#                json.dump(schema, f, indent=2, default=str)
#            print("💾 Schema saved to: schema_info.json")
#        except Exception as e:
#            print(f"❌ Error saving schema: {e}")
#    
#    def load_schema(self) -> Dict:
#        """Load schema from file or extract fresh"""
#        try:
#            with open("schema_info.json", "r") as f:
#                return json.load(f)
#        except:
#            return self.extract_schema()
#    
#    def get_formatted_schema(self) -> str:
#        """Get schema in formatted text for LLM context"""
#        schema = self.load_schema()
#        
#        if "error" in schema:
#            return "Error: Could not load schema"
#        
#        formatted = f"Database: {schema['database']}\n\n"
#        
#        for table_name, table_info in schema["tables"].items():
#            formatted += f"TABLE: {table_name}\n"
#            formatted += f"Description: {table_info['description']}\n"
#            formatted += f"Rows: {table_info['row_count']:,}\n\n"
#            
#            formatted += "COLUMNS:\n"
#            for col_name, col_info in table_info["columns"].items():
#                formatted += f"  • {col_info['description']}\n"
#                
#                if col_info.get('sample_values'):
#                    samples = col_info['sample_values'][:3]
#                    formatted += f"    Sample values: {samples}\n"
#                
#                if col_info['is_primary']:
#                    formatted += f"    ⚠️ Primary Key - Do not use in analysis\n"
#            
#            formatted += "\n" + "="*60 + "\n\n"
#        
#        return formatted
#
## Test
#if __name__ == "__main__":
#    print("Testing schema extractor...")
#    extractor = SchemaExtractor()
#    schema = extractor.extract_schema()
#    
#    if "error" not in schema:
#        print(f"\n✅ Schema Summary:")
#        print(f"Database: {schema['database']}")
#        print(f"Tables: {len(schema['tables'])}")
#        
#        # Show first table details
#        for table_name in list(schema['tables'].keys())[:1]:
#            table = schema['tables'][table_name]
#            print(f"\n📊 Table: {table_name}")
#            print(f"Rows: {table['row_count']:,}")
#            print(f"Description: {table['description'][:100]}...")
#            
#            # Show first 3 columns
#            cols = list(table['columns'].items())[:3]
#            for col_name, col_info in cols:
#                print(f"\n  Column: {col_name}")
#                print(f"  Desc: {col_info['description']}")
#    else:
#        print(f"❌ Error: {schema['error']}")






#This also worked updated - refresh when any changes in db
#"""
#Auto-extracts schema from ANY MySQL database
#Generates detailed descriptions automatically
#"""
#import mysql.connector
#import pandas as pd
#import json
#import os
#import time  # Added for refresh_schema method
#from typing import Dict, Any, List
#from langchain_community.chat_models import ChatOllama
#from langchain_core.messages import HumanMessage
#from dotenv import load_dotenv
#
#load_dotenv()
#
#class SchemaExtractor:
#    def __init__(self):
#        # Initialize Ollama LLM for schema description
#        self.llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
#            temperature=0.1
#        )
#        
#        # Database connection
#        self.host = os.getenv("DB_HOST", "localhost")
#        self.port = int(os.getenv("DB_PORT", 3307))
#        self.user = os.getenv("DB_USER", "root")
#        self.password = os.getenv("DB_PASSWORD", "")
#        self.database = os.getenv("DB_NAME", "")
#    
#    def connect_db(self):
#        """Connect to MySQL database"""
#        try:
#            return mysql.connector.connect(
#                host=self.host,
#                port=self.port,
#                user=self.user,
#                password=self.password,
#                database=self.database
#            )
#        except Exception as e:
#            print(f"❌ Database connection failed: {e}")
#            return None
#    
#    def extract_schema(self) -> Dict[str, Any]:
#        """
#        Extract COMPLETE schema with AI-generated descriptions
#        """
#        print("🔍 Extracting database schema...")
#        
#        conn = self.connect_db()
#        if not conn:
#            return {"error": "Cannot connect to database"}
#        
#        schema = {
#            "database": self.database,
#            "tables": {},
#            "extracted_at": pd.Timestamp.now().isoformat()
#        }
#        
#        try:
#            cursor = conn.cursor(dictionary=True)
#            
#            # Get all tables
#            cursor.execute("SHOW TABLES")
#            tables = cursor.fetchall()
#            
#            for table in tables:
#                table_name = list(table.values())[0]
#                print(f"📊 Analyzing table: {table_name}")
#                
#                # Get column information
#                cursor.execute(f"DESCRIBE {table_name}")
#                columns = cursor.fetchall()
#                
#                # Get sample data
#                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
#                sample_data = cursor.fetchall()
#                
#                # Get row count
#                cursor.execute(f"SELECT COUNT(*) as cnt FROM {table_name}")
#                row_count = cursor.fetchone()['cnt']
#                
#                # Analyze each column with AI
#                column_details = {}
#                for col in columns:
#                    col_name = col['Field']
#                    col_type = col['Type']
#                    
#                    # Get column statistics
#                    stats = self._get_column_stats(cursor, table_name, col_name, col_type)
#                    
#                    # Generate AI description
#                    description = self._generate_column_description(
#                        table_name, col_name, col_type, stats
#                    )
#                    
#                    column_details[col_name] = {
#                        "type": col_type,
#                        "nullable": col['Null'] == 'YES',
#                        "key": col['Key'] or '',
#                        "default": col['Default'],
#                        "description": description,
#                        "sample_values": stats.get('sample_values', []),
#                        "is_primary": 'PRI' in col['Key']
#                    }
#                
#                # Generate table description
#                table_description = self._generate_table_description(
#                    table_name, column_details, row_count
#                )
#                
#                schema["tables"][table_name] = {
#                    "columns": column_details,
#                    "row_count": row_count,
#                    "sample_data": sample_data,
#                    "description": table_description
#                }
#            
#            cursor.close()
#            conn.close()
#            
#            # Save schema
#            self._save_schema(schema)
#            
#            print(f"✅ Schema extracted! Found {len(schema['tables'])} tables")
#            return schema
#            
#        except Exception as e:
#            print(f"❌ Error extracting schema: {e}")
#            return {"error": str(e)}
#    
#    def refresh_schema(self, force: bool = False) -> Dict[str, Any]:
#        """
#        Refresh schema if needed
#        - force: True to always refresh, False to refresh only if schema file is old
#        """
#        schema_file = "schema_info.json"
#        
#        if force:
#            print("🔄 Forced schema refresh requested")
#            return self.extract_schema()
#        
#        # Check if schema file exists and is recent (less than 1 hour old)
#        try:
#            if os.path.exists(schema_file):
#                file_mtime = os.path.getmtime(schema_file)
#                current_time = time.time()
#                
#                # If schema is older than 1 hour, refresh it
#                if (current_time - file_mtime) > 3600:  # 1 hour in seconds
#                    print("🔄 Schema is old (>1 hour), refreshing...")
#                    return self.extract_schema()
#                else:
#                    print("✅ Using cached schema (less than 1 hour old)")
#                    return self.load_schema()
#            else:
#                print("🔄 No schema file found, extracting fresh...")
#                return self.extract_schema()
#                
#        except Exception as e:
#            print(f"❌ Error checking schema age: {e}")
#            return self.extract_schema()
#    
#    def _get_column_stats(self, cursor, table: str, column: str, col_type: str) -> Dict:
#        """Get statistics and sample values for a column"""
#        stats = {}
#        
#        try:
#            # Check if column is numeric
#            if any(t in col_type.lower() for t in ['int', 'decimal', 'float', 'double']):
#                cursor.execute(f"""
#                    SELECT 
#                        MIN({column}) as min_val,
#                        MAX({column}) as max_val,
#                        AVG({column}) as avg_val,
#                        COUNT(DISTINCT {column}) as distinct_count
#                    FROM {table}
#                    WHERE {column} IS NOT NULL
#                """)
#                stats.update(cursor.fetchone())
#            
#            # Get sample values
#            cursor.execute(f"""
#                SELECT DISTINCT {column} 
#                FROM {table} 
#                WHERE {column} IS NOT NULL 
#                LIMIT 5
#            """)
#            stats['sample_values'] = [row[column] for row in cursor.fetchall()]
#            
#        except:
#            stats['sample_values'] = []
#        
#        return stats
#    
#    def _generate_column_description(self, table: str, column: str, col_type: str, stats: Dict) -> str:
#        """Generate column description using LLM"""
#        prompt = f"""
#        Based on this database context, generate a one-line description for this column:
#        
#        Table: {table}
#        Column: {column}
#        Data Type: {col_type}
#        Sample Values: {stats.get('sample_values', [])[:3]}
#        Statistics: {stats}
#        
#        Description format: "[Column Name] : [Data Type] : [Brief description of what this column contains and how to use it in analysis]"
#        
#        Example: "Order_Date : DATE : Date when order was placed - Use for time-based analysis"
#        
#        Generate ONLY the description line, nothing else:
#        """
#        
#        try:
#            response = self.llm.invoke([HumanMessage(content=prompt)])
#            return response.content.strip()
#        except:
#            # Fallback description
#            return f"{column} : {col_type} : Column in {table} table"
#    
#    def _generate_table_description(self, table: str, columns: Dict, row_count: int) -> str:
#        """Generate table description using LLM"""
#        column_list = "\n".join([f"- {name}: {info['type']}" for name, info in columns.items()])
#        
#        prompt = f"""
#        Generate a brief description for this database table:
#        
#        Table Name: {table}
#        Row Count: {row_count:,}
#        Columns:
#        {column_list}
#        
#        Describe what this table contains and its purpose in one paragraph:
#        """
#        
#        try:
#            response = self.llm.invoke([HumanMessage(content=prompt)])
#            return response.content.strip()
#        except:
#            return f"Table {table} with {row_count} rows and {len(columns)} columns"
#    
#    def _save_schema(self, schema: Dict):
#        """Save schema to JSON file"""
#        try:
#            with open("schema_info.json", "w") as f:
#                json.dump(schema, f, indent=2, default=str)
#            print("💾 Schema saved to: schema_info.json")
#        except Exception as e:
#            print(f"❌ Error saving schema: {e}")
#    
#    def load_schema(self) -> Dict:
#        """Load schema from file or extract fresh"""
#        try:
#            with open("schema_info.json", "r") as f:
#                return json.load(f)
#        except:
#            return self.extract_schema()
#    
#    def get_formatted_schema(self) -> str:
#        """Get schema in formatted text for LLM context"""
#        schema = self.load_schema()
#        
#        if "error" in schema:
#            return "Error: Could not load schema"
#        
#        formatted = f"Database: {schema['database']}\n\n"
#        
#        for table_name, table_info in schema["tables"].items():
#            formatted += f"TABLE: {table_name}\n"
#            formatted += f"Description: {table_info['description']}\n"
#            formatted += f"Rows: {table_info['row_count']:,}\n\n"
#            
#            formatted += "COLUMNS:\n"
#            for col_name, col_info in table_info["columns"].items():
#                formatted += f"  • {col_info['description']}\n"
#                
#                if col_info.get('sample_values'):
#                    samples = col_info['sample_values'][:3]
#                    formatted += f"    Sample values: {samples}\n"
#                
#                if col_info['is_primary']:
#                    formatted += f"    ⚠️ Primary Key - Do not use in analysis\n"
#            
#            formatted += "\n" + "="*60 + "\n\n"
#        
#        return formatted
#
## Test
#if __name__ == "__main__":
#    print("Testing schema extractor...")
#    extractor = SchemaExtractor()
#    schema = extractor.extract_schema()
#    
#    if "error" not in schema:
#        print(f"\n✅ Schema Summary:")
#        print(f"Database: {schema['database']}")
#        print(f"Tables: {len(schema['tables'])}")
#        
#        # Show first table details
#        for table_name in list(schema['tables'].keys())[:1]:
#            table = schema['tables'][table_name]
#            print(f"\n📊 Table: {table_name}")
#            print(f"Rows: {table['row_count']:,}")
#            print(f"Description: {table['description'][:100]}...")
#            
#            # Show first 3 columns
#            cols = list(table['columns'].items())[:3]
#            for col_name, col_info in cols:
#                print(f"\n  Column: {col_name}")
#                print(f"  Desc: {col_info['description']}")
#    else:
#        print(f"❌ Error: {schema['error']}")
#






# worked this just updated that above only for - what if someone added new col or make changes in database-then we will get the update schema.

#"""
#Auto-extracts schema from ANY MySQL database
#Generates detailed descriptions automatically
#"""
#import mysql.connector
#import pandas as pd
#import json
#import os
#from typing import Dict, Any, List
#from langchain_community.chat_models import ChatOllama
##from langchain.schema import HumanMessage
##from langchain.schema.messages import HumanMessage
#from langchain_core.messages import HumanMessage
#from dotenv import load_dotenv
#
#load_dotenv()
#
#class SchemaExtractor:
#    def __init__(self):
#        # Initialize Ollama LLM for schema description
#        self.llm = ChatOllama(
#            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
#            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
#            temperature=0.1
#        )
#        
#        # Database connection
#        self.host = os.getenv("DB_HOST", "localhost")
#        self.port = int(os.getenv("DB_PORT", 3307))
#        self.user = os.getenv("DB_USER", "root")
#        self.password = os.getenv("DB_PASSWORD", "")
#        self.database = os.getenv("DB_NAME", "")
#    
#    def connect_db(self):
#        """Connect to MySQL database"""
#        try:
#            return mysql.connector.connect(
#                host=self.host,
#                port=self.port,
#                user=self.user,
#                password=self.password,
#                database=self.database
#            )
#        except Exception as e:
#            print(f"❌ Database connection failed: {e}")
#            return None
#    
#    def extract_schema(self) -> Dict[str, Any]:
#        """
#        Extract COMPLETE schema with AI-generated descriptions
#        """
#        print("🔍 Extracting database schema...")
#        
#        conn = self.connect_db()
#        if not conn:
#            return {"error": "Cannot connect to database"}
#        
#        schema = {
#            "database": self.database,
#            "tables": {},
#            "extracted_at": pd.Timestamp.now().isoformat()
#        }
#        
#        try:
#            cursor = conn.cursor(dictionary=True)
#            
#            # Get all tables
#            cursor.execute("SHOW TABLES")
#            tables = cursor.fetchall()
#            
#            for table in tables:
#                table_name = list(table.values())[0]
#                print(f"📊 Analyzing table: {table_name}")
#                
#                # Get column information
#                cursor.execute(f"DESCRIBE {table_name}")
#                columns = cursor.fetchall()
#                
#                # Get sample data
#                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
#                sample_data = cursor.fetchall()
#                
#                # Get row count
#                cursor.execute(f"SELECT COUNT(*) as cnt FROM {table_name}")
#                row_count = cursor.fetchone()['cnt']
#                
#                # Analyze each column with AI
#                column_details = {}
#                for col in columns:
#                    col_name = col['Field']
#                    col_type = col['Type']
#                    
#                    # Get column statistics
#                    stats = self._get_column_stats(cursor, table_name, col_name, col_type)
#                    
#                    # Generate AI description
#                    description = self._generate_column_description(
#                        table_name, col_name, col_type, stats
#                    )
#                    
#                    column_details[col_name] = {
#                        "type": col_type,
#                        "nullable": col['Null'] == 'YES',
#                        "key": col['Key'] or '',
#                        "default": col['Default'],
#                        "description": description,
#                        "sample_values": stats.get('sample_values', []),
#                        "is_primary": 'PRI' in col['Key']
#                    }
#                
#                # Generate table description
#                table_description = self._generate_table_description(
#                    table_name, column_details, row_count
#                )
#                
#                schema["tables"][table_name] = {
#                    "columns": column_details,
#                    "row_count": row_count,
#                    "sample_data": sample_data,
#                    "description": table_description
#                }
#            
#            cursor.close()
#            conn.close()
#            
#            # Save schema
#            self._save_schema(schema)
#            
#            print(f"✅ Schema extracted! Found {len(schema['tables'])} tables")
#            return schema
#            
#        except Exception as e:
#            print(f"❌ Error extracting schema: {e}")
#            return {"error": str(e)}
#    
#    def _get_column_stats(self, cursor, table: str, column: str, col_type: str) -> Dict:
#        """Get statistics and sample values for a column"""
#        stats = {}
#        
#        try:
#            # Check if column is numeric
#            if any(t in col_type.lower() for t in ['int', 'decimal', 'float', 'double']):
#                cursor.execute(f"""
#                    SELECT 
#                        MIN({column}) as min_val,
#                        MAX({column}) as max_val,
#                        AVG({column}) as avg_val,
#                        COUNT(DISTINCT {column}) as distinct_count
#                    FROM {table}
#                    WHERE {column} IS NOT NULL
#                """)
#                stats.update(cursor.fetchone())
#            
#            # Get sample values
#            cursor.execute(f"""
#                SELECT DISTINCT {column} 
#                FROM {table} 
#                WHERE {column} IS NOT NULL 
#                LIMIT 5
#            """)
#            stats['sample_values'] = [row[column] for row in cursor.fetchall()]
#            
#        except:
#            stats['sample_values'] = []
#        
#        return stats
#    
#    def _generate_column_description(self, table: str, column: str, col_type: str, stats: Dict) -> str:
#        """Generate column description using LLM"""
#        prompt = f"""
#        Based on this database context, generate a one-line description for this column:
#        
#        Table: {table}
#        Column: {column}
#        Data Type: {col_type}
#        Sample Values: {stats.get('sample_values', [])[:3]}
#        Statistics: {stats}
#        
#        Description format: "[Column Name] : [Data Type] : [Brief description of what this column contains and how to use it in analysis]"
#        
#        Example: "Order_Date : DATE : Date when order was placed - Use for time-based analysis"
#        
#        Generate ONLY the description line, nothing else:
#        """
#        
#        try:
#            response = self.llm.invoke([HumanMessage(content=prompt)])
#            return response.content.strip()
#        except:
#            # Fallback description
#            return f"{column} : {col_type} : Column in {table} table"
#    
#    def _generate_table_description(self, table: str, columns: Dict, row_count: int) -> str:
#        """Generate table description using LLM"""
#        column_list = "\n".join([f"- {name}: {info['type']}" for name, info in columns.items()])
#        
#        prompt = f"""
#        Generate a brief description for this database table:
#        
#        Table Name: {table}
#        Row Count: {row_count:,}
#        Columns:
#        {column_list}
#        
#        Describe what this table contains and its purpose in one paragraph:
#        """
#        
#        try:
#            response = self.llm.invoke([HumanMessage(content=prompt)])
#            return response.content.strip()
#        except:
#            return f"Table {table} with {row_count} rows and {len(columns)} columns"
#    
#    def _save_schema(self, schema: Dict):
#        """Save schema to JSON file"""
#        try:
#            with open("schema_info.json", "w") as f:
#                json.dump(schema, f, indent=2, default=str)
#            print("💾 Schema saved to: schema_info.json")
#        except Exception as e:
#            print(f"❌ Error saving schema: {e}")
#    
#    def load_schema(self) -> Dict:
#        """Load schema from file or extract fresh"""
#        try:
#            with open("schema_info.json", "r") as f:
#                return json.load(f)
#        except:
#            return self.extract_schema()
#    
#    def get_formatted_schema(self) -> str:
#        """Get schema in formatted text for LLM context"""
#        schema = self.load_schema()
#        
#        if "error" in schema:
#            return "Error: Could not load schema"
#        
#        formatted = f"Database: {schema['database']}\n\n"
#        
#        for table_name, table_info in schema["tables"].items():
#            formatted += f"TABLE: {table_name}\n"
#            formatted += f"Description: {table_info['description']}\n"
#            formatted += f"Rows: {table_info['row_count']:,}\n\n"
#            
#            formatted += "COLUMNS:\n"
#            for col_name, col_info in table_info["columns"].items():
#                formatted += f"  • {col_info['description']}\n"
#                
#                if col_info.get('sample_values'):
#                    samples = col_info['sample_values'][:3]
#                    formatted += f"    Sample values: {samples}\n"
#                
#                if col_info['is_primary']:
#                    formatted += f"    ⚠️ Primary Key - Do not use in analysis\n"
#            
#            formatted += "\n" + "="*60 + "\n\n"
#        
#        return formatted
#
## Test
#if __name__ == "__main__":
#    print("Testing schema extractor...")
#    extractor = SchemaExtractor()
#    schema = extractor.extract_schema()
#    
#    if "error" not in schema:
#        print(f"\n✅ Schema Summary:")
#        print(f"Database: {schema['database']}")
#        print(f"Tables: {len(schema['tables'])}")
#        
#        # Show first table details
#        for table_name in list(schema['tables'].keys())[:1]:
#            table = schema['tables'][table_name]
#            print(f"\n📊 Table: {table_name}")
#            print(f"Rows: {table['row_count']:,}")
#            print(f"Description: {table['description'][:100]}...")
#            
#            # Show first 3 columns
#            cols = list(table['columns'].items())[:3]
#            for col_name, col_info in cols:
#                print(f"\n  Column: {col_name}")
#                print(f"  Desc: {col_info['description']}")
#    else:
#        print(f"❌ Error: {schema['error']}")
#






#"""
#AUTOMATICALLY extract schema from MySQL database
#"""
#import json
#from database import Database
#import datetime
#
#class SchemaExtractor:
#    def __init__(self):
#        self.db = Database()
#    
#    def extract_schema(self):
#        """
#        Automatically extract ALL schema information from connected database
#        """
#        print("🔍 Extracting database schema...")
#        
#        schema = {
#            "database_name": self.db.database,
#            "extracted_at": str(datetime.datetime.now()),
#            "tables": {}
#        }
#        
#        try:
#            # Connect to database
#            if not self.db.connect():
#                print("❌ Cannot connect to database")
#                return schema
#            
#            cursor = self.db.connection.cursor(dictionary=True)
#            
#            # 1. GET ALL TABLE NAMES
#            cursor.execute("SHOW TABLES")
#            tables_result = cursor.fetchall()
#            
#            if not tables_result:
#                print("❌ No tables found in database")
#                return schema
#            
#            # Extract table names
#            table_names = []
#            for row in tables_result:
#                # Get the first value (table name)
#                table_name = list(row.values())[0]
#                table_names.append(table_name)
#            
#            print(f"✅ Found {len(table_names)} tables: {table_names}")
#            
#            # 2. ANALYZE EACH TABLE
#            for table_name in table_names:
#                print(f"📊 Analyzing table: {table_name}")
#                
#                table_info = {
#                    "columns": {},
#                    "row_count": 0,
#                    "sample_data": []
#                }
#                
#                # Get column information
#                cursor.execute(f"DESCRIBE {table_name}")
#                columns = cursor.fetchall()
#                
#                # Get row count
#                cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
#                row_count = cursor.fetchone()['count']
#                table_info["row_count"] = row_count
#                
#                # Get sample data (5 rows)
#                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
#                sample_rows = cursor.fetchall()
#                table_info["sample_data"] = sample_rows
#                
#                # Analyze each column
#                for col in columns:
#                    col_name = col['Field']
#                    col_type = col['Type']
#                    
#                    # Get sample values for this column
#                    try:
#                        cursor.execute(f"SELECT DISTINCT {col_name} FROM {table_name} LIMIT 5")
#                        sample_values = [row[col_name] for row in cursor.fetchall()]
#                    except:
#                        sample_values = []
#                    
#                    # Auto-detect column type
#                    data_category = self._detect_column_type(col_name, col_type)
#                    
#                    table_info["columns"][col_name] = {
#                        "type": col_type,
#                        "data_category": data_category,
#                        "nullable": col['Null'] == 'YES',
#                        "key": col['Key'],
#                        "default": col['Default'],
#                        "sample_values": sample_values
#                    }
#                
#                schema["tables"][table_name] = table_info
#            
#            cursor.close()
#            
#            # 3. SAVE SCHEMA TO FILE
#            self.save_to_file(schema)
#            
#            print("✅ Schema extraction complete!")
#            return schema
#            
#        except Exception as e:
#            print(f"❌ Error extracting schema: {e}")
#            return schema
#    
#    def _detect_column_type(self, column_name, mysql_type):
#        """
#        Auto-detect what type of data this column contains
#        """
#        name_lower = column_name.lower()
#        type_lower = str(mysql_type).lower()
#        
#        # Check for ID columns
#        if name_lower.endswith(('_id', 'id', 'code', 'key', 'pk', 'fk')):
#            return "identifier"
#        
#        # Check for date/time
#        if any(word in name_lower for word in ['date', 'time', 'year', 'month', 'day', 'created', 'updated']):
#            return "datetime"
#        
#        # Check for numeric
#        if any(word in name_lower for word in ['amount', 'price', 'cost', 'sales', 'profit', 'quantity', 
#                                               'total', 'discount', 'rate', 'percent', 'score', 'number', 'count']):
#            return "numeric"
#        
#        # Check for categories/text
#        if any(word in name_lower for word in ['name', 'title', 'category', 'type', 'status', 'gender',
#                                               'country', 'city', 'state', 'region', 'segment', 'mode', 'class']):
#            return "categorical"
#        
#        # Based on MySQL type
#        if any(num in type_lower for num in ['int', 'decimal', 'float', 'double', 'real']):
#            return "numeric"
#        elif any(date in type_lower for date in ['date', 'time', 'datetime', 'timestamp', 'year']):
#            return "datetime"
#        elif 'char' in type_lower or 'text' in type_lower:
#            return "text"
#        else:
#            return "unknown"
#    
#    def save_to_file(self, schema, filename="schema_info.json"):
#        """
#        Save schema information to JSON file
#        """
#        try:
#            with open(filename, 'w') as f:
#                json.dump(schema, f, indent=2, default=str)
#            print(f"💾 Schema saved to: {filename}")
#        except Exception as e:
#            print(f"❌ Error saving schema: {e}")
#    
#    def get_schema_text(self):
#        """
#        Convert schema to text format for LLM
#        """
#        schema = self.extract_schema()
#        
#        text = f"DATABASE: {schema['database_name']}\n"
#        text += f"Extracted: {schema['extracted_at']}\n\n"
#        
#        for table_name, table_info in schema['tables'].items():
#            text += f"TABLE: {table_name}\n"
#            text += f"Rows: {table_info['row_count']:,}\n\n"
#            
#            text += "COLUMNS:\n"
#            for col_name, col_info in table_info['columns'].items():
#                text += f"  • {col_name} ({col_info['type']})"
#                text += f" - {col_info['data_category']}"
#                if col_info['key']:
#                    text += f" [{col_info['key']}]"
#                text += "\n"
#                
#                # Show sample values if available
#                if col_info['sample_values']:
#                    samples = col_info['sample_values'][:3]  # First 3 samples
#                    text += f"    Samples: {samples}\n"
#            
#            text += "\n" + "="*60 + "\n\n"
#        
#        return text
#
## Test the schema extractor
#if __name__ == "__main__":
#    print("Testing schema extraction...")
#    extractor = SchemaExtractor()
#    
#    # Extract schema
#    schema = extractor.extract_schema()
#    
#    # Display summary
#    print("\n📋 SCHEMA SUMMARY:")
#    print(f"Database: {schema['database_name']}")
#    print(f"Tables: {len(schema['tables'])}")
#    
#    for table_name, table_info in schema['tables'].items():
#        print(f"\n📊 Table: {table_name}")
#        print(f"  Rows: {table_info['row_count']:,}")
#        print(f"  Columns: {len(table_info['columns'])}")
#        
#        # Show first few columns
#        columns = list(table_info['columns'].keys())[:5]
#        print(f"  Sample columns: {', '.join(columns)}")
#    
#    # Also get text version
#    schema_text = extractor.get_schema_text()
#    print(f"\n📝 Schema text length: {len(schema_text)} characters")
#    
#    # Save text version too
#    with open("schema_text.txt", "w") as f:
#        f.write(schema_text)
#    print("💾 Schema text saved to: schema_text.txt")