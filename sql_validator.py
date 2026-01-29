"""
STRICT SQL Validator with Guardrails
Only validates - NO modifications, NO suggestions
"""
import re

class SQLValidator:
    def __init__(self):
        self.validation_rules = {
            "read_only": True,
            "allow_select": True,
            "allow_aggregations": True,
            "allow_subqueries": True,
            "allow_joins": True,
            "allow_unions": True
        }
    
    def validate(self, sql_query: str) -> str:
        """
        This is the method being called from langchain_agent.py
        Simply calls validate_sql() and returns the result
        """
        return self.validate_sql(sql_query)
    
    def validate_sql(self, sql_query: str) -> str:
        """
        STRICT validation ONLY
        Returns: "VALID_SQL" or "INVALID_SQL"
        NO explanations, NO modifications
        """
        # Clean the query
        sql = self._clean_query(sql_query)
        
        # Check 1: Must be a SELECT query
        if not self._is_select_query(sql):
            return "INVALID_SQL"
        
        # Check 2: No dangerous operations
        if self._has_dangerous_operations(sql):
            return "INVALID_SQL"
        
        # Check 3: Basic syntax validation
        if not self._has_valid_syntax(sql):
            return "INVALID_SQL"
        
        # Check 4: Aggregation syntax
        if not self._has_valid_aggregations(sql):
            return "INVALID_SQL"
        
        # Check 5: No system/database modification
        if self._modifies_system(sql):
            return "INVALID_SQL"
        
        # If all checks pass
        return "VALID_SQL"
    
    def _clean_query(self, sql: str) -> str:
        """Remove comments and extra whitespace"""
        # Remove SQL comments
        sql = re.sub(r'--.*', '', sql)  # Line comments
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)  # Block comments
        
        # Remove extra whitespace
        sql = ' '.join(sql.split())
        
        return sql.strip().upper()
    
    def _is_select_query(self, sql: str) -> bool:
        """Check if query starts with SELECT"""
        # Find first significant word (skip WITH clauses)
        words = sql.split()
        for word in words:
            if word in ['WITH', '(', ')']:
                continue
            return word == 'SELECT'
        return False
    
    def _has_dangerous_operations(self, sql: str) -> bool:
        """Check for dangerous SQL operations"""
        dangerous_keywords = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'TRUNCATE', 'ALTER',
            'CREATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE', 'MERGE',
            'REPLACE', 'LOAD', 'OUTFILE', 'INTO OUTFILE', 'SHUTDOWN',
            'KILL', 'LOCK', 'UNLOCK', 'BEGIN', 'COMMIT', 'ROLLBACK',
            'SAVEPOINT', 'SET', 'USE', 'DESCRIBE', 'EXPLAIN', 'SHOW'
        ]
        
        # Check for dangerous keywords (case-insensitive)
        sql_lower = sql.upper()
        for keyword in dangerous_keywords:
            # Match whole words only (not parts of other words)
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, sql_lower):
                # Allow SELECT ... INTO OUTFILE? NO!
                if keyword == 'INTO' and 'INTO OUTFILE' in sql_lower:
                    return True
                # Allow normal INTO in SELECT INTO variable? NO!
                if keyword == 'INTO':
                    return True
                return True
        
        return False
    
    def _has_valid_syntax(self, sql: str) -> bool:
        """Basic syntax validation"""
        # Must have SELECT and FROM
        if 'SELECT' not in sql or 'FROM' not in sql:
            return False
        
        # Check parentheses balance
        if sql.count('(') != sql.count(')'):
            return False
        
        # Check quote balance
        if (sql.count("'") % 2 != 0) or (sql.count('"') % 2 != 0):
            return False
        
        # FROM must come after SELECT
        select_pos = sql.find('SELECT')
        from_pos = sql.find('FROM')
        if from_pos < select_pos:
            return False
        
        return True
    
    def _has_valid_aggregations(self, sql: str) -> bool:
        """Validate aggregation function syntax"""
        # List of valid aggregation functions
        aggregations = ['SUM', 'COUNT', 'AVG', 'MIN', 'MAX', 'GROUP_CONCAT']
        
        # Find all aggregation functions
        for agg in aggregations:
            pattern = r'\b' + agg + r'\s*\('
            matches = re.finditer(pattern, sql, re.IGNORECASE)
            
            for match in matches:
                # Get the text after the opening parenthesis
                start_pos = match.end()
                # Find the matching closing parenthesis
                paren_count = 1
                current_pos = start_pos
                
                while current_pos < len(sql) and paren_count > 0:
                    if sql[current_pos] == '(':
                        paren_count += 1
                    elif sql[current_pos] == ')':
                        paren_count -= 1
                    current_pos += 1
                
                if paren_count != 0:  # Unbalanced parentheses
                    return False
        
        return True
    
    def _modifies_system(self, sql: str) -> bool:
        """Check for system/database modification attempts"""
        system_patterns = [
            r'INFORMATION_SCHEMA\b',
            r'PERFORMANCE_SCHEMA\b',
            r'MYSQL\b',
            r'SYS\b',
            r'@@',
            r'@\w+',  # User variables
            r'SET\s+@',
            r'SELECT\s+.*INTO\s+@'
        ]
        
        for pattern in system_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                return True
        
        return False
    
    def validate_with_guardrail(self, sql_query: str, original_query: str = "") -> dict:
        """
        Validator with guardrail context
        Returns: {"status": "VALID_SQL" or "INVALID_SQL", "message": ""}
        """
        result = self.validate_sql(sql_query)
        
        return {
            "status": result,
            "message": ""  # No explanations per rules
        }

# Simple test function
def test_validator():
    """Test the SQL validator"""
    validator = SQLValidator()
    
    test_queries = [
        # Valid queries
        ("SELECT * FROM store", True),
        ("SELECT Category, SUM(Sales) FROM store GROUP BY Category", True),
        ("SELECT COUNT(*) FROM store WHERE Sales > 100", True),
        ("SELECT AVG(Profit) as avg_profit FROM store", True),
        
        # Invalid queries (dangerous)
        ("DROP TABLE store", False),
        ("DELETE FROM store", False),
        ("UPDATE store SET Sales = 100", False),
        ("INSERT INTO store VALUES (1, 'test')", False),
        
        # Invalid syntax
        ("SELECT FROM", False),  # No columns
        ("SELECT * FROM", False),  # No table
        ("SELECT * FRO store", False),  # Typo
        ("SELECT * FROM store WHERE", False),  # Incomplete WHERE
        
        # System modification attempts
        ("SELECT * FROM mysql.user", False),
        ("SELECT @@version", False),
        ("SELECT * INTO @var FROM store", False),
    ]
    
    print("🧪 Testing SQL Validator...")
    print("=" * 60)
    
    passed = 0
    total = len(test_queries)
    
    for sql, expected_valid in test_queries:
        result = validator.validate_sql(sql)
        is_valid = result == "VALID_SQL"
        
        status = "✅ PASS" if is_valid == expected_valid else "❌ FAIL"
        passed += 1 if is_valid == expected_valid else 0
        
        print(f"{status} | Expected: {'VALID' if expected_valid else 'INVALID'} | Got: {result}")
        print(f"  Query: {sql}")
        print()
    
    print(f"📊 Test Results: {passed}/{total} passed")
    return passed == total

if __name__ == "__main__":
    # Run tests
    success = test_validator()
    
    if success:
        print("🎉 All tests passed! Validator is ready.")
    else:
        print("⚠️ Some tests failed. Check validator logic.")
    
    # Example usage
    print("\n" + "=" * 60)
    print("Example usage in code:")
    
    validator = SQLValidator()
    sample_sql = "SELECT Category, SUM(Sales) as Total_Sales FROM store GROUP BY Category"
    
    print(f"Query: {sample_sql}")
    result = validator.validate(sample_sql)  # Testing the new validate() method
    print(f"Validation result: {result}")
    
    # With guardrail
    guardrail_result = validator.validate_with_guardrail(sample_sql)
    print(f"Guardrail result: {guardrail_result}")


















#"""
#STRICT SQL Validator with Guardrails
#Only validates - NO modifications, NO suggestions
#"""
#import re
#
#class SQLValidator:
#    def __init__(self):
#        self.validation_rules = {
#            "read_only": True,
#            "allow_select": True,
#            "allow_aggregations": True,
#            "allow_subqueries": True,
#            "allow_joins": True,
#            "allow_unions": True
#        }
#    
#    def validate_sql(self, sql_query: str) -> str:
#        """
#        STRICT validation ONLY
#        Returns: "VALID_SQL" or "INVALID_SQL"
#        NO explanations, NO modifications
#        """
#        # Clean the query
#        sql = self._clean_query(sql_query)
#        
#        # Check 1: Must be a SELECT query
#        if not self._is_select_query(sql):
#            return "INVALID_SQL"
#        
#        # Check 2: No dangerous operations
#        if self._has_dangerous_operations(sql):
#            return "INVALID_SQL"
#        
#        # Check 3: Basic syntax validation
#        if not self._has_valid_syntax(sql):
#            return "INVALID_SQL"
#        
#        # Check 4: Aggregation syntax
#        if not self._has_valid_aggregations(sql):
#            return "INVALID_SQL"
#        
#        # Check 5: No system/database modification
#        if self._modifies_system(sql):
#            return "INVALID_SQL"
#        
#        # If all checks pass
#        return "VALID_SQL"
#    
#    def _clean_query(self, sql: str) -> str:
#        """Remove comments and extra whitespace"""
#        # Remove SQL comments
#        sql = re.sub(r'--.*', '', sql)  # Line comments
#        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)  # Block comments
#        
#        # Remove extra whitespace
#        sql = ' '.join(sql.split())
#        
#        return sql.strip().upper()
#    
#    def _is_select_query(self, sql: str) -> bool:
#        """Check if query starts with SELECT"""
#        # Find first significant word (skip WITH clauses)
#        words = sql.split()
#        for word in words:
#            if word in ['WITH', '(', ')']:
#                continue
#            return word == 'SELECT'
#        return False
#    
#    def _has_dangerous_operations(self, sql: str) -> bool:
#        """Check for dangerous SQL operations"""
#        dangerous_keywords = [
#            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'TRUNCATE', 'ALTER',
#            'CREATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE', 'MERGE',
#            'REPLACE', 'LOAD', 'OUTFILE', 'INTO OUTFILE', 'SHUTDOWN',
#            'KILL', 'LOCK', 'UNLOCK', 'BEGIN', 'COMMIT', 'ROLLBACK',
#            'SAVEPOINT', 'SET', 'USE', 'DESCRIBE', 'EXPLAIN', 'SHOW'
#        ]
#        
#        # Check for dangerous keywords (case-insensitive)
#        sql_lower = sql.upper()
#        for keyword in dangerous_keywords:
#            # Match whole words only (not parts of other words)
#            pattern = r'\b' + re.escape(keyword) + r'\b'
#            if re.search(pattern, sql_lower):
#                # Allow SELECT ... INTO OUTFILE? NO!
#                if keyword == 'INTO' and 'INTO OUTFILE' in sql_lower:
#                    return True
#                # Allow normal INTO in SELECT INTO variable? NO!
#                if keyword == 'INTO':
#                    return True
#                return True
#        
#        return False
#    
#    def _has_valid_syntax(self, sql: str) -> bool:
#        """Basic syntax validation"""
#        # Must have SELECT and FROM
#        if 'SELECT' not in sql or 'FROM' not in sql:
#            return False
#        
#        # Check parentheses balance
#        if sql.count('(') != sql.count(')'):
#            return False
#        
#        # Check quote balance
#        if (sql.count("'") % 2 != 0) or (sql.count('"') % 2 != 0):
#            return False
#        
#        # FROM must come after SELECT
#        select_pos = sql.find('SELECT')
#        from_pos = sql.find('FROM')
#        if from_pos < select_pos:
#            return False
#        
#        return True
#    
#    def _has_valid_aggregations(self, sql: str) -> bool:
#        """Validate aggregation function syntax"""
#        # List of valid aggregation functions
#        aggregations = ['SUM', 'COUNT', 'AVG', 'MIN', 'MAX', 'GROUP_CONCAT']
#        
#        # Find all aggregation functions
#        for agg in aggregations:
#            pattern = r'\b' + agg + r'\s*\('
#            matches = re.finditer(pattern, sql, re.IGNORECASE)
#            
#            for match in matches:
#                # Get the text after the opening parenthesis
#                start_pos = match.end()
#                # Find the matching closing parenthesis
#                paren_count = 1
#                current_pos = start_pos
#                
#                while current_pos < len(sql) and paren_count > 0:
#                    if sql[current_pos] == '(':
#                        paren_count += 1
#                    elif sql[current_pos] == ')':
#                        paren_count -= 1
#                    current_pos += 1
#                
#                if paren_count != 0:  # Unbalanced parentheses
#                    return False
#        
#        return True
#    
#    def _modifies_system(self, sql: str) -> bool:
#        """Check for system/database modification attempts"""
#        system_patterns = [
#            r'INFORMATION_SCHEMA\b',
#            r'PERFORMANCE_SCHEMA\b',
#            r'MYSQL\b',
#            r'SYS\b',
#            r'@@',
#            r'@\w+',  # User variables
#            r'SET\s+@',
#            r'SELECT\s+.*INTO\s+@'
#        ]
#        
#        for pattern in system_patterns:
#            if re.search(pattern, sql, re.IGNORECASE):
#                return True
#        
#        return False
#    
#    def validate_with_guardrail(self, sql_query: str, original_query: str = "") -> dict:
#        """
#        Validator with guardrail context
#        Returns: {"status": "VALID_SQL" or "INVALID_SQL", "message": ""}
#        """
#        result = self.validate_sql(sql_query)
#        
#        return {
#            "status": result,
#            "message": ""  # No explanations per rules
#        }
#
## Simple test function
#def test_validator():
#    """Test the SQL validator"""
#    validator = SQLValidator()
#    
#    test_queries = [
#        # Valid queries
#        ("SELECT * FROM store", True),
#        ("SELECT Category, SUM(Sales) FROM store GROUP BY Category", True),
#        ("SELECT COUNT(*) FROM store WHERE Sales > 100", True),
#        ("SELECT AVG(Profit) as avg_profit FROM store", True),
#        
#        # Invalid queries (dangerous)
#        ("DROP TABLE store", False),
#        ("DELETE FROM store", False),
#        ("UPDATE store SET Sales = 100", False),
#        ("INSERT INTO store VALUES (1, 'test')", False),
#        
#        # Invalid syntax
#        ("SELECT FROM", False),  # No columns
#        ("SELECT * FROM", False),  # No table
#        ("SELECT * FRO store", False),  # Typo
#        ("SELECT * FROM store WHERE", False),  # Incomplete WHERE
#        
#        # System modification attempts
#        ("SELECT * FROM mysql.user", False),
#        ("SELECT @@version", False),
#        ("SELECT * INTO @var FROM store", False),
#    ]
#    
#    print("🧪 Testing SQL Validator...")
#    print("=" * 60)
#    
#    passed = 0
#    total = len(test_queries)
#    
#    for sql, expected_valid in test_queries:
#        result = validator.validate_sql(sql)
#        is_valid = result == "VALID_SQL"
#        
#        status = "✅ PASS" if is_valid == expected_valid else "❌ FAIL"
#        passed += 1 if is_valid == expected_valid else 0
#        
#        print(f"{status} | Expected: {'VALID' if expected_valid else 'INVALID'} | Got: {result}")
#        print(f"  Query: {sql}")
#        print()
#    
#    print(f"📊 Test Results: {passed}/{total} passed")
#    return passed == total
#
#if __name__ == "__main__":
#    # Run tests
#    success = test_validator()
#    
#    if success:
#        print("🎉 All tests passed! Validator is ready.")
#    else:
#        print("⚠️ Some tests failed. Check validator logic.")
#    
#    # Example usage
#    print("\n" + "=" * 60)
#    print("Example usage in code:")
#    
#    validator = SQLValidator()
#    sample_sql = "SELECT Category, SUM(Sales) as Total_Sales FROM store GROUP BY Category"
#    
#    print(f"Query: {sample_sql}")
#    result = validator.validate_sql(sample_sql)
#    print(f"Validation result: {result}")
#    
#    # With guardrail
#    guardrail_result = validator.validate_with_guardrail(sample_sql)
#    print(f"Guardrail result: {guardrail_result}")