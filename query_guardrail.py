"""
Query Guardrail - Validates user queries before processing
"""
import re
from typing import Dict, List, Tuple

class QueryGuardrail:
    def __init__(self):
        # Define allowed and blocked patterns
        self.allowed_intents = ["KPI", "REPORT", "TREND", "COMPARISON", "DISTRIBUTION"]
        self.blocked_keywords = [
            "drop", "delete", "update", "insert", "truncate", "alter",
            "create", "grant", "revoke", "exec", "execute", "shutdown",
            "kill", "format", "remove", "erase", "modify", "change"
        ]
    
    def validate_query(self, user_query: str, intent_result: Dict) -> Dict:
        """
        Validate user query with multiple guardrails
        Returns: {"valid": bool, "reason": str, "suggested_action": str}
        """
        results = []
        
        # 1. Validate intent
        results.append(self._validate_intent(intent_result))
        
        # 2. Check for blocked keywords
        results.append(self._check_blocked_keywords(user_query))
        
        
        # Combine results
        for result in results:
            if not result["valid"]:
                return {
                    "valid": False,
                    "reason": result["reason"],
                    "suggested_action": result.get("suggested_action", "Please rephrase your query."),
                    "failed_check": result.get("check_name", "Unknown")
                }
        
        return {
            "valid": True,
            "reason": "Query passed all guardrails",
            "suggested_action": "Proceed with SQL generation",
            "failed_check": None
        }
    
    def _validate_intent(self, intent_result: Dict) -> Dict:
        """Validate intent classification"""
        intent = intent_result.get("intent", "UNKNOWN")
        
        if intent not in self.allowed_intents:
            return {
                "valid": False,
                "check_name": "intent_validation",
                "reason": f"Intent '{intent}' is not supported. Supported intents: {', '.join(self.allowed_intents)}",
                "suggested_action": "Please ask about sales, profit, orders, or other business metrics."
            }
        
        return {
            "valid": True,
            "check_name": "intent_validation"
        }
    
    def _check_blocked_keywords(self, query: str) -> Dict:
        """Check for blocked/dangerous keywords"""
        query_lower = query.lower()
        
        for keyword in self.blocked_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
                return {
                    "valid": False,
                    "check_name": "blocked_keywords",
                    "reason": f"Query contains blocked keyword: '{keyword}'",
                    "suggested_action": "Please ask about data analysis only, not data modification."
                }
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'drop\s+table', r'delete\s+from', r'update\s+\w+\s+set',
            r'insert\s+into', r'truncate\s+table', r'alter\s+table'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, query_lower):
                return {
                    "valid": False,
                    "check_name": "suspicious_patterns",
                    "reason": "Query appears to contain database modification commands",
                    "suggested_action": "Please ask only for data retrieval and analysis queries."
                }
        
        return {
            "valid": True,
            "check_name": "blocked_keywords"
        }


# Test function
if __name__ == "__main__":
    print("🧪 Testing Query Guardrail...")
    
    guardrail = QueryGuardrail()
    
    test_cases = [
        ("Show total sales by region", {
            "intent": "KPI",
            "metrics": ["sales"],
            "dimensions": ["region"],
            "time_range": None
        }),
        ("DROP TABLE store", {
            "intent": "REPORT",
            "metrics": [],
            "dimensions": [],
            "time_range": None
        }),
        ("Show all data", {
            "intent": "REPORT",
            "metrics": ["data"],
            "dimensions": [],
            "time_range": None
        }),
        ("Compare profit across different product categories for last quarter", {
            "intent": "COMPARISON",
            "metrics": ["profit"],
            "dimensions": ["category"],
            "time_range": "last quarter"
        })
    ]
    
    for query, intent in test_cases:
        print(f"\nQuery: {query}")
        result = guardrail.validate_query(query, intent)
        print(f"Valid: {result['valid']}")
        if not result['valid']:
            print(f"Reason: {result['reason']}")
            print(f"Action: {result['suggested_action']}")








#"""
#Query Guardrail - Validates user queries before processing
#"""
#import re
#from typing import Dict, List, Tuple
#
#class QueryGuardrail:
#    def __init__(self):
#        # Define allowed and blocked patterns
#        self.allowed_intents = ["KPI", "REPORT", "TREND", "COMPARISON", "DISTRIBUTION"]
#        self.blocked_keywords = [
#            "drop", "delete", "update", "insert", "truncate", "alter",
#            "create", "grant", "revoke", "exec", "execute", "shutdown",
#            "kill", "format", "remove", "erase", "modify", "change"
#        ]
#        
#        # Business rules specific to Superstore dataset
#        self.allowed_metrics = [
#            "sales", "profit", "quantity", "discount", "price", "cost",
#            "revenue", "margin", "count", "total", "average", "sum", "max", "min"
#        ]
#        
#        self.allowed_dimensions = [
#            "region", "category", "subcategory", "segment", "state", "city",
#            "product", "customer", "order", "ship", "date", "year", "month",
#            "quarter", "day", "period", "mode", "priority", "type"
#        ]
#    
#    def validate_query(self, user_query: str, intent_result: Dict) -> Dict:
#        """
#        Validate user query with multiple guardrails
#        Returns: {"valid": bool, "reason": str, "suggested_action": str}
#        """
#        results = []
#        
#        # 1. Validate intent
#        results.append(self._validate_intent(intent_result))
#        
#        # 2. Check for blocked keywords
#        results.append(self._check_blocked_keywords(user_query))
#        
#        # 3. Validate metrics against allowed list
#        results.append(self._validate_metrics(intent_result))
#        
#        # 4. Validate dimensions against allowed list
#        results.append(self._validate_dimensions(intent_result))
#        
#        # 5. Check query complexity
#        results.append(self._check_query_complexity(user_query))
#        
#        # Combine results
#        for result in results:
#            if not result["valid"]:
#                return {
#                    "valid": False,
#                    "reason": result["reason"],
#                    "suggested_action": result.get("suggested_action", "Please rephrase your query."),
#                    "failed_check": result.get("check_name", "Unknown")
#                }
#        
#        return {
#            "valid": True,
#            "reason": "Query passed all guardrails",
#            "suggested_action": "Proceed with SQL generation",
#            "failed_check": None
#        }
#    
#    def _validate_intent(self, intent_result: Dict) -> Dict:
#        """Validate intent classification"""
#        intent = intent_result.get("intent", "UNKNOWN")
#        
#        if intent not in self.allowed_intents:
#            return {
#                "valid": False,
#                "check_name": "intent_validation",
#                "reason": f"Intent '{intent}' is not supported. Supported intents: {', '.join(self.allowed_intents)}",
#                "suggested_action": "Please ask about sales, profit, orders, or other business metrics."
#            }
#        
#        return {
#            "valid": True,
#            "check_name": "intent_validation"
#        }
#    
#    def _check_blocked_keywords(self, query: str) -> Dict:
#        """Check for blocked/dangerous keywords"""
#        query_lower = query.lower()
#        
#        for keyword in self.blocked_keywords:
#            if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
#                return {
#                    "valid": False,
#                    "check_name": "blocked_keywords",
#                    "reason": f"Query contains blocked keyword: '{keyword}'",
#                    "suggested_action": "Please ask about data analysis only, not data modification."
#                }
#        
#        # Check for suspicious patterns
#        suspicious_patterns = [
#            r'drop\s+table', r'delete\s+from', r'update\s+\w+\s+set',
#            r'insert\s+into', r'truncate\s+table', r'alter\s+table'
#        ]
#        
#        for pattern in suspicious_patterns:
#            if re.search(pattern, query_lower):
#                return {
#                    "valid": False,
#                    "check_name": "suspicious_patterns",
#                    "reason": "Query appears to contain database modification commands",
#                    "suggested_action": "Please ask only for data retrieval and analysis queries."
#                }
#        
#        return {
#            "valid": True,
#            "check_name": "blocked_keywords"
#        }
#    
#    def _validate_metrics(self, intent_result: Dict) -> Dict:
#        """Validate requested metrics against allowed list"""
#        metrics = intent_result.get("metrics", [])
#        
#        if not metrics:
#            return {"valid": True, "check_name": "metrics_validation"}
#        
#        invalid_metrics = []
#        for metric in metrics:
#            metric_lower = metric.lower()
#            # Check if metric contains any allowed word
#            if not any(allowed in metric_lower for allowed in self.allowed_metrics):
#                # Also check for partial matches
#                if not self._is_metric_allowed(metric_lower):
#                    invalid_metrics.append(metric)
#        
#        if invalid_metrics:
#            return {
#                "valid": False,
#                "check_name": "metrics_validation",
#                "reason": f"Metrics not supported: {', '.join(invalid_metrics)}",
#                "suggested_action": f"Try using: {', '.join(self.allowed_metrics)}"
#            }
#        
#        return {
#            "valid": True,
#            "check_name": "metrics_validation"
#        }
#    
#    def _validate_dimensions(self, intent_result: Dict) -> Dict:
#        """Validate requested dimensions against allowed list"""
#        dimensions = intent_result.get("dimensions", [])
#        
#        if not dimensions:
#            return {"valid": True, "check_name": "dimensions_validation"}
#        
#        invalid_dimensions = []
#        for dimension in dimensions:
#            dimension_lower = dimension.lower()
#            if not any(allowed in dimension_lower for allowed in self.allowed_dimensions):
#                invalid_dimensions.append(dimension)
#        
#        if invalid_dimensions:
#            return {
#                "valid": False,
#                "check_name": "dimensions_validation",
#                "reason": f"Dimensions not supported: {', '.join(invalid_dimensions)}",
#                "suggested_action": f"Try using: {', '.join(self.allowed_dimensions)}"
#            }
#        
#        return {
#            "valid": True,
#            "check_name": "dimensions_validation"
#        }
#    
#    def _check_query_complexity(self, query: str) -> Dict:
#        """Check if query is too complex or vague"""
#        words = query.split()
#        
#        # Too short
#        if len(words) < 2:
#            return {
#                "valid": False,
#                "check_name": "query_complexity",
#                "reason": "Query is too vague. Please be more specific.",
#                "suggested_action": "Example: 'Show total sales by region last quarter'"
#            }
#        
#        # Too long
#        if len(words) > 50:
#            return {
#                "valid": False,
#                "check_name": "query_complexity",
#                "reason": "Query is too complex. Please simplify your request.",
#                "suggested_action": "Break down into smaller questions."
#            }
#        
#        # Check for actual business terms
#        business_terms = ['sales', 'profit', 'order', 'customer', 'product', 'region']
#        has_business_term = any(term in query.lower() for term in business_terms)
#        
#        if not has_business_term:
#            return {
#                "valid": True,  # Not invalid, just warning
#                "check_name": "query_complexity",
#                "reason": "Query doesn't appear to be about business data",
#                "suggested_action": "Please ask about business metrics like sales, profit, orders, etc."
#            }
#        
#        return {
#            "valid": True,
#            "check_name": "query_complexity"
#        }
#    
#    def _is_metric_allowed(self, metric: str) -> bool:
#        """Check if metric is allowed (flexible matching)"""
#        # Common variations
#        metric_variations = {
#            'revenue': 'sales',
#            'margin': 'profit',
#            'amount': 'sales',
#            'value': 'sales',
#            'number': 'quantity',
#            'qty': 'quantity',
#            'disc': 'discount',
#            'rev': 'sales'
#        }
#        
#        if metric in metric_variations:
#            return True
#        
#        # Check for partial matches
#        for allowed in self.allowed_metrics:
#            if allowed in metric or metric in allowed:
#                return True
#        
#        return False
#
#
## Test function
#if __name__ == "__main__":
#    print("🧪 Testing Query Guardrail...")
#    
#    guardrail = QueryGuardrail()
#    
#    test_cases = [
#        ("Show total sales by region", {
#            "intent": "KPI",
#            "metrics": ["sales"],
#            "dimensions": ["region"],
#            "time_range": None
#        }),
#        ("DROP TABLE store", {
#            "intent": "REPORT",
#            "metrics": [],
#            "dimensions": [],
#            "time_range": None
#        }),
#        ("Show all data", {
#            "intent": "REPORT",
#            "metrics": ["data"],
#            "dimensions": [],
#            "time_range": None
#        }),
#        ("Compare profit across different product categories for last quarter", {
#            "intent": "COMPARISON",
#            "metrics": ["profit"],
#            "dimensions": ["category"],
#            "time_range": "last quarter"
#        })
#    ]
#    
#    for query, intent in test_cases:
#        print(f"\nQuery: {query}")
#        result = guardrail.validate_query(query, intent)
#        print(f"Valid: {result['valid']}")
#        if not result['valid']:
#            print(f"Reason: {result['reason']}")
#            print(f"Action: {result['suggested_action']}")