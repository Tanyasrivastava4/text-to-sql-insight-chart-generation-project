"""
Data Analysis Plugin - Updated to prepare better chart data
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any

class DataAnalyzer:
    def __init__(self):
        pass
    
    def analyze(self, data: List[Dict], intent: Dict, columns: List[str]) -> Dict[str, Any]:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Prepare chart data
        chart_data = self._prepare_chart_data(df, columns, intent)
        
        return {
            "original_data": data,
            "columns": columns,
            "row_count": len(data),
            "analysis": self._perform_analysis(df, intent),
            "chart_data": chart_data,
            "chart_type": self._determine_chart_type(df, columns, intent)
        }
    
    def _prepare_chart_data(self, df: pd.DataFrame, columns: List[str], intent: Dict) -> Dict:
        """Prepare data specifically for charting"""
        if df.empty:
            return {}
        
        # Try to identify best columns for charting
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Default: use first categorical and first numeric
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            # Sort by numeric value for better visualization
            df_sorted = df.sort_values(num_col, ascending=False)
            
            return {
                "labels": df_sorted[cat_col].tolist(),
                "values": df_sorted[num_col].tolist(),
                "x_label": cat_col,
                "y_label": num_col,
                "title": f"{num_col} by {cat_col}"
            }
        elif numeric_cols:
            # Only numeric columns - show distribution
            num_col = numeric_cols[0]
            return {
                "labels": [f"Row {i+1}" for i in range(len(df))],
                "values": df[num_col].tolist(),
                "x_label": "Index",
                "y_label": num_col,
                "title": f"Distribution of {num_col}"
            }
        else:
            # No numeric columns - just show counts
            return {
                "labels": [str(i) for i in range(len(df))],
                "values": [1] * len(df),
                "x_label": "Index",
                "y_label": "Count",
                "title": "Data Distribution"
            }
    
    def _perform_analysis(self, df: pd.DataFrame, intent: Dict) -> Dict:
        """Perform analysis based on intent"""
        analysis = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if not numeric_cols.empty:
            for col in numeric_cols:
                analysis[col] = {
                    "sum": float(df[col].sum()),
                    "average": float(df[col].mean()),
                    "max": float(df[col].max()),
                    "min": float(df[col].min()),
                    "count": int(df[col].count())
                }
        
        return analysis
    
    def _determine_chart_type(self, df: pd.DataFrame, columns: List[str], intent: Dict) -> str:
        """Determine appropriate chart type"""
        intent_type = intent.get("intent", "REPORT")
        
        if intent_type == "COMPARISON":
            return "bar"
        elif intent_type == "TREND":
            return "line"
        elif intent_type == "DISTRIBUTION":
            return "pie"
        else:
            return "bar"  # Default








##worked but changed bcz not get chart in o/p
#"""
#Data Analysis Plugin
#Performs calculations based on intent
#"""
#import pandas as pd
#import numpy as np
#from typing import Dict, List, Any, Tuple
#
#class DataAnalyzer:
#    def __init__(self):
#        pass
#    
#    def analyze(self, data: List[Dict], intent: Dict, columns: List[str]) -> Dict[str, Any]:
#        """
#        Analyze data based on intent
#        Returns enriched data with calculations
#        """
#        # Convert to DataFrame for easier analysis
#        df = pd.DataFrame(data)
#        
#        analysis_result = {
#            "original_data": data,
#            "columns": columns,
#            "row_count": len(data),
#            "analysis": {},
#            "chart_data": {},
#            "chart_type": self._determine_chart_type(intent, columns, df)
#        }
#        
#        # Perform analysis based on intent
#        intent_type = intent.get("intent", "REPORT")
#        
#        if intent_type == "KPI":
#            analysis_result = self._analyze_kpi(df, intent, analysis_result)
#        elif intent_type == "TREND":
#            analysis_result = self._analyze_trend(df, intent, analysis_result)
#        elif intent_type == "COMPARISON":
#            analysis_result = self._analyze_comparison(df, intent, analysis_result)
#        elif intent_type == "DISTRIBUTION":
#            analysis_result = self._analyze_distribution(df, intent, analysis_result)
#        else:  # REPORT or default
#            analysis_result = self._analyze_report(df, intent, analysis_result)
#        
#        return analysis_result
#    
#    def _determine_chart_type(self, intent: Dict, columns: List[str], df: pd.DataFrame) -> str:
#        """Determine appropriate chart type based on data and intent"""
#        intent_type = intent.get("intent", "REPORT")
#        
#        if intent_type == "COMPARISON" or intent_type == "DISTRIBUTION":
#            if len(columns) == 2:  # Category + Value
#                if len(df) <= 8:
#                    return "bar"  # Bar chart for comparisons
#                else:
#                    return "bar"  # Still bar for many items
#        elif intent_type == "TREND":
#            # Check if we have date/time column
#            date_columns = [col for col in columns if any(word in col.lower() 
#                          for word in ['date', 'time', 'year', 'month', 'day'])]
#            if date_columns:
#                return "line"
#        elif intent_type == "KPI":
#            if len(df) == 1 and len(columns) >= 1:
#                return "gauge" if len(df) == 1 else "bar"
#        
#        # Default to bar chart
#        return "bar"
#    
#    def _analyze_kpi(self, df: pd.DataFrame, intent: Dict, result: Dict) -> Dict:
#        """Analyze for KPI intent - calculate metrics"""
#        metrics = intent.get("metrics", [])
#        
#        result["analysis"]["kpi_metrics"] = {}
#        
#        # Calculate basic statistics for numeric columns
#        numeric_cols = df.select_dtypes(include=[np.number]).columns
#        
#        for col in numeric_cols:
#            result["analysis"]["kpi_metrics"][col] = {
#                "sum": float(df[col].sum()),
#                "average": float(df[col].mean()),
#                "max": float(df[col].max()),
#                "min": float(df[col].min()),
#                "count": int(df[col].count())
#            }
#        
#        # Prepare chart data
#        if len(numeric_cols) > 0:
#            first_numeric = numeric_cols[0]
#            result["chart_data"] = {
#                "labels": ["Value"],
#                "values": [float(df[first_numeric].sum())],
#                "type": "kpi"
#            }
#        
#        return result
#    
#    def _analyze_comparison(self, df: pd.DataFrame, intent: Dict, result: Dict) -> Dict:
#        """Analyze for COMPARISON intent - calculate percentages, ranks, ratios"""
#        dimensions = intent.get("dimensions", [])
#        
#        result["analysis"]["comparison"] = {}
#        
#        # Identify category and value columns
#        numeric_cols = df.select_dtypes(include=[np.number]).columns
#        
#        if len(df) > 0 and len(numeric_cols) > 0:
#            # Assume first column is category, first numeric is value
#            category_col = df.columns[0] if len(df.columns) > 0 else "category"
#            value_col = numeric_cols[0]
#            
#            # Calculate percentages
#            total = df[value_col].sum()
#            df_copy = df.copy()
#            df_copy['percentage'] = (df_copy[value_col] / total * 100).round(2)
#            df_copy['rank'] = df_copy[value_col].rank(ascending=False, method='min')
#            
#            # Add ratio to top
#            if len(df) > 1:
#                top_value = df_copy[value_col].max()
#                df_copy['ratio_to_top'] = (df_copy[value_col] / top_value * 100).round(2)
#            
#            result["analysis"]["comparison"]["data"] = df_copy.to_dict('records')
#            result["analysis"]["comparison"]["total"] = float(total)
#            
#            # Prepare chart data
#            result["chart_data"] = {
#                "labels": df[category_col].tolist() if category_col in df.columns else list(range(len(df))),
#                "values": df[value_col].tolist(),
#                "percentages": df_copy['percentage'].tolist(),
#                "type": "comparison"
#            }
#        
#        return result
#    
#    def _analyze_trend(self, df: pd.DataFrame, intent: Dict, result: Dict) -> Dict:
#        """Analyze for TREND intent - calculate growth rates, differences"""
#        result["analysis"]["trend"] = {}
#        
#        numeric_cols = df.select_dtypes(include=[np.number]).columns
#        
#        if len(df) > 1 and len(numeric_cols) > 0:
#            value_col = numeric_cols[0]
#            
#            # Sort by first column (assumed to be time dimension)
#            df_sorted = df.sort_values(df.columns[0])
#            
#            # Calculate growth rates
#            values = df_sorted[value_col].tolist()
#            growth_rates = []
#            for i in range(1, len(values)):
#                if values[i-1] != 0:
#                    growth = ((values[i] - values[i-1]) / values[i-1]) * 100
#                    growth_rates.append(round(growth, 2))
#                else:
#                    growth_rates.append(0)
#            
#            result["analysis"]["trend"]["growth_rates"] = growth_rates
#            result["analysis"]["trend"]["total_growth"] = round(((values[-1] - values[0]) / values[0] * 100), 2) if values[0] != 0 else 0
#            
#            # Prepare chart data
#            result["chart_data"] = {
#                "labels": df_sorted[df_sorted.columns[0]].tolist(),
#                "values": values,
#                "growth_rates": growth_rates,
#                "type": "trend"
#            }
#        
#        return result
#    
#    def _analyze_distribution(self, df: pd.DataFrame, intent: Dict, result: Dict) -> Dict:
#        """Analyze for DISTRIBUTION intent - calculate frequencies, histograms"""
#        result["analysis"]["distribution"] = {}
#        
#        if len(df.columns) > 0:
#            # For categorical data
#            first_col = df.columns[0]
#            value_counts = df[first_col].value_counts().to_dict()
#            
#            result["analysis"]["distribution"]["frequencies"] = value_counts
#            
#            # Calculate percentages
#            total = sum(value_counts.values())
#            percentages = {k: round((v/total)*100, 2) for k, v in value_counts.items()}
#            result["analysis"]["distribution"]["percentages"] = percentages
#            
#            # Prepare chart data
#            result["chart_data"] = {
#                "labels": list(value_counts.keys()),
#                "values": list(value_counts.values()),
#                "percentages": list(percentages.values()),
#                "type": "distribution"
#            }
#        
#        return result
#    
#    def _analyze_report(self, df: pd.DataFrame, intent: Dict, result: Dict) -> Dict:
#        """Default analysis for REPORT intent"""
#        result["analysis"]["report"] = {
#            "summary_stats": df.describe().to_dict(),
#            "total_rows": len(df),
#            "columns": list(df.columns)
#        }
#        
#        # Prepare basic chart data
#        if len(df.columns) >= 2:
#            numeric_cols = df.select_dtypes(include=[np.number]).columns
#            if len(numeric_cols) > 0:
#                value_col = numeric_cols[0]
#                category_col = df.columns[0]
#                
#                result["chart_data"] = {
#                    "labels": df[category_col].tolist() if category_col in df.columns else [],
#                    "values": df[value_col].tolist() if value_col in df.columns else [],
#                    "type": "report"
#                }
#        
#        return result