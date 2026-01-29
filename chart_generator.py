"""
Smart Chart Generator
Dynamically creates charts based on data structure
"""
import base64
from io import BytesIO
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np

class ChartGenerator:
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    
    def generate_chart(self, analysis_result: Dict) -> Dict[str, Any]:
        """
        Smart chart generation based on data structure
        """
        try:
            # Get data from analysis result
            data = analysis_result.get("original_data", [])
            columns = analysis_result.get("columns", [])
            
            if not data or len(data) == 0:
                return {"error": "No data available for chart"}
            
            # Detect chart type and extract data
            chart_type, chart_data = self._detect_chart_type(data, columns)
            
            if chart_type == "bar":
                return self._generate_bar_chart(chart_data)
            elif chart_type == "line":
                return self._generate_line_chart(chart_data)
            elif chart_type == "pie":
                return self._generate_pie_chart(chart_data)
            elif chart_type == "scatter":
                return self._generate_scatter_chart(chart_data)
            else:
                # Default to bar chart
                return self._generate_bar_chart(chart_data)
                
        except Exception as e:
            return {"error": f"Chart generation error: {str(e)}"}
    
    def _detect_chart_type(self, data: List[Dict], columns: List[str]) -> Tuple[str, Dict]:
        """
        Detect the best chart type based on data structure
        """
        if not data or not columns:
            return "bar", {}
        
        # Convert to lists for easier analysis
        first_row = data[0]
        
        # Try to identify categorical vs numeric columns
        categorical_cols = []
        numeric_cols = []
        
        for col in columns:
            # Check if column contains numeric data
            if self._is_numeric_column(data, col):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        
        print(f"📊 Detected: {len(categorical_cols)} categorical, {len(numeric_cols)} numeric columns")
        print(f"   Categorical: {categorical_cols}")
        print(f"   Numeric: {numeric_cols}")
        
        # Determine best chart type
        if len(numeric_cols) >= 2 and len(categorical_cols) >= 1:
            # Multiple numeric columns with categories - grouped bar chart
            chart_data = self._extract_grouped_data(data, categorical_cols[0], numeric_cols[:2])
            return "bar", chart_data
            
        elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
            # One numeric, one or more categorical - standard bar chart
            chart_data = self._extract_simple_data(data, categorical_cols[0], numeric_cols[0])
            return "bar", chart_data
            
        elif len(numeric_cols) == 2 and len(categorical_cols) == 0:
            # Two numeric columns - scatter plot
            chart_data = self._extract_xy_data(data, numeric_cols[0], numeric_cols[1])
            return "scatter", chart_data
            
        elif len(data) > 10 and len(numeric_cols) == 1:
            # Many rows with one numeric - histogram
            chart_data = self._extract_histogram_data(data, numeric_cols[0])
            return "bar", chart_data
            
        else:
            # Default: extract whatever we can
            chart_data = self._extract_default_data(data, columns)
            return "bar", chart_data
    
    def _is_numeric_column(self, data: List[Dict], column: str) -> bool:
        """Check if a column contains numeric data"""
        for row in data[:10]:  # Check first 10 rows
            value = row.get(column)
            if value is not None:
                try:
                    float(value)
                    return True
                except (ValueError, TypeError):
                    return False
        return False
    
    def _extract_simple_data(self, data: List[Dict], category_col: str, value_col: str) -> Dict:
        """Extract data for simple bar chart"""
        labels = []
        values = []
        
        for row in data:
            label = str(row.get(category_col, ""))
            value = row.get(value_col)
            
            if value is not None:
                try:
                    labels.append(label)
                    values.append(float(value))
                except (ValueError, TypeError):
                    continue
        
        return {
            "type": "bar",
            "labels": labels,
            "values": values,
            "title": f"{value_col} by {category_col}",
            "x_label": category_col,
            "y_label": value_col
        }
    
    def _extract_default_data(self, data: List[Dict], columns: List[str]) -> Dict:
        """Extract data when we can't determine structure"""
        if len(data) == 0:
            return {"type": "bar", "labels": [], "values": [], "title": "Data"}
        
        # Try to use first two columns
        first_row = data[0]
        available_cols = list(first_row.keys())
        
        if len(available_cols) >= 2:
            # Use first column as labels, second as values
            return self._extract_simple_data(data, available_cols[0], available_cols[1])
        else:
            # Just show counts
            labels = [str(i) for i in range(len(data))]
            values = [1] * len(data)
            return {
                "type": "bar",
                "labels": labels,
                "values": values,
                "title": "Data Distribution",
                "x_label": "Index",
                "y_label": "Count"
            }
    
    def _generate_bar_chart(self, chart_data: Dict) -> Dict[str, Any]:
        """Generate bar chart"""
        try:
            labels = chart_data.get("labels", [])
            values = chart_data.get("values", [])
            
            if not labels or not values:
                return {"error": "No data for bar chart"}
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create bars
            bars = ax.bar(range(len(labels)), values, color=self.colors[:len(labels)])
            
            # Customize
            ax.set_xlabel(chart_data.get("x_label", "Categories"))
            ax.set_ylabel(chart_data.get("y_label", "Values"))
            ax.set_title(chart_data.get("title", "Bar Chart"))
            
            # Set x-ticks
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:,.0f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            # Convert to base64
            img_data = self._fig_to_base64(fig)
            plt.close(fig)
            
            return {
                "type": "bar",
                "image_base64": img_data,
                "chart_data": chart_data,
                "success": True
            }
            
        except Exception as e:
            return {"error": f"Bar chart error: {str(e)}"}
    
    def _generate_pie_chart(self, chart_data: Dict) -> Dict[str, Any]:
        """Generate pie chart"""
        try:
            labels = chart_data.get("labels", [])
            values = chart_data.get("values", [])
            
            if not labels or not values:
                return {"error": "No data for pie chart"}
            
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                values, 
                labels=labels, 
                autopct='%1.1f%%',
                startangle=90, 
                colors=self.colors[:len(labels)]
            )
            
            ax.axis('equal')  # Equal aspect ratio
            ax.set_title(chart_data.get("title", "Distribution"))
            
            img_data = self._fig_to_base64(fig)
            plt.close(fig)
            
            return {
                "type": "pie",
                "image_base64": img_data,
                "chart_data": chart_data,
                "success": True
            }
            
        except Exception as e:
            return {"error": f"Pie chart error: {str(e)}"}
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        return img_base64








##changing because didn't get the expected outcome
#"""
#Chart Generator using ECharts
#Generates bar, line, pie charts from analyzed data
#"""
#import json
#import base64
#from io import BytesIO
#from typing import Dict, Any, Optional
#import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('Agg')  # Non-interactive backend
#
#class ChartGenerator:
#    def __init__(self):
#        plt.style.use('seaborn-v0_8-darkgrid')  # Modern style
#    
#    def generate_chart(self, analysis_result: Dict) -> Dict[str, Any]:
#        """
#        Generate chart based on analyzed data
#        Returns dict with chart config and optionally image
#        """
#        chart_type = analysis_result.get("chart_type", "bar")
#        chart_data = analysis_result.get("chart_data", {})
#        
#        if not chart_data or "values" not in chart_data:
#            return {"error": "No chart data available"}
#        
#        labels = chart_data.get("labels", [])
#        values = chart_data.get("values", [])
#        
#        # Generate appropriate chart based on type
#        if chart_type == "bar":
#            return self._generate_bar_chart(labels, values, chart_data)
#        elif chart_type == "line":
#            return self._generate_line_chart(labels, values, chart_data)
#        elif chart_type == "pie":
#            return self._generate_pie_chart(labels, values, chart_data)
#        else:
#            return self._generate_bar_chart(labels, values, chart_data)  # Default
#    
#    def _generate_bar_chart(self, labels: list, values: list, metadata: Dict) -> Dict:
#        """Generate bar chart"""
#        try:
#            # Create figure
#            fig, ax = plt.subplots(figsize=(10, 6))
#            
#            # Create bars
#            bars = ax.bar(labels, values, color=plt.cm.Set3(range(len(labels))))
#            
#            # Add value labels on top of bars
#            for bar in bars:
#                height = bar.get_height()
#                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(values),
#                        f'{height:,.0f}', ha='center', va='bottom')
#            
#            # Customize
#            ax.set_ylabel('Value')
#            ax.set_title('Comparison Chart')
#            plt.xticks(rotation=45, ha='right')
#            plt.tight_layout()
#            
#            # Convert to base64
#            img_data = self._fig_to_base64(fig)
#            plt.close(fig)
#            
#            # ECharts config
#            echarts_config = {
#                "title": {"text": "Data Comparison", "left": "center"},
#                "tooltip": {"trigger": "axis"},
#                "xAxis": {
#                    "type": "category",
#                    "data": labels,
#                    "axisLabel": {"rotate": 45}
#                },
#                "yAxis": {"type": "value"},
#                "series": [{
#                    "data": values,
#                    "type": "bar",
#                    "itemStyle": {"color": "#5470c6"}
#                }]
#            }
#            
#            return {
#                "type": "bar",
#                "image_base64": img_data,
#                "echarts_config": echarts_config,
#                "data": {"labels": labels, "values": values}
#            }
#            
#        except Exception as e:
#            return {"error": f"Chart generation failed: {str(e)}"}
#    
#    def _generate_line_chart(self, labels: list, values: list, metadata: Dict) -> Dict:
#        """Generate line chart for trends"""
#        try:
#            fig, ax = plt.subplots(figsize=(10, 6))
#            
#            ax.plot(labels, values, marker='o', linewidth=2, markersize=8)
#            
#            # Add growth rates if available
#            if "growth_rates" in metadata:
#                growth = metadata["growth_rates"]
#                for i in range(1, len(values)):
#                    ax.annotate(f"{growth[i-1]}%", 
#                              xy=(labels[i], values[i]),
#                              xytext=(0, 10),
#                              textcoords="offset points",
#                              ha='center',
#                              fontsize=9)
#            
#            ax.set_ylabel('Value')
#            ax.set_title('Trend Analysis')
#            plt.xticks(rotation=45, ha='right')
#            plt.grid(True, alpha=0.3)
#            plt.tight_layout()
#            
#            img_data = self._fig_to_base64(fig)
#            plt.close(fig)
#            
#            # ECharts config
#            echarts_config = {
#                "title": {"text": "Trend Analysis", "left": "center"},
#                "tooltip": {"trigger": "axis"},
#                "xAxis": {
#                    "type": "category",
#                    "data": labels,
#                    "axisLabel": {"rotate": 45}
#                },
#                "yAxis": {"type": "value"},
#                "series": [{
#                    "data": values,
#                    "type": "line",
#                    "smooth": True,
#                    "itemStyle": {"color": "#91cc75"}
#                }]
#            }
#            
#            return {
#                "type": "line",
#                "image_base64": img_data,
#                "echarts_config": echarts_config,
#                "data": {"labels": labels, "values": values}
#            }
#            
#        except Exception as e:
#            return {"error": f"Line chart failed: {str(e)}"}
#    
#    def _generate_pie_chart(self, labels: list, values: list, metadata: Dict) -> Dict:
#        """Generate pie chart for distributions"""
#        try:
#            fig, ax = plt.subplots(figsize=(8, 8))
#            
#            # Create pie chart
#            wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
#                                             startangle=90, colors=plt.cm.Set3.colors)
#            
#            # Equal aspect ratio ensures pie is circular
#            ax.axis('equal')
#            ax.set_title('Distribution Analysis')
#            
#            img_data = self._fig_to_base64(fig)
#            plt.close(fig)
#            
#            # ECharts config
#            echarts_config = {
#                "title": {"text": "Distribution", "left": "center"},
#                "tooltip": {"trigger": "item", "formatter": "{a} <br/>{b}: {c} ({d}%)"},
#                "series": [{
#                    "name": "Distribution",
#                    "type": "pie",
#                    "radius": "50%",
#                    "data": [{"value": v, "name": l} for v, l in zip(values, labels)],
#                    "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowOffsetX": 0}}
#                }]
#            }
#            
#            return {
#                "type": "pie",
#                "image_base64": img_data,
#                "echarts_config": echarts_config,
#                "data": {"labels": labels, "values": values}
#            }
#            
#        except Exception as e:
#            return {"error": f"Pie chart failed: {str(e)}"}
#    
#    def _fig_to_base64(self, fig) -> str:
#        """Convert matplotlib figure to base64 string"""
#        buf = BytesIO()
#        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
#        buf.seek(0)
#        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
#        buf.close()
#        return img_base64
#    
#    def generate_chart_html(self, echarts_config: Dict) -> str:
#        """Generate HTML for ECharts"""
#        config_json = json.dumps(echarts_config, ensure_ascii=False)
#        
#        html = f"""
#        <!DOCTYPE html>
#        <html>
#        <head>
#            <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
#            <style>
#                #chart {{ width: 100%; height: 400px; }}
#            </style>
#        </head>
#        <body>
#            <div id="chart"></div>
#            <script>
#                var chartDom = document.getElementById('chart');
#                var myChart = echarts.init(chartDom);
#                var option = {config_json};
#                myChart.setOption(option);
#                
#                // Handle window resize
#                window.addEventListener('resize', function() {{
#                    myChart.resize();
#                }});
#            </script>
#        </body>
#        </html>
#        """
#        
#        return html