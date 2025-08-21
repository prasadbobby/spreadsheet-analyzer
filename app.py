# app.py - Enhanced with conversational AI capabilities

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, date
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid
import traceback
import sys
from pathlib import Path
import requests
from dotenv import load_dotenv
import mimetypes
import xml.etree.ElementTree as ET
from io import StringIO
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import difflib

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"], supports_credentials=True)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-default-secret-key-here')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'data/uploads')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 10 * 1024 * 1024 * 1024))  # 10GB

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)

print(f"App configured with:")
print(f"- Upload folder: {app.config['UPLOAD_FOLDER']}")
print(f"- Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024*1024)} GB")

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime objects"""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super().default(obj)

def safe_json_dumps(obj):
    """Safe JSON serialization with custom encoder"""
    return json.dumps(obj, cls=DateTimeEncoder, default=str)

def convert_dataframe_for_json(df):
    """Convert DataFrame to JSON-safe format"""
    # Make a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    
    # Convert all datetime columns to strings
    for col in df_copy.columns:
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        elif df_copy[col].dtype == 'object':
            # Convert any remaining datetime objects in object columns
            df_copy[col] = df_copy[col].apply(lambda x: x.isoformat() if isinstance(x, (datetime, date, pd.Timestamp)) else x)
    
    return df_copy.to_dict('records')

def get_file_size_formatted(file_path):
    """Get file size with appropriate unit (B, KB, MB, GB)"""
    try:
        if os.path.exists(file_path):
            size_bytes = os.path.getsize(file_path)
            
            # Convert to appropriate unit
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
                size_mb = size_bytes / (1024 * 1024)
            elif size_bytes < 1024 * 1024:  # Less than 1 MB
                size_kb = size_bytes / 1024
                size_str = f"{size_kb:.1f} KB"
                size_mb = size_bytes / (1024 * 1024)
            elif size_bytes < 1024 * 1024 * 1024:  # Less than 1 GB
                size_mb = size_bytes / (1024 * 1024)
                size_str = f"{size_mb:.1f} MB"
            else:  # 1 GB or more
                size_gb = size_bytes / (1024 * 1024 * 1024)
                size_str = f"{size_gb:.1f} GB"
                size_mb = size_bytes / (1024 * 1024)
            
            return {
                'size_bytes': size_bytes,
                'size_mb': round(size_mb, 3),
                'size_formatted': size_str
            }
        else:
            return {
                'size_bytes': 0,
                'size_mb': 0.0,
                'size_formatted': '0 B'
            }
    except Exception as e:
        print(f"Error calculating file size: {e}")
        return {
            'size_bytes': 0,
            'size_mb': 0.0,
            'size_formatted': '0 B'
        }

def detect_actual_file_format(file_path):
    """Detect the actual file format regardless of extension"""
    try:
        # Read first few bytes to detect format
        with open(file_path, 'rb') as f:
            header = f.read(2048)
        
        # Check for HTML/XML content
        if (header.startswith(b'<html') or header.startswith(b'<?xml') or 
            b'<html xm' in header or b'xmlns:' in header or
            b'<worksheet' in header.lower() or b'<table' in header.lower()):
            return 'html_xml'
        
        # Check for true Excel formats
        if header.startswith(b'PK\x03\x04'):  # ZIP signature (xlsx)
            return 'xlsx'
        elif header.startswith(b'\xd0\xcf\x11\xe0'):  # OLE signature (xls)
            return 'xls'
        elif header.startswith(b'\x09\x08\x10\x00') or header.startswith(b'\x09\x08\x08\x00'):  # BIFF signature
            return 'xls'
        
        # Check if it's CSV-like
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_lines = f.read(1024)
            if (',' in first_lines or '\t' in first_lines) and not '<' in first_lines:
                return 'csv_like'
        except:
            pass
        
        return 'unknown'
        
    except Exception as e:
        print(f"Error detecting file format: {e}")
        return 'unknown'

def read_html_xml_as_dataframe(file_path):
    """Read HTML/XML file and convert to DataFrame"""
    try:
        print("Attempting to read as HTML tables...")
        # Try to read as HTML tables first
        tables = pd.read_html(file_path, encoding='utf-8')
        if tables and len(tables) > 0:
            # Use the largest table
            df = max(tables, key=len) if len(tables) > 1 else tables[0]
            print(f"Successfully read HTML table with {len(df)} rows and {len(df.columns)} columns")
            return df
    except Exception as e:
        print(f"HTML table reading failed: {e}")
    
    try:
        print("Attempting to read as XML...")
        # Try to read as XML
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # If it looks like an Excel XML format
        if ('xmlns:ss=' in content or 'xmlns:x=' in content or 'xmlns:o=' in content or
            'mso-application' in content or '<Workbook' in content or '<Worksheet' in content):
            return read_excel_xml_format(content)
        
        # Try generic XML parsing
        return read_generic_xml(content)
        
    except Exception as e:
        print(f"XML reading failed: {e}")
        raise Exception(f"Could not parse HTML/XML file: {e}")

def read_excel_xml_format(content):
    """Read Excel XML format (common with .xls exports that are actually XML)"""
    try:
        print("Parsing Excel XML format...")
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'xml')
        except ImportError:
            # Fallback to basic XML parsing if BeautifulSoup not available
            return read_basic_xml_table(content)
        
        # Look for various Excel XML table structures
        rows_data = []
        headers = []
        
        # Try different XML structures
        worksheets = soup.find_all(['Worksheet', 'worksheet'])
        if worksheets:
            # Excel XML Spreadsheet format
            tables = worksheets[0].find_all(['Table', 'table'])
            if tables:
                rows = tables[0].find_all(['Row', 'row'])
            else:
                rows = worksheets[0].find_all(['Row', 'row'])
        else:
            # Generic table structure
            tables = soup.find_all(['Table', 'table'])
            if tables:
                rows = tables[0].find_all(['Row', 'row', 'tr'])
            else:
                rows = soup.find_all(['Row', 'row', 'tr'])
        
        for i, row in enumerate(rows):
            cells = row.find_all(['Cell', 'cell', 'td'])
            row_data = []
            
            for cell in cells:
                # Extract text content
                data_elem = cell.find(['Data', 'data'])
                if data_elem:
                    text = data_elem.get_text(strip=True)
                else:
                    text = cell.get_text(strip=True)
                
                # Clean up the text
                text = text.replace('\n', ' ').replace('\r', '').strip()
                row_data.append(text)
            
            if row_data and any(cell.strip() for cell in row_data):  # Skip empty rows
                if i == 0 and not headers:
                    headers = [col.strip() or f'Column_{j+1}' for j, col in enumerate(row_data)]
                else:
                    # Pad row to match header length
                    while len(row_data) < len(headers):
                        row_data.append('')
                    rows_data.append(row_data[:len(headers)])  # Trim if too long
        
        if not headers and rows_data:
            # Generate headers if not found
            max_cols = max(len(row) for row in rows_data) if rows_data else 0
            headers = [f'Column_{i+1}' for i in range(max_cols)]
        
        # Create DataFrame
        if rows_data:
            df = pd.DataFrame(rows_data, columns=headers)
        else:
            df = pd.DataFrame(columns=headers)
        
        print(f"Successfully parsed Excel XML format: {len(df)} rows, {len(headers)} columns")
        return df
        
    except Exception as e:
        print(f"Excel XML parsing failed: {e}")
        # Fallback to basic table extraction
        return read_basic_xml_table(content)

def read_basic_xml_table(content):
    """Basic XML table extraction as fallback"""
    try:
        import re
        
        # Extract table-like data using regex
        # Look for row patterns
        row_pattern = r'<(?:tr|row)[^>]*>(.*?)</(?:tr|row)>'
        cell_pattern = r'<(?:td|cell|data)[^>]*>(.*?)</(?:td|cell|data)>'
        
        rows = re.findall(row_pattern, content, re.IGNORECASE | re.DOTALL)
        
        table_data = []
        for row in rows:
            cells = re.findall(cell_pattern, row, re.IGNORECASE | re.DOTALL)
            if cells:
                # Clean cell content
                clean_cells = []
                for cell in cells:
                    # Remove HTML tags and clean text
                    clean_cell = re.sub(r'<[^>]+>', '', cell).strip()
                    clean_cell = clean_cell.replace('&nbsp;', ' ').replace('&amp;', '&')
                    clean_cells.append(clean_cell)
                table_data.append(clean_cells)
        
        if table_data:
            # Use first row as headers if it looks like headers
            headers = table_data[0] if table_data else []
            data_rows = table_data[1:] if len(table_data) > 1 else []
            
            # Ensure all rows have same length
            if data_rows:
                max_cols = max(len(headers), max(len(row) for row in data_rows))
                headers = headers + [f'Column_{i+1}' for i in range(len(headers), max_cols)]
                
                for row in data_rows:
                    while len(row) < max_cols:
                        row.append('')
            
            df = pd.DataFrame(data_rows, columns=headers)
            print(f"Successfully parsed basic XML table: {len(df)} rows, {len(headers)} columns")
            return df
        else:
            raise Exception("No table data found in XML")
            
    except Exception as e:
        print(f"Basic XML parsing failed: {e}")
        raise Exception(f"Could not extract table from XML: {e}")

def read_generic_xml(content):
    """Read generic XML and try to extract tabular data"""
    try:
        root = ET.fromstring(content)
        
        # Find repeating elements (likely rows)
        children = list(root)
        if not children:
            raise Exception("No data elements found in XML")
        
        # Get all unique element names to find the most common (likely row elements)
        element_counts = {}
        for child in children:
            element_counts[child.tag] = element_counts.get(child.tag, 0) + 1
        
        # Find the most common element type
        row_element = max(element_counts.keys(), key=lambda x: element_counts[x])
        row_elements = [child for child in children if child.tag == row_element]
        
        if not row_elements:
            raise Exception("No repeating elements found")
        
        # Extract data from row elements
        all_keys = set()
        for elem in row_elements:
            for child in elem:
                all_keys.add(child.tag)
        
        headers = sorted(list(all_keys))
        rows_data = []
        
        for elem in row_elements:
            row_data = []
            for header in headers:
                child_elem = elem.find(header)
                value = child_elem.text if child_elem is not None else ''
                row_data.append(value or '')
            rows_data.append(row_data)
        
        df = pd.DataFrame(rows_data, columns=headers)
        print(f"Successfully parsed generic XML: {len(df)} rows, {len(headers)} columns")
        return df
        
    except Exception as e:
        print(f"Generic XML parsing failed: {e}")
        raise Exception(f"Could not parse XML structure: {e}")

class ConversationalAI:
    """Enhanced conversational AI for any dataset type"""
    
    def __init__(self):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.gemini_url = os.getenv('GEMINI_API_URL', 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent')
        
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
    
    def analyze_dataset_context(self, dataset_info):
        """Analyze dataset to understand its context and type"""
        try:
            columns = dataset_info['columns']
            sample_data = dataset_info['sample_data']
            
            # Common patterns for different dataset types
            jira_indicators = [
                'key', 'summary', 'description', 'status', 'priority', 'assignee', 
                'reporter', 'issue_type', 'resolution', 'created', 'updated'
            ]
            
            sales_indicators = [
                'customer', 'product', 'revenue', 'sales', 'order', 'amount', 
                'price', 'quantity', 'transaction'
            ]
            
            hr_indicators = [
                'employee', 'name', 'department', 'salary', 'position', 'manager',
                'hire_date', 'performance', 'review'
            ]
            
            finance_indicators = [
                'account', 'balance', 'transaction', 'credit', 'debit', 'amount',
                'invoice', 'payment', 'expense', 'budget'
            ]
            
            # Check which type this dataset might be
            column_str = ' '.join([col.lower() for col in columns])
            
            dataset_type = 'general'
            confidence = 0
            
            jira_score = sum(1 for indicator in jira_indicators if indicator in column_str)
            sales_score = sum(1 for indicator in sales_indicators if indicator in column_str)
            hr_score = sum(1 for indicator in hr_indicators if indicator in column_str)
            finance_score = sum(1 for indicator in finance_indicators if indicator in column_str)
            
            scores = {
                'jira': jira_score,
                'sales': sales_score,
                'hr': hr_score,
                'finance': finance_score
            }
            
            max_score = max(scores.values())
            if max_score >= 3:
                dataset_type = max(scores.keys(), key=lambda x: scores[x])
                confidence = min(max_score / 5, 1.0)
            
            return {
                'type': dataset_type,
                'confidence': confidence,
                'characteristics': self._extract_characteristics(columns, sample_data),
                'suggestions': self._generate_suggestions(dataset_type, columns)
            }
            
        except Exception as e:
            print(f"Error analyzing dataset context: {e}")
            return {
                'type': 'general',
                'confidence': 0,
                'characteristics': {},
                'suggestions': []
            }
    
    def _extract_characteristics(self, columns, sample_data):
        """Extract key characteristics of the dataset"""
        characteristics = {
            'total_columns': len(columns),
            'text_columns': [],
            'numeric_columns': [],
            'date_columns': [],
            'categorical_columns': []
        }
        
        if sample_data:
            first_row = sample_data[0] if sample_data else {}
            
            for col in columns:
                if col in first_row:
                    value = first_row[col]
                    
                    # Check if it looks like a date
                    if isinstance(value, str) and self._is_date_like(value):
                        characteristics['date_columns'].append(col)
                    # Check if numeric
                    elif isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit()):
                        characteristics['numeric_columns'].append(col)
                    # Text columns
                    else:
                        characteristics['text_columns'].append(col)
        
        return characteristics
    
    def _is_date_like(self, value):
        """Check if a string value looks like a date"""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # 2023-12-01
            r'\d{2}/\d{2}/\d{4}',  # 12/01/2023
            r'\d{1,2}/\w{3}/\d{2,4}',  # 1/Jan/23
            r'\w{3}\s+\d{1,2},?\s+\d{4}',  # Jan 1, 2023
        ]
        
        if isinstance(value, str):
            for pattern in date_patterns:
                if re.search(pattern, value):
                    return True
        return False
    
    def _generate_suggestions(self, dataset_type, columns):
        """Generate contextual suggestions based on dataset type"""
        suggestions = []
        
        if dataset_type == 'jira':
            suggestions = [
                "Find tickets with similar error messages to a specific bug",
                "Analyze ticket trends over time periods",
                "Show all critical or high priority issues",
                "Find resolved tickets with similar symptoms",
                "Group tickets by component or assignee",
                "Identify recurring issues or patterns"
            ]
        elif dataset_type == 'sales':
            suggestions = [
                "Show top performing products this quarter",
                "Analyze sales trends by region or time",
                "Find customers with highest revenue",
                "Compare performance across different periods",
                "Identify seasonal patterns in sales",
                "Show revenue breakdown by category"
            ]
        elif dataset_type == 'hr':
            suggestions = [
                "Show employees by department and role",
                "Analyze salary distributions and ranges",
                "Find employees with specific qualifications",
                "Track performance ratings over time",
                "Show hiring trends by period",
                "Compare compensation across departments"
            ]
        elif dataset_type == 'finance':
            suggestions = [
                "Analyze spending patterns by category",
                "Show account balances and transactions",
                "Find unusual or large transactions",
                "Track budget vs actual expenses",
                "Identify payment delays or issues",
                "Show financial trends over time"
            ]
        else:
            # General suggestions
            suggestions = [
                "Show me a summary of the data",
                "Find records with specific criteria",
                "Analyze trends and patterns",
                "Group data by categories",
                "Show statistics and distributions",
                "Find similar or duplicate records"
            ]
        
        return suggestions
    
    def generate_conversational_response(self, question, results, dataset_info, sql_query, dataset_context):
        """Generate a conversational response about the dataset"""
        try:
            result_count = len(results) if results else 0
            dataset_type = dataset_context.get('type', 'general')
            
            # Prepare context for AI
            if result_count > 10:
                results_for_ai = results[:10]
            else:
                results_for_ai = results
            
            results_json = safe_json_dumps(results_for_ai) if results_for_ai else "[]"
            
            # Create a conversational prompt based on dataset type
            if dataset_type == 'jira':
                prompt = self._create_jira_prompt(question, results_json, result_count, dataset_info, sql_query)
            elif dataset_type == 'sales':
                prompt = self._create_sales_prompt(question, results_json, result_count, dataset_info, sql_query)
            elif dataset_type == 'hr':
                prompt = self._create_hr_prompt(question, results_json, result_count, dataset_info, sql_query)
            elif dataset_type == 'finance':
                prompt = self._create_finance_prompt(question, results_json, result_count, dataset_info, sql_query)
            else:
                prompt = self._create_general_prompt(question, results_json, result_count, dataset_info, sql_query)
            
            response = self._call_gemini_api(prompt)
            return response
            
        except Exception as e:
            print(f"Error generating conversational response: {e}")
            return f"I found {result_count} results for your query. The data shows various patterns that might be helpful for your analysis."
    
    def _create_jira_prompt(self, question, results_json, result_count, dataset_info, sql_query):
        """Create JIRA-specific conversational prompt"""
        return f"""
You are a JIRA data analyst assistant helping with ticket analysis. The user asked: "{question}"

JIRA DATASET CONTEXT: 
- Total tickets in dataset: {dataset_info['total_rows']:,}
- Analysis found: {result_count:,} matching tickets
- Columns include: {', '.join(dataset_info['columns'][:10])}{'...' if len(dataset_info['columns']) > 10 else ''}

SQL EXECUTED: {sql_query}
SAMPLE RESULTS (first 10 of {result_count}): {results_json}

Provide a conversational, helpful analysis that:

1. **TICKET OVERVIEW**: Start with "I found {result_count:,} tickets that match your query..."

2. **KEY INSIGHTS**: 
   - Summarize the main findings about the tickets
   - Highlight patterns in status, priority, assignees, or components
   - Mention any trends in creation dates or resolution times
   - Point out critical or high-priority issues if present

3. **SIMILARITY ANALYSIS** (if relevant):
   - Group tickets by similar symptoms or error messages
   - Identify recurring issues or root causes
   - Suggest related tickets that might be helpful

4. **ACTIONABLE RECOMMENDATIONS**:
   - Suggest follow-up queries or investigations
   - Recommend priority actions based on the findings
   - Point out any tickets that need immediate attention

5. **CONTEXT**: Mention that the complete ticket details are available in the table below

Be conversational, specific, and focus on practical insights that would help with ticket management and bug resolution.

Response:
"""

    def _create_sales_prompt(self, question, results_json, result_count, dataset_info, sql_query):
        """Create sales-specific conversational prompt"""
        return f"""
You are a sales data analyst assistant. The user asked: "{question}"

SALES DATASET CONTEXT:
- Total records: {dataset_info['total_rows']:,}
- Analysis results: {result_count:,} records
- Data includes: {', '.join(dataset_info['columns'][:10])}{'...' if len(dataset_info['columns']) > 10 else ''}

SQL EXECUTED: {sql_query}
SAMPLE RESULTS (first 10 of {result_count}): {results_json}

Provide a business-focused analysis:

1. **SALES OVERVIEW**: "I analyzed {result_count:,} sales records and found..."

2. **KEY METRICS & TRENDS**:
   - Revenue totals, averages, or patterns
   - Top performing products, customers, or regions
   - Time-based trends (monthly, quarterly, etc.)
   - Notable achievements or concerning drops

3. **BUSINESS INSIGHTS**:
   - What the data reveals about sales performance
   - Opportunities for growth or improvement
   - Seasonal patterns or market trends
   - Customer behavior insights

4. **RECOMMENDATIONS**:
   - Strategic actions based on the findings
   - Areas to focus on or investigate further
   - Potential risks or opportunities

Be business-oriented, highlight key metrics, and provide actionable insights.

Response:
"""

    def _create_hr_prompt(self, question, results_json, result_count, dataset_info, sql_query):
        """Create HR-specific conversational prompt"""
        return f"""
You are an HR data analyst assistant. The user asked: "{question}"

HR DATASET CONTEXT:
- Total employee records: {dataset_info['total_rows']:,}
- Analysis results: {result_count:,} records
- Data includes: {', '.join(dataset_info['columns'][:10])}{'...' if len(dataset_info['columns']) > 10 else ''}

SQL EXECUTED: {sql_query}
SAMPLE RESULTS (first 10 of {result_count}): {results_json}

Provide an HR-focused analysis:

1. **WORKFORCE OVERVIEW**: "I found {result_count:,} employee records that show..."

2. **KEY HR METRICS**:
   - Department distributions and team sizes
   - Salary ranges and compensation patterns
   - Experience levels and tenure
   - Performance ratings or reviews

3. **PEOPLE INSIGHTS**:
   - Workforce composition and diversity
   - Career progression patterns
   - Compensation equity analysis
   - Skill gaps or strengths

4. **HR RECOMMENDATIONS**:
   - Talent management opportunities
   - Compensation review suggestions
   - Training or development needs
   - Retention strategies

Be professional, focus on people analytics, and ensure privacy considerations.

Response:
"""

    def _create_finance_prompt(self, question, results_json, result_count, dataset_info, sql_query):
        """Create finance-specific conversational prompt"""
        return f"""
You are a financial data analyst assistant. The user asked: "{question}"

FINANCIAL DATASET CONTEXT:
- Total transactions/records: {dataset_info['total_rows']:,}
- Analysis results: {result_count:,} records
- Data includes: {', '.join(dataset_info['columns'][:10])}{'...' if len(dataset_info['columns']) > 10 else ''}

SQL EXECUTED: {sql_query}
SAMPLE RESULTS (first 10 of {result_count}): {results_json}

Provide a financial analysis:

1. **FINANCIAL OVERVIEW**: "I analyzed {result_count:,} financial records and discovered..."

2. **KEY FINANCIAL METRICS**:
   - Transaction volumes and amounts
   - Account balances and movements
   - Spending patterns and categories
   - Revenue or expense trends

3. **FINANCIAL INSIGHTS**:
   - Cash flow patterns
   - Budget performance vs actuals
   - Cost center analysis
   - Financial health indicators

4. **FINANCIAL RECOMMENDATIONS**:
   - Cost optimization opportunities
   - Revenue enhancement strategies
   - Risk management suggestions
   - Financial planning insights

Be analytical, focus on financial KPIs, and provide strategic financial insights.

Response:
"""

    def _create_general_prompt(self, question, results_json, result_count, dataset_info, sql_query):
        """Create general conversational prompt"""
        return f"""
You are a helpful data analyst assistant. The user asked: "{question}"

DATASET CONTEXT:
- Total records: {dataset_info['total_rows']:,}
- Analysis results: {result_count:,} records
- Columns: {len(dataset_info['columns'])} total
- Key fields: {', '.join(dataset_info['columns'][:8])}{'...' if len(dataset_info['columns']) > 8 else ''}

SQL EXECUTED: {sql_query}
SAMPLE RESULTS (first 10 of {result_count}): {results_json}

Provide a conversational, insightful analysis:

1. **DATA OVERVIEW**: "I found {result_count:,} records in your dataset that..."

2. **KEY FINDINGS**:
   - Main patterns or trends in the data
   - Notable values, outliers, or interesting observations
   - Relationships between different data fields
   - Summary statistics where relevant

3. **INSIGHTS & ANALYSIS**:
   - What the data reveals about the underlying patterns
   - Correlations or connections you can identify
   - Trends over time if date fields are present
   - Groupings or categories that emerge

4. **SUGGESTIONS**:
   - Follow-up questions that might be interesting
   - Other ways to analyze or slice the data
   - Potential areas for deeper investigation

Be conversational, curious, and help the user understand their data in a meaningful way.

Response:
"""

    def _call_gemini_api(self, prompt):
        """Call Gemini API with proper error handling for rate limits"""
        headers = {'Content-Type': 'application/json'}
        
        data = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'temperature': 0.3,
                'topP': 0.8,
                'topK': 40,
                'maxOutputTokens': 2500,
            }
        }
        
        try:
            response = requests.post(
                f"{self.gemini_url}?key={self.gemini_api_key}",
                headers=headers,
                json=data,
                timeout=30,
                verify=False
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    raise Exception("No response from Gemini")
            elif response.status_code == 429:
                # Rate limit exceeded
                try:
                    error_data = response.json()
                    error_message = error_data.get('error', {}).get('message', 'Rate limit exceeded')
                    if 'RESOURCE_EXHAUSTED' in error_message:
                        raise Exception("RATE_LIMIT_EXCEEDED: You've exceeded the Gemini API rate limit. Please wait a moment or update your API key with a higher quota.")
                    else:
                        raise Exception(f"RATE_LIMIT_EXCEEDED: {error_message}")
                except:
                    raise Exception("RATE_LIMIT_EXCEEDED: Gemini API rate limit exceeded. Please try again later or upgrade your API quota.")
            else:
                # Other API errors
                try:
                    error_data = response.json()
                    error_message = error_data.get('error', {}).get('message', f'HTTP {response.status_code}')
                    raise Exception(f"GEMINI_API_ERROR: {error_message}")
                except:
                    raise Exception(f"GEMINI_API_ERROR: HTTP {response.status_code} - {response.text[:200]}")
                
        except requests.exceptions.Timeout:
            raise Exception("GEMINI_API_TIMEOUT: Request to Gemini API timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            raise Exception("GEMINI_API_CONNECTION: Cannot connect to Gemini API. Please check your internet connection.")
        except Exception as e:
            if str(e).startswith(('RATE_LIMIT_EXCEEDED:', 'GEMINI_API_ERROR:', 'GEMINI_API_TIMEOUT:', 'GEMINI_API_CONNECTION:')):
                raise e
            else:
                print(f"Gemini API error: {e}")
                raise Exception(f"GEMINI_API_ERROR: {str(e)}")

class SimilarityAnalyzer:
    """Advanced similarity detection for any dataset"""
    
    def __init__(self):
        self.vectorizer = None
        self.embeddings_cache = {}
    
    def analyze_dataset_similarity(self, dataset_info, chat_id, analysis_type='comprehensive', 
                                 threshold=0.7, search_query='', categories=None):
        """Main similarity analysis function"""
        try:
            sqlite_path = dataset_info['sqlite_path']
            
            # Get sample data for analysis
            sample_data = self._get_sample_data(sqlite_path, search_query, limit=1000)
            
            if not sample_data:
                return {'success': False, 'error': 'No data to analyze'}
            
            results = {}
            
            if analysis_type in ['comprehensive', 'duplicates']:
                results['duplicates'] = self._find_duplicates(sample_data, threshold)
            
            if analysis_type in ['comprehensive', 'semantic']:
                results['semantic_groups'] = self._find_semantic_similarity(sample_data, threshold)
            
            if analysis_type in ['comprehensive', 'categorical']:
                results['categories'] = self._smart_categorization(sample_data)
            
            # Generate analysis summary
            summary = self._generate_analysis_summary(results, analysis_type)
            
            # Prepare results for display
            display_data = self._prepare_display_data(results, sample_data)
            
            return {
                'success': True,
                'results': {
                    'data': display_data,
                    'summary': summary,
                    'total_matches': len(display_data),
                    'analysis_type': analysis_type,
                    'threshold': threshold
                }
            }
            
        except Exception as e:
            print(f"Similarity analysis error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_sample_data(self, sqlite_path, search_query='', limit=1000):
        """Get sample data from SQLite database"""
        try:
            conn = sqlite3.connect(sqlite_path)
            
            if search_query:
                # Dynamic search across all text columns
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(data)")
                columns = [row[1] for row in cursor.fetchall()]
                
                where_conditions = []
                for col in columns:
                    where_conditions.append(f'"{col}" LIKE ?')
                
                where_clause = ' OR '.join(where_conditions)
                search_pattern = f'%{search_query}%'
                search_params = [search_pattern] * len(columns)
                
                query = f'SELECT * FROM data WHERE {where_clause} LIMIT {limit}'
                cursor.execute(query, search_params)
            else:
                query = f'SELECT * FROM data LIMIT {limit}'
                cursor.execute(query)
            
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
            data = []
            for row in rows:
                record = {}
                for i, col in enumerate(columns):
                    record[col] = row[i]
                data.append(record)
            
            conn.close()
            return data
            
        except Exception as e:
            print(f"Error getting sample data: {e}")
            return []
    
    def _find_duplicates(self, data, threshold=0.8):
        """Find exact and near-duplicate records"""
        duplicates = []
        exact_duplicates = defaultdict(list)
        near_duplicates = []
        
        # Find exact duplicates
        for i, record in enumerate(data):
            # Create a signature from all text fields
            text_fields = []
            for key, value in record.items():
                if isinstance(value, str) and len(value.strip()) > 0:
                    text_fields.append(value.strip().lower())
            
            signature = '|'.join(sorted(text_fields))
            exact_duplicates[signature].append({'index': i, 'record': record})
        
        # Process exact duplicates
        for signature, records in exact_duplicates.items():
            if len(records) > 1:
                duplicates.append({
                    'type': 'exact',
                    'similarity_score': 1.0,
                    'records': records,
                    'count': len(records)
                })
        
        # Find near-duplicates using text similarity
        processed_indices = set()
        for i, record1 in enumerate(data):
            if i in processed_indices:
                continue
                
            text1 = self._extract_text_content(record1)
            if not text1:
                continue
                
            similar_records = [{'index': i, 'record': record1, 'similarity': 1.0}]
            
            for j, record2 in enumerate(data[i+1:], i+1):
                if j in processed_indices:
                    continue
                    
                text2 = self._extract_text_content(record2)
                if not text2:
                    continue
                
                similarity = self._calculate_text_similarity(text1, text2)
                
                if similarity >= threshold:
                    similar_records.append({
                        'index': j, 
                        'record': record2, 
                        'similarity': similarity
                    })
                    processed_indices.add(j)
            
            if len(similar_records) > 1:
                processed_indices.add(i)
                near_duplicates.append({
                    'type': 'near_duplicate',
                    'similarity_score': max(r['similarity'] for r in similar_records[1:]),
                    'records': similar_records,
                    'count': len(similar_records)
                })
        
        return {
            'exact_duplicates': [d for d in duplicates if d['type'] == 'exact'],
            'near_duplicates': near_duplicates,
            'total_duplicate_groups': len(duplicates) + len(near_duplicates)
        }
    
    def _find_semantic_similarity(self, data, threshold=0.7):
        """Find semantically similar records using TF-IDF and cosine similarity"""
        try:
            # Extract text content from all records
            texts = []
            valid_indices = []
            
            for i, record in enumerate(data):
                text = self._extract_text_content(record)
                if text and len(text.strip()) > 5:  # Only non-empty texts
                    texts.append(text)
                    valid_indices.append(i)
            
            if len(texts) < 2:
                return {'groups': [], 'total_groups': 0}
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find similarity groups
            groups = []
            processed_indices = set()
            
            for i in range(len(texts)):
                if i in processed_indices:
                    continue
                
                # Find all records similar to current record
                similar_indices = []
                for j in range(len(texts)):
                    if i != j and similarity_matrix[i][j] >= threshold:
                        similar_indices.append(j)
                
                if similar_indices:
                    # Create group
                    group_records = [{'index': valid_indices[i], 'record': data[valid_indices[i]], 'similarity': 1.0}]
                    
                    for j in similar_indices:
                        if j not in processed_indices:
                            group_records.append({
                                'index': valid_indices[j],
                                'record': data[valid_indices[j]],
                                'similarity': similarity_matrix[i][j]
                            })
                            processed_indices.add(j)
                    
                    groups.append({
                        'type': 'semantic_group',
                        'avg_similarity': np.mean([r['similarity'] for r in group_records]),
                        'records': group_records,
                        'count': len(group_records),
                        'keywords': self._extract_group_keywords(vectorizer, tfidf_matrix, [i] + similar_indices)
                    })
                    
                    processed_indices.add(i)
            
            return {
                'groups': groups,
                'total_groups': len(groups)
            }
            
        except Exception as e:
            print(f"Semantic similarity error: {e}")
            return {'groups': [], 'total_groups': 0}
    
    def _smart_categorization(self, data):
        """Automatic categorization and pattern recognition"""
        categories = defaultdict(list)
        patterns = defaultdict(int)
        
        # Analyze patterns in the data
        for i, record in enumerate(data):
            # Extract categorizable features
            text_content = self._extract_text_content(record)
            
            # Simple keyword-based categorization
            keywords = self._extract_keywords(text_content)
            
            # Categorize by dominant keywords
            if keywords:
                primary_category = max(keywords.items(), key=lambda x: x[1])[0]
                categories[primary_category].append({
                    'index': i,
                    'record': record,
                    'keywords': keywords
                })
            
            # Pattern detection (e.g., email patterns, ID patterns, etc.)
            detected_patterns = self._detect_patterns(record)
            for pattern in detected_patterns:
                patterns[pattern] += 1
        
        # Priority scoring
        priority_scores = {}
        for category, records in categories.items():
            # Score based on frequency and keyword strength
            score = len(records) * sum(
                max(r['keywords'].values()) for r in records if r['keywords']
            ) / len(records) if records else 0
            priority_scores[category] = score
        
        return {
            'categories': dict(categories),
            'patterns': dict(patterns),
            'priority_scores': priority_scores,
            'total_categories': len(categories)
        }
    
    def _extract_text_content(self, record):
        """Extract all text content from a record"""
        text_parts = []
        for key, value in record.items():
            if isinstance(value, str) and len(value.strip()) > 0:
                # Clean and normalize text
                cleaned = re.sub(r'[^\w\s]', ' ', str(value))
                cleaned = ' '.join(cleaned.split())
                text_parts.append(cleaned)
        return ' '.join(text_parts)
    
    def _calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two texts using multiple methods"""
        # Method 1: Sequence matching
        seq_similarity = difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        
        # Method 2: Word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return seq_similarity
        
        intersection = words1 & words2
        union = words1 | words2
        jaccard_similarity = len(intersection) / len(union) if union else 0
        
        # Combined score
        return (seq_similarity + jaccard_similarity) / 2
    
    def _extract_keywords(self, text, top_k=5):
        """Extract important keywords from text"""
        if not text:
            return {}
        
        # Simple frequency-based keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        words = [w for w in words if len(w) > 3]  # Filter short words
        
        # Common stop words to exclude
        stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'have', 'were', 'said', 'each', 'which', 'their', 'time', 'would', 'about', 'there', 'could', 'other', 'more', 'very', 'what', 'know', 'just', 'first', 'into', 'over', 'think', 'also', 'your', 'work', 'life', 'only', 'can', 'still', 'should', 'after', 'being', 'now', 'made', 'before', 'here', 'through', 'when', 'where', 'much', 'some', 'these', 'many', 'then', 'them', 'well', 'were'}
        
        words = [w for w in words if w not in stop_words]
        
        word_counts = Counter(words)
        return dict(word_counts.most_common(top_k))
    
    def _detect_patterns(self, record):
        """Detect common patterns in record fields"""
        patterns = []
        
        for key, value in record.items():
            if not isinstance(value, str):
                continue
                
            value_str = str(value).strip()
            
            # Email pattern
            if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value_str):
                patterns.append('email')
            
            # Phone pattern
            elif re.match(r'^[\+]?[1-9]?[0-9]{7,15}$', re.sub(r'[^\d\+]', '', value_str)):
                patterns.append('phone')
            
            # URL pattern
            elif re.match(r'^https?://', value_str):
                patterns.append('url')
            
            # ID pattern (alphanumeric)
            elif re.match(r'^[A-Z0-9]{5,}$', value_str):
                patterns.append('id_code')
            
            # Date pattern
            elif re.match(r'\d{4}-\d{2}-\d{2}', value_str) or re.match(r'\d{2}/\d{2}/\d{4}', value_str):
                patterns.append('date')
            
            # Number pattern
            elif re.match(r'^\d+\.?\d*$', value_str):
                patterns.append('numeric')
        
        return patterns
    
    def _extract_group_keywords(self, vectorizer, tfidf_matrix, indices):
        """Extract keywords that define a similarity group"""
        try:
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Average TF-IDF scores for the group
            group_tfidf = tfidf_matrix[indices].mean(axis=0).A1
            
            # Get top keywords
            top_indices = group_tfidf.argsort()[-10:][::-1]
            keywords = [(feature_names[i], group_tfidf[i]) for i in top_indices if group_tfidf[i] > 0]
            
            return keywords[:5]  # Return top 5 keywords
            
        except Exception as e:
            print(f"Error extracting group keywords: {e}")
            return []
    
    def _generate_analysis_summary(self, results, analysis_type):
        """Generate human-readable analysis summary"""
        summary_parts = []
        
        if 'duplicates' in results:
            dup_data = results['duplicates']
            exact_count = len(dup_data['exact_duplicates'])
            near_count = len(dup_data['near_duplicates'])
            
            if exact_count > 0:
                summary_parts.append(f"Found {exact_count} exact duplicate groups")
            if near_count > 0:
                summary_parts.append(f"Found {near_count} near-duplicate groups")
        
        if 'semantic_groups' in results:
            semantic_data = results['semantic_groups']
            group_count = semantic_data['total_groups']
            if group_count > 0:
                summary_parts.append(f"Identified {group_count} semantic similarity groups")
        
        if 'categories' in results:
            cat_data = results['categories']
            cat_count = cat_data['total_categories']
            if cat_count > 0:
                summary_parts.append(f"Automatically categorized into {cat_count} groups")
                
                # Top patterns
                patterns = cat_data['patterns']
                if patterns:
                    top_pattern = max(patterns.items(), key=lambda x: x[1])
                    summary_parts.append(f"Most common pattern: {top_pattern[0]} ({top_pattern[1]} occurrences)")
        
        if not summary_parts:
            summary_parts.append("No significant similarity patterns detected")
        
        return "\n".join(summary_parts)
    
    def _prepare_display_data(self, results, original_data):
        """Prepare similarity results for table display"""
        display_records = []
        
        # Process duplicates
        if 'duplicates' in results:
            for dup_group in results['duplicates']['exact_duplicates']:
                for i, record_info in enumerate(dup_group['records']):
                    display_record = record_info['record'].copy()
                    display_record['_similarity_type'] = 'Exact Duplicate'
                    display_record['_similarity_score'] = '100%'
                    display_record['_group_id'] = f"exact_{len(display_records) // 10}"
                    display_record['_group_size'] = dup_group['count']
                    display_records.append(display_record)
            
            for dup_group in results['duplicates']['near_duplicates']:
                for i, record_info in enumerate(dup_group['records']):
                    display_record = record_info['record'].copy()
                    display_record['_similarity_type'] = 'Near Duplicate'
                    display_record['_similarity_score'] = f"{record_info['similarity']:.1%}"
                    display_record['_group_id'] = f"near_{len(display_records) // 10}"
                    display_record['_group_size'] = dup_group['count']
                    display_records.append(display_record)
        
        # Process semantic groups
        if 'semantic_groups' in results:
            for group in results['semantic_groups']['groups']:
                for record_info in group['records']:
                    display_record = record_info['record'].copy()
                    display_record['_similarity_type'] = 'Semantic Match'
                    display_record['_similarity_score'] = f"{record_info['similarity']:.1%}"
                    display_record['_group_id'] = f"semantic_{len(display_records) // 10}"
                    display_record['_group_size'] = group['count']
                    if group['keywords']:
                        display_record['_keywords'] = ', '.join([kw[0] for kw in group['keywords'][:3]])
                    display_records.append(display_record)
        
        # Process categories
        if 'categories' in results:
            for category, records in results['categories']['categories'].items():
                for record_info in records:
                    display_record = record_info['record'].copy()
                    display_record['_similarity_type'] = 'Category Match'
                    display_record['_category'] = category
                    display_record['_group_size'] = len(records)
                    display_records.append(display_record)
        
        return display_records

class ChatMCPProcessor:
    """Enhanced Chat-specific MCP processor with conversational AI"""
    def __init__(self):
        # Store chat-specific datasets: {chat_id: dataset_info}
        self.chat_datasets = {}
        self.conversational_ai = ConversationalAI()
        
    def load_dataset_for_chat(self, file_path, chat_id):
        """Load dataset specifically for a chat session"""
        try:
            print(f"Loading dataset for chat {chat_id} from: {file_path}")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_size_info = get_file_size_formatted(file_path)
            file_extension = Path(file_path).suffix.lower()
            file_name = Path(file_path).name
            
            print(f"Processing file: {file_name} ({file_size_info['size_formatted']})")
            
            # Detect actual file format
            actual_format = detect_actual_file_format(file_path)
            print(f"Detected file format: {actual_format}")
            
            sample_df = None
            total_rows = 0
            
            if actual_format == 'csv_like' or file_extension == '.csv':
                print("Reading as CSV...")
                sample_df = pd.read_csv(file_path, nrows=100, low_memory=False)
                
                # Count total rows efficiently
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    total_rows = sum(1 for line in f) - 1
                    
            elif actual_format == 'html_xml':
                print("Reading as HTML/XML...")
                full_df = read_html_xml_as_dataframe(file_path)
                sample_df = full_df.head(100)
                total_rows = len(full_df)
                # Keep full_df for later processing
                
            elif actual_format in ['xlsx', 'xls'] or file_extension in ['.xlsx', '.xls']:
                print("Reading as Excel...")
                # Enhanced Excel reading with multiple engine attempts
                engines_to_try = []
                if file_extension == '.xlsx' or actual_format == 'xlsx':
                    engines_to_try = ['openpyxl', 'xlrd']
                else:  # .xls
                    engines_to_try = ['xlrd', 'openpyxl']
                
                # Add calamine as a backup engine if available
                try:
                    import python_calamine
                    engines_to_try.append('calamine')
                except ImportError:
                    pass
                
                last_error = None
                
                for engine in engines_to_try:
                    try:
                        print(f"Trying to read Excel file with engine: {engine}")
                        
                        if engine == 'xlrd' and (file_extension == '.xlsx' or actual_format == 'xlsx'):
                            # xlrd doesn't support .xlsx files in newer versions
                            continue
                            
                        sample_df = pd.read_excel(file_path, nrows=100, engine=engine)
                        print(f"Successfully read sample with engine: {engine}")
                        
                        # Count total rows
                        try:
                            if engine == 'openpyxl' and (file_extension == '.xlsx' or actual_format == 'xlsx'):
                                import openpyxl
                                wb = openpyxl.load_workbook(file_path, read_only=True)
                                ws = wb.active
                                total_rows = ws.max_row - 1  # Subtract header
                                wb.close()
                            else:
                                # Fallback: load entire file to count rows
                                print("Using fallback method to count rows...")
                                full_df = pd.read_excel(file_path, engine=engine)
                                total_rows = len(full_df)
                                del full_df  # Free memory immediately
                        except Exception as count_error:
                            print(f"Error counting rows with {engine}, using fallback: {count_error}")
                            total_rows = len(sample_df)
                        
                        break  # Success, exit the loop
                        
                    except Exception as e:
                        last_error = e
                        print(f"Engine {engine} failed: {str(e)}")
                        continue
                
                if sample_df is None:
                    # If all Excel engines failed, try as HTML/XML
                    print("All Excel engines failed, trying as HTML/XML...")
                    try:
                        full_df = read_html_xml_as_dataframe(file_path)
                        sample_df = full_df.head(100)
                        total_rows = len(full_df)
                    except Exception as xml_error:
                        raise Exception(f"Could not read file with any method. Excel error: {last_error}, XML error: {xml_error}")
                        
            else:
                raise ValueError(f"Unsupported file format: {file_extension} (detected as {actual_format})")
            
            if sample_df is None or len(sample_df) == 0:
                raise Exception("No data could be extracted from the file")
            
            print(f"Dataset has {total_rows:,} rows and {len(sample_df.columns)} columns")
            
            # Clean column names and handle duplicates
            original_columns = list(sample_df.columns)
            clean_columns = []
            for i, col in enumerate(original_columns):
                clean_col = str(col).strip()
                if not clean_col or clean_col in clean_columns:
                    clean_col = f'Column_{i+1}'
                clean_columns.append(clean_col)
            
            sample_df.columns = clean_columns
            
            # Convert datetime columns to strings for JSON compatibility
            sample_df_json_safe = sample_df.copy()
            for col in sample_df_json_safe.columns:
                if pd.api.types.is_datetime64_any_dtype(sample_df_json_safe[col]):
                    sample_df_json_safe.loc[:, col] = sample_df_json_safe[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Store dataset info for this specific chat
            dataset_info = {
                'file_path': file_path,
                'file_name': file_name,
                'total_rows': int(total_rows),
                'total_columns': int(len(sample_df.columns)),
                'columns': clean_columns,
                'original_columns': original_columns,
                'dtypes': {col: str(dtype) for col, dtype in sample_df.dtypes.items()},
                'sample_data': convert_dataframe_for_json(sample_df_json_safe.head(5)),
                'file_size_mb': file_size_info['size_mb'],
                'file_size_formatted': file_size_info['size_formatted'],
                'file_size_bytes': file_size_info['size_bytes'],
                'cleaned_columns': [self._clean_column_name(col) for col in clean_columns],
                'actual_format': actual_format
            }
            
            # Analyze dataset context for conversational AI
            dataset_context = self.conversational_ai.analyze_dataset_context(dataset_info)
            dataset_info['context'] = dataset_context
            
            # Create chat-specific SQLite database
            sqlite_path = file_path.replace(file_extension, f'_chat_{chat_id}.db')
            dataset_info['sqlite_path'] = sqlite_path
            
            success = self._create_sqlite_db(file_path, sqlite_path, dataset_info)
            
            if success:
                # Store dataset info for this chat
                self.chat_datasets[chat_id] = dataset_info
                print(f"Dataset loaded successfully for chat {chat_id}: {total_rows:,} rows, {len(sample_df.columns)} columns")
                print(f"Dataset type detected: {dataset_context['type']} (confidence: {dataset_context['confidence']:.1%})")
                
                # Update chat database with dataset info
                self._save_chat_dataset_info(chat_id, dataset_info)
                
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Error loading dataset for chat {chat_id}: {e}")
            traceback.print_exc()
            return False
    
    def _clean_column_name(self, col_name):
        """Clean column names for SQLite compatibility"""
        import re
        cleaned = re.sub(r'[^\w]', '_', str(col_name))
        if cleaned and cleaned[0].isdigit():
            cleaned = 'col_' + cleaned
        return cleaned or 'unnamed_col'
    
    def _save_chat_dataset_info(self, chat_id, dataset_info):
        """Save dataset info to chat database"""
        try:
            conn = sqlite3.connect('data/chat_history.db')
            cursor = conn.cursor()
            
            # Create dataset table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_datasets (
                    chat_id TEXT PRIMARY KEY,
                    dataset_info TEXT,
                    file_path TEXT,
                    sqlite_path TEXT,
                    created_at TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES chats (id)
                )
            ''')
            
            # Save dataset info
            cursor.execute('''
                INSERT OR REPLACE INTO chat_datasets 
                (chat_id, dataset_info, file_path, sqlite_path, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                chat_id, 
                safe_json_dumps(dataset_info),
                dataset_info['file_path'],
                dataset_info['sqlite_path'],
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error saving chat dataset info: {e}")
    
    def load_chat_dataset(self, chat_id):
        """Load dataset info for a specific chat"""
        try:
            # First check in memory
            if chat_id in self.chat_datasets:
                dataset_info = self.chat_datasets[chat_id]
                # Verify files still exist
                if (os.path.exists(dataset_info['file_path']) and 
                    os.path.exists(dataset_info['sqlite_path'])):
                    return True
                else:
                    # Files missing, remove from memory
                    del self.chat_datasets[chat_id]
            
            # Try to load from database
            conn = sqlite3.connect('data/chat_history.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT dataset_info, file_path, sqlite_path 
                FROM chat_datasets 
                WHERE chat_id = ?
            ''', (chat_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                dataset_info = json.loads(row[0])
                file_path = row[1]
                sqlite_path = row[2]
                
                # Verify files still exist
                if os.path.exists(file_path) and os.path.exists(sqlite_path):
                    dataset_info['file_path'] = file_path
                    dataset_info['sqlite_path'] = sqlite_path
                    self.chat_datasets[chat_id] = dataset_info
                    return True
                else:
                    # Files missing, clean up database
                    self._cleanup_chat_dataset(chat_id)
            
            return False
            
        except Exception as e:
            print(f"Error loading chat dataset: {e}")
            return False
    
    def _cleanup_chat_dataset(self, chat_id):
        """Clean up missing dataset files from database"""
        try:
            conn = sqlite3.connect('data/chat_history.db')
            cursor = conn.cursor()
            cursor.execute('DELETE FROM chat_datasets WHERE chat_id = ?', (chat_id,))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error cleaning up chat dataset: {e}")
   
    def get_chat_dataset_info(self, chat_id):
       """Get dataset info for a specific chat"""
       if chat_id in self.chat_datasets:
           return self.chat_datasets[chat_id]
       return None
   
    def has_dataset_for_chat(self, chat_id):
       """Check if chat has a dataset loaded"""
       return chat_id in self.chat_datasets
   
    def _create_sqlite_db(self, file_path, sqlite_path, dataset_info):
       """Create SQLite database from file with robust handling"""
       try:
           print(f"Creating SQLite database: {sqlite_path}")
           
           file_extension = Path(file_path).suffix.lower()
           actual_format = dataset_info.get('actual_format', file_extension[1:])
           
           conn = sqlite3.connect(sqlite_path)
           
           # Increase SQLite limits for large datasets
           conn.execute("PRAGMA temp_store = MEMORY")
           conn.execute("PRAGMA mmap_size = 268435456")
           conn.execute("PRAGMA page_size = 65536")
           conn.execute("PRAGMA cache_size = 10000")
           conn.execute("PRAGMA synchronous = OFF")
           conn.execute("PRAGMA journal_mode = MEMORY")
           
           # Determine appropriate chunk size
           num_columns = len(dataset_info['columns'])
           
           if num_columns > 100:
               chunk_size = max(1, int(800 / num_columns))
           elif num_columns > 50:
               chunk_size = max(10, int(900 / num_columns))
           else:
               chunk_size = 1000
               
           print(f"Using chunk size: {chunk_size} rows (for {num_columns} columns)")
           
           processed_chunks = 0
           total_processed = 0
           
           if actual_format == 'csv_like' or file_extension == '.csv':
               chunk_iterator = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)
               
               for chunk in chunk_iterator:
                   # Ensure columns match
                   chunk.columns = dataset_info['columns']
                   success = self._process_chunk_to_sqlite(chunk, conn, processed_chunks == 0, dataset_info)
                   if not success:
                       break
                       
                   processed_chunks += 1
                   total_processed += len(chunk)
                   
                   if processed_chunks % 10 == 0:
                       print(f"Processed {total_processed:,} rows...")
                       
           elif actual_format == 'html_xml':
               # Read the full HTML/XML file
               full_df = read_html_xml_as_dataframe(file_path)
               full_df.columns = dataset_info['columns']
               
               # Process in chunks
               for i in range(0, len(full_df), chunk_size):
                   chunk = full_df.iloc[i:i+chunk_size]
                   success = self._process_chunk_to_sqlite(chunk, conn, processed_chunks == 0, dataset_info)
                   if not success:
                       break
                   
                   processed_chunks += 1
                   total_processed += len(chunk)
                   
                   if processed_chunks % 10 == 0:
                       print(f"Processed {total_processed:,} rows...")
                       
           else:
               # Enhanced Excel processing with engine detection
               print("Processing Excel file in batches...")
               
               # Determine which engine worked for this file
               engines_to_try = ['openpyxl', 'xlrd']
               if file_extension == '.xls':
                   engines_to_try = ['xlrd', 'openpyxl']
               
               # Add calamine if available
               try:
                   import python_calamine
                   engines_to_try.append('calamine')
               except ImportError:
                   pass
               
               working_engine = None
               for engine in engines_to_try:
                   try:
                       # Test the engine
                       test_df = pd.read_excel(file_path, nrows=1, engine=engine)
                       working_engine = engine
                       print(f"Using engine {engine} for batch processing")
                       break
                   except:
                       continue
               
               if not working_engine:
                   raise Exception("No working Excel engine found for batch processing")
               
               # Get total rows for progress tracking
               total_rows = dataset_info['total_rows'] + 1  # Add header
               start_row = 0
               
               while start_row < total_rows - 1:
                   try:
                       if start_row == 0:
                           chunk = pd.read_excel(file_path, skiprows=0, nrows=chunk_size, engine=working_engine)
                       else:
                           chunk = pd.read_excel(file_path, skiprows=start_row + 1, nrows=chunk_size, 
                                               header=None, engine=working_engine)
                           chunk.columns = dataset_info['columns']
                       
                       if len(chunk) == 0:
                           break
                           
                       success = self._process_chunk_to_sqlite(chunk, conn, processed_chunks == 0, dataset_info)
                       if not success:
                           break
                       
                       processed_chunks += 1
                       total_processed += len(chunk)
                       start_row += chunk_size
                       
                       print(f"Processed {total_processed:,} of {total_rows-1:,} rows...")
                       
                   except Exception as e:
                       print(f"Error processing Excel batch starting at row {start_row}: {e}")
                       break
           
           # Create indexes
           print("Creating database indexes...")
           cursor = conn.cursor()
           
           for i, col in enumerate(dataset_info['cleaned_columns'][:3]):
               try:
                   cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{col} ON data({col})")
               except Exception as e:
                   print(f"Could not create index on {col}: {e}")
           
           conn.commit()
           conn.close()
           
           print(f"SQLite database created successfully: {sqlite_path}")
           return True
           
       except Exception as e:
           print(f"Error creating SQLite database: {e}")
           traceback.print_exc()
           return False
   
    def _process_chunk_to_sqlite(self, chunk, conn, is_first_chunk, dataset_info):
       """Process a single chunk to SQLite"""
       try:
           # Clean column names
           chunk.columns = [self._clean_column_name(col) for col in chunk.columns]
           chunk = self._prepare_chunk_for_sqlite(chunk)
           
           num_columns = len(chunk.columns)
           
           if num_columns > 100:
               return self._insert_chunk_individually(chunk, conn, is_first_chunk)
           else:
               batch_size = min(100, max(1, int(800 / num_columns)))
               
               for i in range(0, len(chunk), batch_size):
                   batch = chunk.iloc[i:i+batch_size]
                   batch.to_sql('data', conn, 
                              if_exists='append' if not (is_first_chunk and i == 0) else 'replace',
                              index=False, 
                              method=None)
               
               return True
               
       except Exception as e:
           print(f"Error processing chunk: {e}")
           return False
   
    def _insert_chunk_individually(self, chunk, conn, is_first_chunk):
       """Insert chunk using individual INSERT statements"""
       try:
           cursor = conn.cursor()
           
           if is_first_chunk:
               cursor.execute("DROP TABLE IF EXISTS data")
               columns_def = ', '.join([f'"{col}" TEXT' for col in chunk.columns])
               create_sql = f"CREATE TABLE data ({columns_def})"
               cursor.execute(create_sql)
           
           placeholders = ', '.join(['?' for _ in chunk.columns])
           column_names = ', '.join([f'"{col}"' for col in chunk.columns])
           insert_sql = f"INSERT INTO data ({column_names}) VALUES ({placeholders})"
           
           for _, row in chunk.iterrows():
               values = [str(val) if val is not None else '' for val in row.values]
               cursor.execute(insert_sql, values)
           
           conn.commit()
           return True
           
       except Exception as e:
           print(f"Error in individual insert: {e}")
           return False
   
    def _prepare_chunk_for_sqlite(self, chunk):
       """Prepare chunk for SQLite"""
       chunk = chunk.copy()
       
       for col in chunk.columns:
           if pd.api.types.is_datetime64_any_dtype(chunk[col]):
               chunk.loc[:, col] = chunk[col].dt.strftime('%Y-%m-%d %H:%M:%S')
           elif chunk[col].dtype == 'object':
               chunk.loc[:, col] = chunk[col].apply(lambda x: 
                   x.isoformat() if isinstance(x, (datetime, date, pd.Timestamp)) 
                   else str(x) if x is not None else ''
               )
       
       chunk = chunk.fillna('')
       return chunk
   
    def query_data_for_chat(self, question, chat_id):
       """Process query for a specific chat's dataset with conversational AI"""
       try:
           if chat_id not in self.chat_datasets:
               return {'success': False, 'error': 'No dataset loaded for this chat'}
           
           dataset_info = self.chat_datasets[chat_id]
           dataset_context = dataset_info.get('context', {})
           
           print(f"Processing question for chat {chat_id}: {question}")
           print(f"Dataset type: {dataset_context.get('type', 'general')}")
           
           # Generate SQL query
           sql_query = self._generate_sql_query(question, dataset_info, dataset_context)
           if not sql_query:
               return {'success': False, 'error': 'Could not understand the question'}
           
           print(f"Generated SQL: {sql_query}")
           
           # Execute query
           results = self._execute_sql_query(sql_query, dataset_info['sqlite_path'])
           if results is None:
               return {'success': False, 'error': 'Query execution failed'}
           
           # Generate conversational response
           response = self.conversational_ai.generate_conversational_response(
               question, results, dataset_info, sql_query, dataset_context
           )
           
           # Prepare display results
           display_results = self._prepare_display_results(results, question)
           
           # Determine analysis type based on question and dataset
           analysis_type = self._determine_analysis_type(question, dataset_context)
           
           return {
               'success': True,
               'answer': response,
               'sql_query': sql_query,
               'results': display_results,
               'total_results': len(results),
               'analysis_type': analysis_type,
               'dataset_type': dataset_context.get('type', 'general'),
               'show_all_data': self._should_show_all_data(question, len(results))
           }
           
       except Exception as e:
           print(f"Error in query processing for chat {chat_id}: {e}")
           traceback.print_exc()
           return {'success': False, 'error': f'Analysis failed: {str(e)}'}
   
    def _determine_analysis_type(self, question, dataset_context):
       """Determine the type of analysis based on question and context"""
       question_lower = question.lower()
       dataset_type = dataset_context.get('type', 'general')
       
       # Similarity-related keywords
       similarity_keywords = ['similar', 'duplicate', 'matching', 'like', 'resembling']
       if any(keyword in question_lower for keyword in similarity_keywords):
           return 'similarity_analysis'
       
       # Trend analysis keywords
       trend_keywords = ['trend', 'over time', 'monthly', 'quarterly', 'yearly', 'growth']
       if any(keyword in question_lower for keyword in trend_keywords):
           return 'trend_analysis'
       
       # Summary keywords
       summary_keywords = ['summary', 'overview', 'total', 'count', 'average', 'sum']
       if any(keyword in question_lower for keyword in summary_keywords):
           return 'summary_analysis'
       
       # Dataset-specific analysis types
       if dataset_type == 'jira':
           if any(word in question_lower for word in ['bug', 'issue', 'ticket', 'priority']):
               return 'jira_analysis'
       elif dataset_type == 'sales':
           if any(word in question_lower for word in ['revenue', 'sales', 'customer', 'product']):
               return 'sales_analysis'
       elif dataset_type == 'hr':
           if any(word in question_lower for word in ['employee', 'salary', 'department', 'performance']):
               return 'hr_analysis'
       elif dataset_type == 'finance':
           if any(word in question_lower for word in ['transaction', 'balance', 'expense', 'budget']):
               return 'finance_analysis'
       
       return 'general_analysis'
   
    def _prepare_display_results(self, results, question):
       """Decide how many results to show based on question context"""
       question_lower = question.lower()
       
       show_all_keywords = [
           'all', 'complete', 'entire', 'full', 'every', 'total',
           'show me all', 'give me all', 'list all', 'all data',
           'complete data', 'entire dataset', 'everything'
       ]
       
       limit_keywords = [
           'top', 'first', 'sample', 'few', 'some', 'example'
       ]
       
       wants_all_data = any(keyword in question_lower for keyword in show_all_keywords)
       wants_limited = any(keyword in question_lower for keyword in limit_keywords)
       
       if wants_all_data and not wants_limited:
           return results
       elif len(results) <= 100:
           return results
       elif self._is_aggregated_query(results):
           return results
       else:
           return results[:50]
   
    def _is_aggregated_query(self, results):
       """Check if results are aggregated"""
       if not results:
           return False
       
       first_result = results[0]
       column_names = [col.lower() for col in first_result.keys()]
       
       aggregation_indicators = [
           'count', 'sum', 'avg', 'average', 'min', 'max', 
           'total', 'group', 'by'
       ]
       
       return any(indicator in ' '.join(column_names) for indicator in aggregation_indicators)
   
    def _should_show_all_data(self, question, result_count):
       """Determine if we should show all data in the UI"""
       question_lower = question.lower()
       
       show_all_keywords = [
           'all', 'complete', 'entire', 'full', 'every', 'total',
           'show me all', 'give me all', 'list all'
       ]
       
       return any(keyword in question_lower for keyword in show_all_keywords) or result_count <= 100
   
    def _generate_sql_query(self, question, dataset_info, dataset_context):
       """Generate SQL query with dataset context awareness"""
       try:
           dataset_type = dataset_context.get('type', 'general')
           
           schema_info = f"""
Table: data
Total rows: {dataset_info['total_rows']:,}
Columns: {len(dataset_info['columns'])}
Dataset Type: {dataset_type.upper()}

Column mapping (original -> cleaned):
"""
           for orig, cleaned in zip(dataset_info['columns'], dataset_info['cleaned_columns']):
               dtype = dataset_info['dtypes'].get(orig, 'text')
               schema_info += f"- '{orig}' -> {cleaned} ({dtype})\n"
           
           sample_data_str = safe_json_dumps(dataset_info['sample_data'][:3])
           schema_info += f"\nSample data:\n{sample_data_str}"
           
           # Add dataset-specific context
           if dataset_type == 'jira':
               context_hints = """
JIRA CONTEXT HINTS:
- Use "Status" for ticket status queries (Open, Closed, In Progress, etc.)
- Use "Priority" for priority filtering (Critical, High, Medium, Low)
- Use "Issue_Type" for bug/story/task filtering
- Use "Summary" and "Description" for text-based searches
- Use "Assignee" and "Reporter" for people queries
- Use "Created" and "Updated" for date-based queries
"""
           elif dataset_type == 'sales':
               context_hints = """
SALES CONTEXT HINTS:
- Look for revenue, amount, price columns for financial queries
- Use customer, product columns for segmentation
- Use date fields for trend analysis
- Calculate totals, averages for performance metrics
"""
           elif dataset_type == 'hr':
               context_hints = """
HR CONTEXT HINTS:
- Use salary, compensation columns for pay analysis
- Use department, team columns for organization queries
- Use performance, rating columns for evaluation analysis
- Use hire_date, tenure for employment timeline queries
"""
           else:
               context_hints = ""
           
           prompt = f"""
You are an expert SQL query generator with deep understanding of {dataset_type.upper()} data. 
Convert the natural language question to a SQLite query.

DATABASE SCHEMA:
{schema_info}

{context_hints}

IMPORTANT RULES:
1. Use the cleaned column names (right side of mapping) in your SQL
2. DO NOT add arbitrary LIMIT clauses unless the user specifically asks for "top N" or "first N"
3. If user asks for "all data" or "show me all" - return ALL data without LIMIT
4. Use appropriate aggregations (COUNT, SUM, AVG, etc.) when needed
5. Handle text searches with LIKE and % wildcards for similarity searches
6. For {dataset_type} queries, focus on domain-specific patterns and common use cases
7. Only add LIMIT if the user specifically requests a limited number of results
8. Return only the SQL query, no explanations
9. For context questions, use SELECT * FROM data to show all data
10. Always wrap column names in double quotes like "column_name" for SQLite compatibility
11. For similarity queries, use multiple LIKE conditions with OR to catch variations

USER QUESTION: {question}

Generate SQLite query:
"""

           response = self.conversational_ai._call_gemini_api(prompt)
           sql_query = self._extract_sql_from_response(response)
           
           return sql_query
           
       except Exception as e:
           print(f"Error generating SQL: {e}")
           return None
   
    def _extract_sql_from_response(self, response):
       """Extract clean SQL from Gemini response"""
       response = response.strip()
       
       if '```sql' in response:
           response = response.split('```sql')[1].split('```')[0]
       elif '```' in response:
           response = response.split('```')[1]
       
       response = response.strip().rstrip(';')
       
       if not response.upper().startswith('SELECT'):
           lines = response.split('\n')
           for line in lines:
               if line.strip().upper().startswith('SELECT'):
                   response = line.strip()
                   break
       
       return response
   
    def _execute_sql_query(self, sql_query, sqlite_path):
       """Execute SQL query on chat-specific database"""
       try:
           forbidden_words = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
           if any(word in sql_query.upper() for word in forbidden_words):
               raise ValueError("Query contains forbidden operations")
           
           conn = sqlite3.connect(sqlite_path)
           
           conn.execute("PRAGMA temp_store = MEMORY")
           conn.execute("PRAGMA cache_size = 10000")
           
           cursor = conn.cursor()
           cursor.execute(sql_query)
           
           columns = [desc[0] for desc in cursor.description]
           rows = cursor.fetchall()
           
           results = []
           for row in rows:
               result_dict = {}
               for i, col in enumerate(columns):
                   result_dict[col] = row[i]
               results.append(result_dict)
           
           conn.close()
           
           print(f"Query executed successfully: {len(results)} results returned")
           return results
           
       except Exception as e:
           print(f"Error executing SQL: {e}")
           return None

# Global instances
chat_processor = ChatMCPProcessor()
similarity_analyzer = SimilarityAnalyzer()

def init_db():
   """Initialize chat history database"""
   conn = sqlite3.connect('data/chat_history.db')
   cursor = conn.cursor()
   
   cursor.execute('''
       CREATE TABLE IF NOT EXISTS chats (
           id TEXT PRIMARY KEY,
           title TEXT,
           created_at TIMESTAMP,
           updated_at TIMESTAMP
       )
   ''')
   
   cursor.execute('''
       CREATE TABLE IF NOT EXISTS messages (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           chat_id TEXT,
           type TEXT,
           content TEXT,
           query TEXT,
           result TEXT,
           timestamp TIMESTAMP,
           FOREIGN KEY (chat_id) REFERENCES chats (id)
       )
   ''')
   
   cursor.execute('''
       CREATE TABLE IF NOT EXISTS chat_datasets (
           chat_id TEXT PRIMARY KEY,
           dataset_info TEXT,
           file_path TEXT,
           sqlite_path TEXT,
           created_at TIMESTAMP,
           FOREIGN KEY (chat_id) REFERENCES chats (id)
       )
   ''')
   
   conn.commit()
   conn.close()

init_db()
print("Enhanced conversational AI processor initialized successfully!")



@app.route('/')
def index():
    """API status endpoint - returns JSON instead of rendering template"""
    return jsonify({
        'status': 'success',
        'message': 'AI Sheet Chat API is running',
        'version': '2.0.0',
        'features': [
            'Conversational AI for any dataset',
            'Auto-detection of dataset types (JIRA, Sales, HR, Finance)',
            'Advanced similarity analysis',
            'Natural language query processing',
            'Multi-format file support (CSV, Excel, HTML, XML)',
            'Real-time data visualization',
            'Chat-based session management'
        ],
        'endpoints': {
            'upload': '/upload',
            'query': '/query',
            'chat_new': '/chat/new',
            'chat_status': '/chat/<chat_id>/status',
            'similarity_analysis': '/api/similarity-analysis',
            'chat_history': '/chat/history'
        },
        'supported_formats': ['CSV', 'XLSX', 'XLS', 'HTML', 'XML'],
        'max_file_size': '10GB',
        'ai_capabilities': {
            'jira_analysis': 'Bug similarity detection, ticket trends, priority analysis',
            'sales_analysis': 'Revenue trends, customer insights, product performance',
            'hr_analysis': 'Employee analytics, salary analysis, performance tracking',
            'finance_analysis': 'Transaction analysis, budget tracking, anomaly detection',
            'general_analysis': 'Pattern recognition, data summarization, trend analysis'
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime': 'Running'
    })



@app.route('/upload', methods=['POST'])
def upload_file():
   """Handle file upload for specific chat"""
   if 'file' not in request.files:
       return jsonify({'success': False, 'error': 'No file selected'})
   
   file = request.files['file']
   chat_id = request.form.get('chat_id')
   
   if file.filename == '':
       return jsonify({'success': False, 'error': 'No file selected'})
   
   if not chat_id:
       return jsonify({'success': False, 'error': 'Chat ID required'})
   
   # Validate file type
   allowed_extensions = {'.csv', '.xlsx', '.xls'}
   file_ext = Path(file.filename).suffix.lower()
   
   if file_ext not in allowed_extensions:
       return jsonify({'success': False, 'error': 'Only CSV and Excel files are supported'})
   
   try:
       filename = secure_filename(file.filename)
       # Make filename unique per chat
       unique_filename = f"{chat_id}_{filename}"
       file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
       
       print(f"Saving file to: {file_path}")
       file.save(file_path)
       
       if os.path.exists(file_path):
           print(f"File saved successfully for chat {chat_id}")
       else:
           return jsonify({'success': False, 'error': 'File save failed'})
       
       # Process through chat-specific processor
       success = chat_processor.load_dataset_for_chat(file_path, chat_id)
       
       if success:
           return jsonify({'success': True, 'message': 'Dataset loaded successfully for this chat!'})
       else:
           return jsonify({'success': False, 'error': 'Failed to process dataset'})
           
   except Exception as e:
       print(f"Upload error: {e}")
       traceback.print_exc()
       return jsonify({'success': False, 'error': f'Upload failed: {str(e)}'})

@app.route('/status')
def get_status():
   """Get overall status - now chat-agnostic"""
   return jsonify({
       'stage': 'waiting',
       'progress': 0,
       'message': 'Upload data to a specific chat to begin analysis'
   })

@app.route('/chat/<chat_id>/status')
def get_chat_status(chat_id):
   """Get dataset status for specific chat"""
   if chat_processor.has_dataset_for_chat(chat_id):
       dataset_info = chat_processor.get_chat_dataset_info(chat_id)
       dataset_context = dataset_info.get('context', {})
       
       # Generate context-aware insights
       base_insights = [
           f"✅ Dataset loaded: {dataset_info['total_rows']:,} rows × {dataset_info['total_columns']} columns",
           f"📊 File size: {dataset_info['file_size_formatted']}",
           f"🗃️ SQLite database created for efficient querying"
       ]
       
       # Add dataset-type specific insights
       if dataset_context.get('type') != 'general':
           dataset_type = dataset_context['type'].title()
           confidence = dataset_context.get('confidence', 0)
           base_insights.append(f"🎯 Detected as {dataset_type} dataset (confidence: {confidence:.0%})")
           
           # Add suggestions based on dataset type
           suggestions = dataset_context.get('suggestions', [])
           if suggestions:
               base_insights.append(f"💡 Ready for {dataset_type.lower()}-specific analysis")
       
       base_insights.extend([
           "🤖 Enhanced conversational AI ready",
           "⚡ Handles any dataset size and complexity"
       ])
       
       return jsonify({
           'stage': 'complete',
           'progress': 100,
           'dataset_info': {
               'shape': {'rows': dataset_info['total_rows'], 'columns': dataset_info['total_columns']},
               'columns': dataset_info['columns'],
               'file_size_mb': dataset_info['file_size_mb'],
               'file_size_formatted': dataset_info['file_size_formatted'],
               'file_size_bytes': dataset_info['file_size_bytes'],
               'file_name': dataset_info.get('file_name', 'Unknown'),
               'file_path': dataset_info.get('file_path', ''),
               'dataset_type': dataset_context.get('type', 'general'),
               'suggestions': dataset_context.get('suggestions', [])
           },
           'insights': base_insights
       })
   else:
       return jsonify({
           'stage': 'waiting',
           'progress': 0,
           'message': 'No dataset loaded for this chat'
       })

@app.route('/chat/new', methods=['POST'])
def create_new_chat():
   """Create new chat session"""
   chat_id = str(uuid.uuid4())
   timestamp = datetime.now()
   
   conn = sqlite3.connect('data/chat_history.db')
   cursor = conn.cursor()
   
   cursor.execute('''
       INSERT INTO chats (id, title, created_at, updated_at)
       VALUES (?, ?, ?, ?)
   ''', (chat_id, 'New Analysis', timestamp, timestamp))
   
   conn.commit()
   conn.close()
   
   return jsonify({'success': True, 'chat_id': chat_id})

@app.route('/chat/<chat_id>/dataset-status')
def get_chat_dataset_status(chat_id):
   """Check if there's an active dataset for this specific chat"""
   try:
       if chat_processor.load_chat_dataset(chat_id):
           dataset_info = chat_processor.get_chat_dataset_info(chat_id)
           dataset_context = dataset_info.get('context', {})
           
           return jsonify({
               'has_dataset': True,
               'dataset_info': {
                   'shape': {
                       'rows': dataset_info['total_rows'], 
                       'columns': dataset_info['total_columns']
                   },
                   'columns': dataset_info['columns'],
                   'file_size_formatted': dataset_info.get('file_size_formatted', '0 B'),
                   'file_name': dataset_info.get('file_name', 'Unknown'),
                   'dataset_type': dataset_context.get('type', 'general'),
                   'suggestions': dataset_context.get('suggestions', [])
               }
           })
       
       return jsonify({'has_dataset': False})
       
   except Exception as e:
       print(f"Error checking dataset status for chat {chat_id}: {e}")
       return jsonify({'has_dataset': False})

@app.route('/chat/<chat_id>/delete', methods=['DELETE'])
def delete_chat(chat_id):
   """Delete a specific chat and its associated dataset"""
   try:
       conn = sqlite3.connect('data/chat_history.db')
       cursor = conn.cursor()
       
       # Get dataset files before deletion
       cursor.execute('SELECT file_path, sqlite_path FROM chat_datasets WHERE chat_id = ?', (chat_id,))
       dataset_row = cursor.fetchone()
       
       # Delete from all tables
       cursor.execute('DELETE FROM messages WHERE chat_id = ?', (chat_id,))
       cursor.execute('DELETE FROM chat_datasets WHERE chat_id = ?', (chat_id,))
       cursor.execute('DELETE FROM chats WHERE id = ?', (chat_id,))
       
       conn.commit()
       conn.close()
       
       # Clean up files
       if dataset_row:
           file_path, sqlite_path = dataset_row
           try:
               if os.path.exists(file_path):
                   os.remove(file_path)
               if os.path.exists(sqlite_path):
                   os.remove(sqlite_path)
           except Exception as e:
               print(f"Error cleaning up files: {e}")
       
       # Remove from memory
       if chat_id in chat_processor.chat_datasets:
           del chat_processor.chat_datasets[chat_id]
       
       return jsonify({'success': True})
       
   except Exception as e:
       print(f"Error deleting chat: {e}")
       return jsonify({'success': False, 'error': str(e)})

@app.route('/chat/<chat_id>/rename', methods=['PUT'])
def rename_chat(chat_id):
   """Rename a specific chat"""
   try:
       data = request.get_json()
       new_title = data.get('title', '').strip()
       
       if not new_title:
           return jsonify({'success': False, 'error': 'Title is required'})
       
       if len(new_title) > 100:
           return jsonify({'success': False, 'error': 'Title is too long'})
       
       conn = sqlite3.connect('data/chat_history.db')
       cursor = conn.cursor()
       
       cursor.execute('''
           UPDATE chats SET title = ?, updated_at = ?
           WHERE id = ?
       ''', (new_title, datetime.now(), chat_id))
       
       if cursor.rowcount == 0:
           conn.close()
           return jsonify({'success': False, 'error': 'Chat not found'})
       
       conn.commit()
       conn.close()
       
       return jsonify({'success': True, 'message': 'Chat renamed successfully'})
       
   except Exception as e:
       print(f"Error renaming chat: {e}")
       return jsonify({'success': False, 'error': str(e)})

@app.route('/api/similarity-analysis', methods=['POST'])
def analyze_similarity():
   """Analyze similarity patterns in chat dataset"""
   try:
       data = request.get_json()
       chat_id = data.get('chat_id')
       analysis_type = data.get('analysis_type', 'comprehensive')
       threshold = data.get('threshold', 0.7)
       search_query = data.get('search_query', '')
       categories = data.get('categories', [])
       
       if not chat_id:
           return jsonify({'success': False, 'error': 'Chat ID required'})
       
       # Check if chat has dataset
       if not chat_processor.has_dataset_for_chat(chat_id):
           return jsonify({'success': False, 'error': 'No dataset loaded for this chat'})
       
       dataset_info = chat_processor.get_chat_dataset_info(chat_id)
       
       # Perform similarity analysis
       result = similarity_analyzer.analyze_dataset_similarity(
           dataset_info, chat_id, analysis_type, threshold, search_query, categories
       )
       
       return jsonify(result)
       
   except Exception as e:
       print(f"Similarity analysis error: {e}")
       traceback.print_exc()
       return jsonify({'success': False, 'error': str(e)})

@app.route('/query', methods=['POST'])
def process_query():
    """Process user query for specific chat with conversational AI"""
    data = request.get_json()
    question = data.get('question', '').strip()
    chat_id = data.get('chat_id')
    
    if not question:
        return jsonify({'success': False, 'error': 'Please provide a question'})
    
    if not chat_id:
        return jsonify({'success': False, 'error': 'Chat ID required'})
    
    # Load chat dataset if not in memory
    if not chat_processor.load_chat_dataset(chat_id):
        return jsonify({'success': False, 'error': 'No dataset loaded for this chat. Please upload a dataset first.'})
    
    print(f"Processing conversational query for chat {chat_id}: {question}")
    
    try:
        # Process through enhanced chat-specific MCP with conversational AI
        result = chat_processor.query_data_for_chat(question, chat_id)
        
        # Save to chat history if successful
        if result.get('success') and chat_id:
            try:
                conn = sqlite3.connect('data/chat_history.db')
                cursor = conn.cursor()
                
                # Save user message
                cursor.execute('''
                    INSERT INTO messages (chat_id, type, content, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (chat_id, 'user', question, datetime.now()))
                
                # Save assistant response
                result_json = safe_json_dumps(result.get('results', []))
                cursor.execute('''
                    INSERT INTO messages (chat_id, type, content, query, result, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (chat_id, 'assistant', result['answer'], 
                      result.get('sql_query'), result_json, datetime.now()))
                
                # Update chat title if it's the first real question
                cursor.execute('''
                    UPDATE chats SET title = ?, updated_at = ?
                    WHERE id = ? AND title = 'New Analysis'
                ''', (question[:50] + ('...' if len(question) > 50 else ''), datetime.now(), chat_id))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                print(f"Error saving to chat history: {e}")
        
        return jsonify(result)
        
    except Exception as e:
        error_message = str(e)
        print(f"Query processing error: {error_message}")
        
        # Handle specific API errors
        if error_message.startswith('RATE_LIMIT_EXCEEDED:'):
            return jsonify({
                'success': False, 
                'error': error_message.replace('RATE_LIMIT_EXCEEDED: ', ''),
                'error_type': 'RATE_LIMIT_EXCEEDED',
                'error_code': 429,
                'user_message': 'API rate limit exceeded. Please wait a moment or contact your administrator to upgrade the API quota.'
            })
        elif error_message.startswith('GEMINI_API_ERROR:'):
            return jsonify({
                'success': False,
                'error': error_message.replace('GEMINI_API_ERROR: ', ''),
                'error_type': 'GEMINI_API_ERROR',
                'user_message': 'There was an issue with the AI service. Please try again or contact support.'
            })
        elif error_message.startswith('GEMINI_API_TIMEOUT:'):
            return jsonify({
                'success': False,
                'error': error_message.replace('GEMINI_API_TIMEOUT: ', ''),
                'error_type': 'GEMINI_API_TIMEOUT',
                'user_message': 'The AI service is taking too long to respond. Please try again.'
            })
        elif error_message.startswith('GEMINI_API_CONNECTION:'):
            return jsonify({
                'success': False,
                'error': error_message.replace('GEMINI_API_CONNECTION: ', ''),
                'error_type': 'GEMINI_API_CONNECTION',
                'user_message': 'Cannot connect to AI service. Please check your internet connection.'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Analysis failed: {error_message}',
                'error_type': 'GENERAL_ERROR',
                'user_message': 'An unexpected error occurred. Please try again.'
            })


@app.route('/chat/history')
def get_chat_history():
   """Get chat history"""
   try:
       conn = sqlite3.connect('data/chat_history.db')
       cursor = conn.cursor()
       
       cursor.execute('''
           SELECT c.id, c.title, c.created_at, c.updated_at,
                  m.type, m.content, m.timestamp
           FROM chats c
           LEFT JOIN messages m ON c.id = m.chat_id
           ORDER BY c.updated_at DESC, m.timestamp ASC
       ''')
       
       rows = cursor.fetchall()
       conn.close()
       
       chats = {}
       for row in rows:
           chat_id = row[0]
           if chat_id not in chats:
               chats[chat_id] = {
                   'id': chat_id,
                   'title': row[1],
                   'created_at': row[2],
                   'updated_at': row[3],
                   'messages': []
               }
           
           if row[4]:  # If message exists
               chats[chat_id]['messages'].append({
                   'type': row[4],
                   'content': row[5],
                   'timestamp': row[6]
               })
       
       return jsonify({'chats': list(chats.values())})
       
   except Exception as e:
       print(f"Error getting chat history: {e}")
       return jsonify({'chats': []})

@app.route('/chat/<chat_id>')
def get_chat(chat_id):
   """Get specific chat"""
   try:
       conn = sqlite3.connect('data/chat_history.db')
       cursor = conn.cursor()
       
       cursor.execute('''
           SELECT c.id, c.title, c.created_at, c.updated_at
           FROM chats c
           WHERE c.id = ?
       ''', (chat_id,))
       
       chat_row = cursor.fetchone()
       if not chat_row:
           return jsonify({'error': 'Chat not found'}), 404
       
       cursor.execute('''
           SELECT type, content, query, result, timestamp
           FROM messages
           WHERE chat_id = ?
           ORDER BY timestamp ASC
       ''', (chat_id,))
       
       message_rows = cursor.fetchall()
       conn.close()
       
       chat = {
           'id': chat_row[0],
           'title': chat_row[1],
           'created_at': chat_row[2],
           'updated_at': chat_row[3],
           'messages': []
       }
       
       for row in message_rows:
           message = {
               'type': row[0],
               'content': row[1],
               'query': row[2],
               'timestamp': row[4]
           }
           
           if row[3]:
               try:
                   message['result'] = json.loads(row[3])
               except:
                   message['result'] = None
           else:
               message['result'] = None
               
           chat['messages'].append(message)
       
       return jsonify({'chat': chat})
       
   except Exception as e:
       print(f"Error getting chat: {e}")
       return jsonify({'error': 'Failed to load chat'}), 500

if __name__ == '__main__':
   port = int(os.getenv('PORT', 8080))
   debug = os.getenv('DEBUG', 'True').lower() == 'true'
   host = os.getenv('HOST', '0.0.0.0')
   
   print(f"🚀 Starting Enhanced Conversational AI Sheet Chat on {host}:{port}")
   app.run(debug=debug, host=host, port=port)