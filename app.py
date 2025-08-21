# app.py
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

def get_file_size_mb(file_path):
    """Get accurate file size in MB"""
    try:
        if os.path.exists(file_path):
            size_bytes = os.path.getsize(file_path)
            size_mb = size_bytes / (1024 * 1024)
            print(f"File size calculated: {size_bytes} bytes = {size_mb:.2f} MB")
            return round(size_mb, 2)
        else:
            print(f"File not found: {file_path}")
            return 0.0
    except Exception as e:
        print(f"Error calculating file size: {e}")
        return 0.0
    

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
            
            print(f"File size calculated: {size_bytes} bytes = {size_str}")
            return {
                'size_bytes': size_bytes,
                'size_mb': round(size_mb, 3),  # Keep MB for internal calculations
                'size_formatted': size_str      # Human readable format
            }
        else:
            print(f"File not found: {file_path}")
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

class DirectMCPProcessor:
    """Direct MCP-style data processor without external server"""
    def __init__(self):
        self.dataset_info = {}
        self.sqlite_path = None
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.gemini_url = os.getenv('GEMINI_API_URL', 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent')
        
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
            
    def load_dataset(self, file_path):
        """Load dataset efficiently for any size"""
        try:
            print(f"Loading dataset from: {file_path}")
            
            # Verify file exists and get size immediately
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_size_info = get_file_size_formatted(file_path)
            file_extension = Path(file_path).suffix.lower()
            file_name = Path(file_path).name
            
            print(f"Processing file: {file_name} ({file_size_info['size_formatted']})")
            
            # Get basic info without loading full dataset
            if file_extension == '.csv':
                # Read sample to understand structure
                sample_df = pd.read_csv(file_path, nrows=100, low_memory=False)
                
                # Count total rows efficiently
                print("Counting total rows...")
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    total_rows = sum(1 for line in f) - 1  # Subtract header
                    
            elif file_extension in ['.xlsx', '.xls']:
                # For Excel files, we need a different approach
                print("Loading Excel file...")
                
                # Read sample to understand structure
                sample_df = pd.read_excel(file_path, nrows=100)
                
                # Count total rows by loading the full Excel file
                print("Counting Excel rows...")
                try:
                    # Try to get row count efficiently using openpyxl
                    import openpyxl
                    wb = openpyxl.load_workbook(file_path, read_only=True)
                    ws = wb.active
                    total_rows = ws.max_row - 1  # Subtract header
                    wb.close()
                except:
                    # Fallback: load the entire Excel file
                    print("Using fallback method to count Excel rows...")
                    full_df = pd.read_excel(file_path)
                    total_rows = len(full_df)
                    del full_df  # Free memory immediately
                    
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            print(f"Dataset has {total_rows:,} rows and {len(sample_df.columns)} columns")
            
            # Convert datetime columns to strings for JSON compatibility
            sample_df_json_safe = sample_df.copy()
            for col in sample_df_json_safe.columns:
                if pd.api.types.is_datetime64_any_dtype(sample_df_json_safe[col]):
                    sample_df_json_safe.loc[:, col] = sample_df_json_safe[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Store dataset info with JSON-safe data
            self.dataset_info = {
                'file_path': file_path,
                'file_name': file_name,
                'total_rows': int(total_rows),
                'total_columns': int(len(sample_df.columns)),
                'columns': list(sample_df.columns),
                'dtypes': {col: str(dtype) for col, dtype in sample_df.dtypes.items()},
                'sample_data': convert_dataframe_for_json(sample_df_json_safe.head(5)),
                'file_size_mb': file_size_info['size_mb'],           # For internal calculations
                'file_size_formatted': file_size_info['size_formatted'],  # For display
                'file_size_bytes': file_size_info['size_bytes'],     # Raw bytes
                'cleaned_columns': [self._clean_column_name(col) for col in sample_df.columns]
            }
            
            print(f"Stored dataset info with file size: {file_size_info['size_formatted']}")
            
            # Create SQLite database for efficient querying
            self.sqlite_path = file_path.replace(file_extension, '.db')
            success = self._create_sqlite_db(file_path)
            
            if success:
                print(f"Dataset loaded successfully: {total_rows:,} rows, {len(sample_df.columns)} columns")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            traceback.print_exc()
            return False
    def _clean_column_name(self, col_name):
        """Clean column names for SQLite compatibility"""
        import re
        cleaned = re.sub(r'[^\w]', '_', str(col_name))
        if cleaned and cleaned[0].isdigit():
            cleaned = 'col_' + cleaned
        return cleaned or 'unnamed_col'
    
    def _create_sqlite_db(self, file_path):
        """Create SQLite database from file with robust handling"""
        try:
            print("Creating SQLite database for efficient querying...")
            
            file_extension = Path(file_path).suffix.lower()
            conn = sqlite3.connect(self.sqlite_path)
            
            # Increase SQLite limits for large datasets
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA mmap_size = 268435456")  # 256MB
            conn.execute("PRAGMA page_size = 65536")
            conn.execute("PRAGMA cache_size = 10000")
            conn.execute("PRAGMA synchronous = OFF")
            conn.execute("PRAGMA journal_mode = MEMORY")
            
            # Determine appropriate chunk size based on number of columns
            num_columns = len(self.dataset_info['columns'])
            
            # SQLite has a default limit of 999 variables per query
            # Calculate safe chunk size: (999 / num_columns) - safety margin
            if num_columns > 100:
                chunk_size = max(1, int(800 / num_columns))  # Very wide datasets
            elif num_columns > 50:
                chunk_size = max(10, int(900 / num_columns))  # Wide datasets
            else:
                chunk_size = 1000  # Normal datasets
                
            print(f"Using chunk size: {chunk_size} rows (for {num_columns} columns)")
            
            processed_chunks = 0
            total_processed = 0
            
            if file_extension == '.csv':
                # CSV supports chunking
                chunk_iterator = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)
                
                for chunk in chunk_iterator:
                    success = self._process_chunk_to_sqlite(chunk, conn, processed_chunks == 0)
                    if not success:
                        print(f"Failed to process chunk {processed_chunks}")
                        break
                        
                    processed_chunks += 1
                    total_processed += len(chunk)
                    
                    if processed_chunks % 10 == 0:
                        print(f"Processed {total_processed:,} rows...")
                        
            else:
                # Excel processing in batches
                print("Processing Excel file in batches...")
                
                # Get total rows
                try:
                    import openpyxl
                    wb = openpyxl.load_workbook(file_path, read_only=True)
                    ws = wb.active
                    total_rows = ws.max_row
                    wb.close()
                except:
                    temp_df = pd.read_excel(file_path)
                    total_rows = len(temp_df) + 1
                    del temp_df
                
                start_row = 0
                
                while start_row < total_rows - 1:
                    try:
                        if start_row == 0:
                            chunk = pd.read_excel(file_path, skiprows=0, nrows=chunk_size)
                        else:
                            chunk = pd.read_excel(file_path, skiprows=start_row + 1, nrows=chunk_size, header=None)
                            chunk.columns = self.dataset_info['cleaned_columns']
                        
                        if len(chunk) == 0:
                            break
                            
                        success = self._process_chunk_to_sqlite(chunk, conn, processed_chunks == 0)
                        if not success:
                            print(f"Failed to process Excel chunk starting at row {start_row}")
                            break
                        
                        processed_chunks += 1
                        total_processed += len(chunk)
                        start_row += chunk_size
                        
                        print(f"Processed {total_processed:,} of {total_rows-1:,} rows...")
                        
                    except Exception as e:
                        print(f"Error processing Excel batch starting at row {start_row}: {e}")
                        break
            
            # Create indexes for better performance
            print("Creating database indexes...")
            cursor = conn.cursor()
            
            # Index first few columns for better query performance
            for i, col in enumerate(self.dataset_info['cleaned_columns'][:3]):
                try:
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{col} ON data({col})")
                    print(f"Created index on column: {col}")
                except Exception as e:
                    print(f"Could not create index on {col}: {e}")
            
            conn.commit()
            conn.close()
            
            print(f"SQLite database created successfully: {self.sqlite_path}")
            print(f"Total processed: {total_processed:,} rows")
            return True
            
        except Exception as e:
            print(f"Error creating SQLite database: {e}")
            traceback.print_exc()
            return False
    
    def _process_chunk_to_sqlite(self, chunk, conn, is_first_chunk):
        """Process a single chunk to SQLite with error handling"""
        try:
            # Clean column names
            chunk.columns = [self._clean_column_name(col) for col in chunk.columns]
            
            # Prepare chunk for SQLite
            chunk = self._prepare_chunk_for_sqlite(chunk)
            
            # Use individual INSERT statements for very wide datasets
            num_columns = len(chunk.columns)
            
            if num_columns > 100:  # Very wide dataset - use individual inserts
                return self._insert_chunk_individually(chunk, conn, is_first_chunk)
            else:  # Normal dataset - use pandas to_sql with smaller batches
                batch_size = min(100, max(1, int(800 / num_columns)))
                
                # Split chunk into smaller batches if needed
                for i in range(0, len(chunk), batch_size):
                    batch = chunk.iloc[i:i+batch_size]
                    batch.to_sql('data', conn, 
                               if_exists='append' if not (is_first_chunk and i == 0) else 'replace',
                               index=False, 
                               method=None)  # Use default method, not 'multi'
                
                return True
                
        except Exception as e:
            print(f"Error processing chunk: {e}")
            return False
    
    def _insert_chunk_individually(self, chunk, conn, is_first_chunk):
        """Insert chunk using individual INSERT statements for very wide datasets"""
        try:
            cursor = conn.cursor()
            
            # Create table if first chunk
            if is_first_chunk:
                # Drop table if exists
                cursor.execute("DROP TABLE IF EXISTS data")
                
                # Create table with all columns as TEXT
                columns_def = ', '.join([f'"{col}" TEXT' for col in chunk.columns])
                create_sql = f"CREATE TABLE data ({columns_def})"
                cursor.execute(create_sql)
                print(f"Created table with {len(chunk.columns)} columns")
            
            # Prepare INSERT statement
            placeholders = ', '.join(['?' for _ in chunk.columns])
            column_names = ', '.join([f'"{col}"' for col in chunk.columns])
            insert_sql = f"INSERT INTO data ({column_names}) VALUES ({placeholders})"
            
            # Insert rows one by one
            for _, row in chunk.iterrows():
                values = [str(val) if val is not None else '' for val in row.values]
                cursor.execute(insert_sql, values)
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"Error in individual insert: {e}")
            return False
    
    def _prepare_chunk_for_sqlite(self, chunk):
        """Prepare chunk for SQLite by handling data types"""
        # Make a copy to avoid warnings
        chunk = chunk.copy()
        
        # Convert datetime columns to strings
        for col in chunk.columns:
            if pd.api.types.is_datetime64_any_dtype(chunk[col]):
                chunk.loc[:, col] = chunk[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            elif chunk[col].dtype == 'object':
                # Handle mixed types in object columns
                chunk.loc[:, col] = chunk[col].apply(lambda x: 
                    x.isoformat() if isinstance(x, (datetime, date, pd.Timestamp)) 
                    else str(x) if x is not None else ''
                )
        
        # Fill NaN values
        chunk = chunk.fillna('')
        
        return chunk
    
    def query_data(self, question):
        """Process query without result limitations"""
        try:
            if not self.dataset_info:
                return {'success': False, 'error': 'No dataset loaded'}
            
            print(f"Processing question: {question}")
            
            # Step 1: Generate SQL query
            sql_query = self._generate_sql_query(question)
            if not sql_query:
                return {'success': False, 'error': 'Could not understand the question'}
            
            print(f"Generated SQL: {sql_query}")
            
            # Step 2: Execute query - get ALL results
            results = self._execute_sql_query(sql_query)
            if results is None:
                return {'success': False, 'error': 'Query execution failed'}
            
            # Step 3: Generate response
            response = self._generate_response(question, sql_query, results)
            
            # Step 4: Determine how many results to show in UI
            display_results = self._prepare_display_results(results, question)
            
            return {
                'success': True,
                'answer': response,
                'sql_query': sql_query,
                'results': display_results,
                'total_results': len(results),
                'analysis_type': 'mcp_analysis',
                'show_all_data': self._should_show_all_data(question, len(results))
            }
            
        except Exception as e:
            print(f"Error in query processing: {e}")
            traceback.print_exc()
            return {'success': False, 'error': f'Analysis failed: {str(e)}'}
    
    def _prepare_display_results(self, results, question):
        """Decide how many results to show based on question context"""
        question_lower = question.lower()
        
        # Keywords that indicate user wants ALL data
        show_all_keywords = [
            'all', 'complete', 'entire', 'full', 'every', 'total',
            'show me all', 'give me all', 'list all', 'all data',
            'complete data', 'entire dataset', 'everything'
        ]
        
        # Keywords that indicate user wants limited data
        limit_keywords = [
            'top', 'first', 'sample', 'few', 'some', 'example'
        ]
        
        # Check if user wants all data
        wants_all_data = any(keyword in question_lower for keyword in show_all_keywords)
        wants_limited = any(keyword in question_lower for keyword in limit_keywords)
        
        # If user explicitly asks for all data, return everything
        if wants_all_data and not wants_limited:
            print(f"User requested all data - returning {len(results)} results")
            return results
        
        # If it's a small result set (under 100), show all
        elif len(results) <= 100:
            print(f"Small result set ({len(results)} results) - showing all")
            return results
        
        # If it's aggregated data (like counts, averages), show all
        elif self._is_aggregated_query(results):
            print(f"Aggregated data - showing all {len(results)} results")
            return results
        
        # For large detailed datasets, show first 50 unless user asked for all
        else:
            print(f"Large result set ({len(results)} results) - showing first 50")
            return results[:50]
    
    def _is_aggregated_query(self, results):
        """Check if results are aggregated (counts, sums, averages)"""
        if not results:
            return False
        
        # Check column names for aggregation functions
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
    
    def _generate_sql_query(self, question):
        """Generate SQL query - let AI decide limits, not us"""
        try:
            # Create schema description
            schema_info = f"""
Table: data
Total rows: {self.dataset_info['total_rows']:,}
Columns: {len(self.dataset_info['columns'])}

Column mapping (original -> cleaned):
"""
            for orig, cleaned in zip(self.dataset_info['columns'], self.dataset_info['cleaned_columns']):
                dtype = self.dataset_info['dtypes'].get(orig, 'text')
                schema_info += f"- '{orig}' -> {cleaned} ({dtype})\n"
            
            # Use safe JSON serialization for sample data
            sample_data_str = safe_json_dumps(self.dataset_info['sample_data'][:3])
            schema_info += f"\nSample data:\n{sample_data_str}"
            
            prompt = f"""
You are an expert SQL query generator. Convert the natural language question to a SQLite query.

DATABASE SCHEMA:
{schema_info}

IMPORTANT RULES:
1. Use the cleaned column names (right side of mapping) in your SQL
2. DO NOT add arbitrary LIMIT clauses unless the user specifically asks for "top N" or "first N"
3. If user asks for "all data" or "show me all" - return ALL data without LIMIT
4. Use appropriate aggregations (COUNT, SUM, AVG, etc.) when needed
5. Handle text searches with LIKE and % wildcards
6. Only add LIMIT if the user specifically requests a limited number of results
7. Return only the SQL query, no explanations
8. For context questions, use SELECT * FROM data to show all data
9. Always wrap column names in double quotes like "column_name" for SQLite compatibility

USER QUESTION: {question}

Generate SQLite query:
"""

            response = self._call_gemini_api(prompt)
            sql_query = self._extract_sql_from_response(response)
            
            return sql_query
            
        except Exception as e:
            print(f"Error generating SQL: {e}")
            traceback.print_exc()
            return None
    
    def _extract_sql_from_response(self, response):
        """Extract clean SQL from Gemini response"""
        response = response.strip()
        
        # Remove markdown code blocks
        if '```sql' in response:
            response = response.split('```sql')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1]
        
        # Clean up
        response = response.strip().rstrip(';')
        
        # Basic validation
        if not response.upper().startswith('SELECT'):
            # Try to find SELECT statement
            lines = response.split('\n')
            for line in lines:
                if line.strip().upper().startswith('SELECT'):
                    response = line.strip()
                    break
        
        return response
    
    def _execute_sql_query(self, sql_query):
        """Execute SQL query without artificial limits"""
        try:
            # Basic SQL injection protection
            forbidden_words = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
            if any(word in sql_query.upper() for word in forbidden_words):
                raise ValueError("Query contains forbidden operations")
            
            conn = sqlite3.connect(self.sqlite_path)
            
            # Set pragmas for better performance
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA cache_size = 10000")
            
            # Execute query
            cursor = conn.cursor()
            cursor.execute(sql_query)
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            
            # Fetch ALL results - no artificial limits
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
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
    
    def _generate_response(self, question, sql_query, results):
        """Generate response that acknowledges the full dataset"""
        try:
            result_count = len(results)
            
            # For large datasets, use a sample for AI response but mention full count
            if result_count > 20:
                results_for_ai = results[:10]
                results_note = f"Analyzed all {result_count:,} results. Here's a summary based on the complete dataset:"
            else:
                results_for_ai = results
                results_note = f"Found {result_count} results:"
            
            # Use safe JSON serialization
            results_json = safe_json_dumps(results_for_ai)
            
            prompt = f"""
You are a data analyst explaining query results. The user asked: "{question}"

IMPORTANT: You analyzed the COMPLETE dataset with {result_count:,} total results.

SQL EXECUTED: {sql_query}
SAMPLE RESULTS (first 10 of {result_count}): {results_json}

Provide a comprehensive analysis that:
1. States you found {result_count:,} total results
2. Directly answers the user's question using the complete dataset
3. Includes specific insights, patterns, and key statistics
4. Mentions that the complete data is available in the table below
5. Be conversational and insightful

Start your response with: "I found {result_count:,} results for your query about..."

Response:
"""

            response = self._call_gemini_api(prompt)
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Query executed successfully. Found {len(results):,} results in the dataset."
    
    def _call_gemini_api(self, prompt):
        """Call Gemini API"""
        headers = {'Content-Type': 'application/json'}
        
        data = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'temperature': 0.1,
                'topP': 0.8,
                'topK': 40,
                'maxOutputTokens': 2000,
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
            else:
                raise Exception(f"Gemini API error: {response.status_code}")
                
        except Exception as e:
            print(f"Gemini API error: {e}")
            raise e

# Global processor instance
mcp_processor = DirectMCPProcessor()

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
    
    conn.commit()
    conn.close()

init_db()
print("MCP-style processor initialized successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload with accurate file size tracking"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    # Validate file type
    allowed_extensions = {'.csv', '.xlsx', '.xls'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        return jsonify({'success': False, 'error': 'Only CSV and Excel files are supported'})
    
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        print(f"Saving file to: {file_path}")
        
        # Save the file first
        file.save(file_path)
        
        # Verify file was saved and get size
        if os.path.exists(file_path):
            actual_size = get_file_size_mb(file_path)
            print(f"File saved successfully. Size: {actual_size} MB")
        else:
            print("ERROR: File was not saved properly")
            return jsonify({'success': False, 'error': 'File save failed'})
        
        # Process through MCP processor
        success = mcp_processor.load_dataset(file_path)
        
        if success:
            return jsonify({'success': True, 'message': 'Dataset loaded successfully!'})
        else:
            return jsonify({'success': False, 'error': 'Failed to process dataset'})
            
    except Exception as e:
        print(f"Upload error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Upload failed: {str(e)}'})

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
        
        # Update the chat title
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

@app.route('/status')
def get_status():
    """Get dataset status with persistent information"""
    if mcp_processor.dataset_info:
        info = mcp_processor.dataset_info
        print(f"Returning status with file_size_formatted: {info.get('file_size_formatted', '0 B')}")
        return jsonify({
            'stage': 'complete',
            'progress': 100,
            'dataset_info': {
                'shape': {'rows': info['total_rows'], 'columns': info['total_columns']},
                'columns': info['columns'],
                'file_size_mb': info['file_size_mb'],
                'file_size_formatted': info['file_size_formatted'],
                'file_size_bytes': info['file_size_bytes'],
                'file_name': info.get('file_name', 'Unknown'),
                'file_path': info.get('file_path', '')
            },
            'insights': [
                f"✅ Dataset loaded: {info['total_rows']:,} rows × {info['total_columns']} columns",
                f"📊 File size: {info['file_size_formatted']}",
                f"🗃️ SQLite database created for efficient querying",
                f"🤖 Ready for natural language analysis",
                f"⚡ Handles any dataset size and width"
            ]
        })
    else:
        return jsonify({
            'stage': 'waiting',
            'progress': 0
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
    """Check if there's an active dataset for this chat"""
    try:
        # Check if we have dataset info and if the file still exists
        if mcp_processor.dataset_info and mcp_processor.dataset_info.get('file_path'):
            file_path = mcp_processor.dataset_info['file_path']
            if os.path.exists(file_path):
                return jsonify({
                    'has_dataset': True,
                    'dataset_info': {
                        'shape': {
                            'rows': mcp_processor.dataset_info['total_rows'], 
                            'columns': mcp_processor.dataset_info['total_columns']
                        },
                        'columns': mcp_processor.dataset_info['columns'],
                        'file_size_formatted': mcp_processor.dataset_info.get('file_size_formatted', '0 B'),
                        'file_name': mcp_processor.dataset_info.get('file_name', 'Unknown')
                    }
                })
        
        return jsonify({'has_dataset': False})
        
    except Exception as e:
        print(f"Error checking dataset status: {e}")
        return jsonify({'has_dataset': False})

@app.route('/chat/<chat_id>/delete', methods=['DELETE'])
def delete_chat(chat_id):
    """Delete a specific chat"""
    try:
        conn = sqlite3.connect('data/chat_history.db')
        cursor = conn.cursor()
        
        # Delete messages first (foreign key constraint)
        cursor.execute('DELETE FROM messages WHERE chat_id = ?', (chat_id,))
        
        # Delete the chat
        cursor.execute('DELETE FROM chats WHERE id = ?', (chat_id,))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"Error deleting chat: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/query', methods=['POST'])
def process_query():
    """Process user query"""
    data = request.get_json()
    question = data.get('question', '').strip()
    chat_id = data.get('chat_id')
    
    if not question:
        return jsonify({'success': False, 'error': 'Please provide a question'})
    
    if not mcp_processor.dataset_info:
        return jsonify({'success': False, 'error': 'No dataset loaded. Please upload a dataset first.'})
    
    print(f"Processing query: {question}")
    
    # Process through MCP
    result = mcp_processor.query_data(question)
    
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
            
            # Save assistant response - use safe JSON serialization
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
            
            # Safely parse result JSON
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
    
    print(f"🚀 Starting MCP-powered AI Sheet Chat on {host}:{port}")
    app.run(debug=debug, host=host, port=port)