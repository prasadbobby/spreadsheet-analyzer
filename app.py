import os
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import requests
import re
import uuid
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configuration from environment variables
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-default-secret-key-here')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 1024 * 1024 * 1024))  # Default 1GB

# Gemini API configuration from environment
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_API_URL = os.getenv('GEMINI_API_URL', 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent')

# Validate required environment variables
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)

print(f"App configured with:")
print(f"- Upload folder: {app.config['UPLOAD_FOLDER']}")
print(f"- Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024)} MB")
print(f"- Gemini API URL: {GEMINI_API_URL}")
print(f"- API Key configured: {'Yes' if GEMINI_API_KEY else 'No'}")

class DataProcessor:
    def __init__(self):
        self.df = None
        self.chunks = []
        self.metadata = {}
        self.data_context = ""
        
    def load_data(self, file_path):
        """Load data from CSV or Excel file"""
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")
            
            print(f"Loaded data shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            print(f"First few rows:\n{self.df.head()}")
            
            self.metadata = {
                'shape': {'rows': len(self.df), 'columns': len(self.df.columns)},
                'columns': list(self.df.columns),
                'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
                'null_counts': self.df.isnull().sum().to_dict(),
                'file_path': file_path
            }
            
            # Create comprehensive data context for AI
            self.create_data_context()
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            traceback.print_exc()
            return False
    
    def create_data_context(self):
        """Create comprehensive data context for AI understanding"""
        context_parts = []
        
        # Basic dataset info
        context_parts.append(f"DATASET OVERVIEW:")
        context_parts.append(f"- Total records: {len(self.df)}")
        context_parts.append(f"- Total columns: {len(self.df.columns)}")
        context_parts.append(f"- Column names: {', '.join(self.df.columns)}")
        
        # Detailed column analysis with data types
        context_parts.append(f"\nCOLUMN DETAILS:")
        for col in self.df.columns:
            col_type = str(self.df[col].dtype)
            unique_count = self.df[col].nunique()
            
            if col_type in ['int64', 'float64', 'int32', 'float32']:
                stats = self.df[col].describe()
                context_parts.append(f"- {col}: NUMERIC ({col_type}) - Range: {stats['min']} to {stats['max']}, Average: {stats['mean']:.2f}")
            else:
                # Show actual unique values for text columns
                unique_values = self.df[col].dropna().unique()
                if len(unique_values) <= 15:
                    context_parts.append(f"- {col}: TEXT ({col_type}) - Values: {list(unique_values)}")
                else:
                    sample_values = list(unique_values[:10])
                    context_parts.append(f"- {col}: TEXT ({col_type}) - {unique_count} unique values, Sample: {sample_values}")
        
        # Show ALL actual records (not just sample)
        context_parts.append(f"\nCOMPLETE DATASET RECORDS:")
        for idx, row in self.df.iterrows():
            record_parts = []
            for col, val in row.items():
                if pd.notna(val):
                    record_parts.append(f"{col}={val}")
            context_parts.append(f"Record {idx+1}: {', '.join(record_parts)}")
        
        self.data_context = "\n".join(context_parts)
        print("Complete data context created with all records")
    
    def create_chunks(self):
        """Create text chunks from the dataframe"""
        self.chunks = []
        
        self.chunks.append({
            'content': self.data_context,
            'type': 'complete_dataset',
            'source': 'full_data_analysis'
        })
        
        return len(self.chunks)

class IntelligentAIAnalyst:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.df = None
        self.update_dataframe()
    
    def update_dataframe(self):
        """Update the dataframe reference"""
        self.df = self.data_processor.df
        if self.df is not None:
            print(f"AI Analyst: DataFrame updated - Shape: {self.df.shape}")
            print(f"AI Analyst: Columns: {list(self.df.columns)}")
        else:
            print("AI Analyst: No DataFrame available")
    
    def process_query(self, question):
        """Process query with pure AI intelligence - no hardcoded patterns"""
        try:
            print(f"AI Analyst processing: {question}")
            
            # Update dataframe reference
            self.update_dataframe()
            
            # Check if dataframe is available
            if self.df is None or len(self.df) == 0:
                return {
                    'success': False,
                    'error': 'No data available. Please upload a dataset first.'
                }
            
            # Let AI handle everything - no hardcoded logic
            return self.ai_pure_analysis(question)
                
        except Exception as e:
            print(f"Error in AI processing: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'error': f"I encountered an error analyzing your data: {str(e)}"
            }
    
    def ai_pure_analysis(self, question):
        """Pure AI analysis - AI determines everything dynamically"""
        try:
            # Get complete data context
            data_context = self.data_processor.data_context
            
            # Let AI analyze the question and determine what data to return
            analysis_prompt = f"""
You are an expert data analyst. You have been provided with a complete dataset and the user is asking a question.

COMPLETE DATASET WITH ALL RECORDS:
{data_context}

USER QUESTION: {question}

Your task:
1. Analyze the user's question and understand what they want
2. Look through the complete dataset provided above to find relevant information
3. Answer the question based on the actual data shown
4. If they're asking about specific records, people, values, or patterns - find them in the actual data
5. Provide a natural, conversational response as a data analyst would

Important:
- Use ONLY the actual data provided above
- Don't make assumptions about data not shown
- Be specific with actual values, names, and numbers from the dataset
- If looking for specific records, search through all the records provided
- Provide insights based on the real data patterns you see

Respond naturally as if you've thoroughly examined this specific dataset.
"""

            # Get AI response
            ai_response = self.call_gemini_api(analysis_prompt)
            
            # Let AI determine what supporting data to show
            supporting_data = self.ai_determine_supporting_data(question, ai_response)
            
            # Let AI determine if visualization is needed
            chart_config = self.ai_determine_visualization(question, ai_response)
            
            return {
                'success': True,
                'answer': ai_response,
                'query': None,  # No SQL needed - AI handles everything
                'result': supporting_data,
                'chart_config': chart_config,
                'analysis_type': 'pure_ai_analysis'
            }
            
        except Exception as e:
            print(f"Error in pure AI analysis: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'error': f"I couldn't analyze your data: {str(e)}"
            }
    
    def ai_determine_supporting_data(self, question, ai_response):
        """Let AI determine what supporting data to show"""
        try:
            # Create prompt for AI to determine supporting data
            data_prompt = f"""
Based on this question: "{question}"
And this analysis response: "{ai_response}"

Look at the dataset and determine what specific records or data should be shown to support the analysis.

DATASET:
{self.data_processor.data_context}

Return the specific records or data that are most relevant to the question in JSON format.
If it's about specific people/items, return those records.
If it's about analysis/statistics, return relevant aggregated data.
If it's about patterns, return representative data.

Return as a JSON array of objects.
"""

            try:
                ai_data_response = self.call_gemini_api(data_prompt)
                
                # Try to parse JSON from AI response
                import json
                # Clean response and extract JSON
                clean_response = ai_data_response.strip()
                if '```json' in clean_response:
                    clean_response = clean_response.split('```json')[1].split('```')[0].strip()
                elif '```' in clean_response:
                    clean_response = clean_response.split('```')[1].strip()
                
                # Find JSON array in response
                json_match = re.search(r'\[.*\]', clean_response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                
                # If AI couldn't determine, use smart fallback
                return self.smart_data_fallback(question)
                
            except Exception as e:
                print(f"AI data determination failed: {e}")
                return self.smart_data_fallback(question)
                
        except Exception as e:
            print(f"Error determining supporting data: {e}")
            return None
    
    def smart_data_fallback(self, question):
        """Smart fallback when AI can't determine data - still no hardcoding"""
        try:
            question_lower = question.lower()
            
            # Dynamic pattern detection without hardcoded values
            words = question_lower.split()
            
            # Check if question contains any text values from the dataset
            text_cols = self.df.select_dtypes(include=['object']).columns
            for col in text_cols:
                unique_values = self.df[col].dropna().unique()
                for value in unique_values:
                    if str(value).lower() in question_lower:
                        # Found a match - return records containing this value
                        matching_records = self.df[self.df[col].str.contains(str(value), case=False, na=False)]
                        if len(matching_records) > 0:
                            return matching_records.to_dict('records')
            
            # Check for numeric values mentioned in question
            numbers = re.findall(r'\b\d+\.?\d*\b', question)
            if numbers:
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                for num_str in numbers:
                    num_val = float(num_str)
                    for col in numeric_cols:
                        # Find records with this numeric value
                        matching_records = self.df[self.df[col] == num_val]
                        if len(matching_records) > 0:
                            return matching_records.to_dict('records')
            
            # If question asks about analysis/overview, return summary
            analysis_words = ['analyze', 'analysis', 'overview', 'summary', 'insight', 'pattern']
            if any(word in question_lower for word in analysis_words):
                # Return basic stats for numeric columns
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats_data = []
                    for col in numeric_cols:
                        desc = self.df[col].describe()
                        stats_data.append({
                            'column': col,
                            'count': int(desc['count']),
                            'mean': round(desc['mean'], 2),
                            'min': desc['min'],
                            'max': desc['max']
                        })
                    return stats_data
            
            # Default: return first few records
            return self.df.head(10).to_dict('records')
            
        except Exception as e:
            print(f"Smart fallback error: {e}")
            return self.df.head(5).to_dict('records')
    
    def ai_determine_visualization(self, question, ai_response):
        """Let AI determine if visualization is needed and what type"""
        try:
            question_lower = question.lower()
            
            # Simple check for visualization keywords
            viz_words = ['visualize', 'chart', 'graph', 'plot', 'show']
            needs_viz = any(word in question_lower for word in viz_words)
            
            if not needs_viz:
                return None
            
            # Create appropriate visualization based on data structure
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                # Bar chart: categorical vs numeric
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                
                grouped_data = self.df.groupby(cat_col)[num_col].mean().round(2)
                
                if len(grouped_data) <= 15:  # Reasonable number for visualization
                    return {
                        'type': 'bar',
                        'data': {
                            'labels': grouped_data.index.tolist(),
                            'datasets': [{
                                'label': f'{num_col.replace("_", " ").title()}',
                                'data': grouped_data.values.tolist(),
                                'backgroundColor': 'rgba(59, 130, 246, 0.6)',
                                'borderColor': 'rgba(59, 130, 246, 1)',
                                'borderWidth': 1
                            }]
                        },
                        'options': {
                            'responsive': True,
                            'plugins': {
                                'legend': {'display': True},
                                'title': {
                                    'display': True,
                                    'text': f'{num_col.replace("_", " ").title()} by {cat_col.replace("_", " ").title()}'
                                }
                            },
                            'scales': {
                                'y': {'beginAtZero': True}
                            }
                        }
                    }
            
            elif len(categorical_cols) > 0:
                # Pie chart for categorical distribution
                cat_col = categorical_cols[0]
                value_counts = self.df[cat_col].value_counts()
                
                if len(value_counts) <= 10:
                    return {
                        'type': 'pie',
                        'data': {
                            'labels': value_counts.index.tolist(),
                            'datasets': [{
                                'data': value_counts.values.tolist(),
                                'backgroundColor': [
                                    'rgba(255, 99, 132, 0.8)',
                                    'rgba(54, 162, 235, 0.8)',
                                    'rgba(255, 205, 86, 0.8)',
                                    'rgba(75, 192, 192, 0.8)',
                                    'rgba(153, 102, 255, 0.8)',
                                    'rgba(255, 159, 64, 0.8)'
                                ]
                            }]
                        },
                        'options': {
                            'responsive': True,
                            'plugins': {
                                'legend': {'position': 'bottom'},
                                'title': {
                                    'display': True,
                                    'text': f'{cat_col.replace("_", " ").title()} Distribution'
                                }
                            }
                        }
                    }
            
            return None
            
        except Exception as e:
            print(f"Visualization determination error: {e}")
            return None
    
    def call_gemini_api(self, prompt):
        """Call Gemini API with robust error handling"""
        headers = {
            'Content-Type': 'application/json',
        }
        
        data = {
            'contents': [{
                'parts': [{
                    'text': prompt
                }]
            }],
            'generationConfig': {
                'temperature': 0.1,  # Lower temperature for more consistent analysis
                'topP': 0.8,
                'topK': 40,
                'maxOutputTokens': 3000,  # Increased for complete analysis
            }
        }
        
        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                headers=headers,
                json=data,
                timeout=45  # Increased timeout for complex analysis
            )
            
            print(f"Gemini API Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    return content
                else:
                    raise Exception("No candidates in Gemini response")
            else:
                print(f"Gemini API Error: {response.text}")
                raise Exception(f"Gemini API error: {response.status_code}")
                
        except Exception as e:
            print(f"Gemini API call failed: {e}")
            raise e

# Global instances
data_processor = DataProcessor()
ai_analyst = IntelligentAIAnalyst(data_processor)

def init_db():
    """Initialize SQLite database for chat history"""
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        print(f"File saved to: {file_path}")
        
        if data_processor.load_data(file_path):
            ai_analyst.update_dataframe()
            print("Data loaded successfully - completely dynamic processing")
            return jsonify({'success': True, 'message': 'File uploaded successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to process file'})

@app.route('/status')
def get_status():
    """Get processing status"""
    if data_processor.df is not None:
        print(f"Status check: DataFrame exists with shape {data_processor.df.shape}")
        
        if len(data_processor.chunks) == 0:
            chunks_created = data_processor.create_chunks()
            insights = generate_dynamic_insights()
            
            return jsonify({
                'stage': 'complete',
                'progress': 100,
                'dataset_info': data_processor.metadata,
                'chunks_info': {
                    'total_chunks': chunks_created,
                    'embeddings_created': len(data_processor.chunks)
                },
                'insights': insights
            })
        else:
            return jsonify({
                'stage': 'complete',
                'progress': 100,
                'dataset_info': data_processor.metadata,
                'chunks_info': {
                    'total_chunks': len(data_processor.chunks),
                    'embeddings_created': len(data_processor.chunks)
                }
            })
    else:
        print("Status check: No DataFrame available")
        return jsonify({
            'stage': 'waiting',
            'progress': 0
        })

def generate_dynamic_insights():
    """Generate insights dynamically based on actual data"""
    insights = []
    df = data_processor.df
    
    insights.append(f"Dataset loaded: {len(df)} rows × {len(df.columns)} columns")
    
    # Dynamic column type analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    text_cols = df.select_dtypes(include=['object']).columns
    
    if len(numeric_cols) > 0:
        insights.append(f"Numeric columns detected: {len(numeric_cols)}")
    
    if len(text_cols) > 0:
        insights.append(f"Text/categorical columns detected: {len(text_cols)}")
    
    # Data quality check
    missing_data = df.isnull().sum().sum()
    if missing_data == 0:
        insights.append("Clean dataset - no missing values")
    else:
        insights.append(f"Dataset has {missing_data} missing values")
    
    # Dynamic uniqueness analysis
    for col in text_cols[:2]:  # Check first 2 text columns
        unique_count = df[col].nunique()
        total_count = len(df)
        if unique_count == total_count:
            insights.append(f"Column '{col}' has unique values (good for identification)")
        elif unique_count < total_count * 0.1:
            insights.append(f"Column '{col}' has few categories (good for grouping)")
    
    return insights[:6]

@app.route('/chat/new', methods=['POST'])
def create_new_chat():
    """Create a new chat session"""
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

@app.route('/chat/history')
def get_chat_history():
    """Get chat history"""
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
        
        if row[4]:
            chats[chat_id]['messages'].append({
                'type': row[4],
                'content': row[5],
                'timestamp': row[6]
            })
    
    return jsonify({'chats': list(chats.values())})

@app.route('/chat/<chat_id>')
def get_chat(chat_id):
    """Get specific chat"""
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
        'messages': [
            {
                'type': row[0],
                'content': row[1],
                'query': row[2],
                'result': json.loads(row[3]) if row[3] else None,
                'timestamp': row[4]
            }
            for row in message_rows
        ]
    }
    
    return jsonify({'chat': chat})

@app.route('/query', methods=['POST'])
def process_query():
    """Process user query using pure AI with dynamic data"""
    data = request.get_json()
    question = data.get('question', '')
    chat_id = data.get('chat_id')
    
    print(f"Processing dynamic query: {question}")
    
    if not question:
        return jsonify({'success': False, 'error': 'No question provided'})
    
    if data_processor.df is None:
        return jsonify({'success': False, 'error': 'No dataset loaded. Please upload a file first.'})
    
    print(f"Processing with dynamic dataframe shape: {data_processor.df.shape}")
    print(f"Dynamic columns: {list(data_processor.df.columns)}")
    
    # Process the query using pure AI analysis
    result = ai_analyst.process_query(question)
    
    if result['success']:
        # Save to database
        conn = sqlite3.connect('data/chat_history.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO messages (chat_id, type, content, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (chat_id, 'user', question, datetime.now()))
        
        cursor.execute('''
            INSERT INTO messages (chat_id, type, content, query, result, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (chat_id, 'assistant', result['answer'], result.get('query'), 
              json.dumps(result.get('result')), datetime.now()))
        
        cursor.execute('''
            UPDATE chats SET title = ?, updated_at = ?
            WHERE id = ? AND title = 'New Analysis'
        ''', (question[:50] + '...' if len(question) > 50 else question, 
              datetime.now(), chat_id))
        
        conn.commit()
        conn.close()
    
    return jsonify(result)

if __name__ == '__main__':
    # Get port and debug settings from environment
    port = int(os.getenv('PORT', 8080))
    debug = os.getenv('DEBUG', 'True').lower() == 'true'
    host = os.getenv('HOST', '0.0.0.0')
    
    print(f"Starting server on {host}:{port} (debug={debug})")
    app.run(debug=debug, host=host, port=port)