from flask import Flask, render_template, request, jsonify, send_file
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from forecasting import process_data
import tempfile
import json
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Configure CORS to allow requests from any origin
CORS(app, resources={r"/*": {"origins": "*"}}, methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

# Increase maximum file size to 100MB
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Register error handler for 413 Request Entity Too Large
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({
        'status': 'error',
        'message': f'File is too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)}MB.'
    }), 413

# Register error handler for all exceptions
@app.errorhandler(Exception)
def handle_exception(e):
    # Pass through HTTP errors
    if isinstance(e, RequestEntityTooLarge):
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 413
    
    # Log the full error for debugging
    logger.error(f"Server error: {str(e)}")
    logger.error(traceback.format_exc())
    
    # Return JSON response for other errors
    return jsonify({
        'status': 'error',
        'message': f'Server error: {str(e)}'
    }), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'success'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response
    
    try:
        # Check if file is in request
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'status': 'error', 'message': 'No file part'}), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        
        # Check file type
        if file and allowed_file(file.filename):
            # Secure the filename and save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Log successful file save
            logger.info(f"File saved successfully: {filepath}")
            
            try:
                # Process the file
                results = process_data(filepath)
                
                # Log successful processing
                logger.info("File processed successfully")
                logger.info(f"Results keys: {list(results.keys())}")
                logger.info(f"MAPE: {results.get('mape', 'N/A')}")
                logger.info(f"RMSE: {results.get('rmse', 'N/A')}")
                logger.info(f"Accuracy: {results.get('accuracy', 'N/A')}")
                
                # Ensure all required keys are present
                required_keys = ['dates', 'history', 'arima', 'prophet', 'business_impact', 
                               'demand_projections', 'mape', 'rmse', 'accuracy']
                for key in required_keys:
                    if key not in results:
                        logger.warning(f"Missing key in results: {key}")
                        results[key] = None
                
                # Add processing timestamp
                results['processed_at'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Log the results structure before sending
                logger.info("Sending response with structure:")
                logger.info(f"- Status: success")
                logger.info(f"- Results keys: {list(results.keys())}")
                if results.get('arima'):
                    logger.info(f"- ARIMA keys: {list(results['arima'].keys())}")
                if results.get('prophet'):
                    logger.info(f"- Prophet keys: {list(results['prophet'].keys())}")
                
                return jsonify({'status': 'success', 'results': results})
            except Exception as e:
                # Log processing error
                logger.error(f"Error processing file: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({'status': 'error', 'message': f'Processing error: {str(e)}'}), 500
        else:
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'status': 'error', 'message': 'File type not allowed. Please upload CSV or Excel files.'}), 400
    except Exception as e:
        # Log general error
        logger.error(f"Upload error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'status': 'error', 'message': f'Server error: {str(e)}'}), 500

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)