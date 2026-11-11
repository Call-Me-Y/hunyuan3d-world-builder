from flask import Flask, render_template, url_for, request, jsonify, send_file, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix
import requests
import json
import os
import base64
import uuid
import sqlite3
from datetime import datetime
import re
from dotenv import load_dotenv

# 获取当前文件的目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 项目根目录与本地数据目录
PROJECT_ROOT = os.path.dirname(BASE_DIR)
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, 
            static_folder=STATIC_FOLDER,
            static_url_path='/static',
            template_folder=os.path.join(BASE_DIR, 'templates'))

# 配置 ProxyFix 以正确处理反向代理（HTTPS）
# 这会让 Flask 正确识别 X-Forwarded-Proto 等头部
app.wsgi_app = ProxyFix(
    app.wsgi_app,
    x_for=1,
    x_proto=1,
    x_host=1,
    x_port=1,
    x_prefix=1
)

# API server base URL
API_BASE_URL = os.getenv('API_BASE_URL', 'http://127.0.0.1:8081')

# Directory paths
TEMP_DIR = os.path.join(PROJECT_ROOT, 'temp')
UPLOADS_DIR = os.path.join(PROJECT_ROOT, 'uploads')
SCENE_SAVE_DIR = os.path.join(PROJECT_ROOT, 'save')

# Database path
DB_PATH = os.path.join(PROJECT_ROOT, 'database', 'models.db')

def init_database():
    """Initialize the database and create tables if they don't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create models table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            category TEXT NOT NULL,
            name TEXT,
            file_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(session_id, category)
        )
    ''')
    
    conn.commit()
    conn.close()

def save_model_to_db(session_id, category, name, file_url):
    """
    Save model information to database
    
    Args:
        session_id: Session ID
        category: 'item' or 'scene'
        name: Model name
        file_url: URL or path to the model file
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Insert or replace (update if exists)
        cursor.execute('''
            INSERT OR REPLACE INTO models (session_id, category, name, file_url, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, category, name, file_url, datetime.now()))
        
        conn.commit()
        conn.close()
        print(f"Model saved to database: {session_id}, {category}, {name}, {file_url}")
    except Exception as e:
        print(f"Error saving model to database: {e}")

def get_models_from_db(category=None, limit=None):
    """
    Get models from database
    
    Args:
        category: Filter by category ('item' or 'scene'), None for all
        limit: Maximum number of results
    
    Returns:
        List of model dictionaries
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        if category:
            cursor.execute('''
                SELECT id, session_id, category, name, file_url, created_at
                FROM models
                WHERE category = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (category, limit or 100))
        else:
            cursor.execute('''
                SELECT id, session_id, category, name, file_url, created_at
                FROM models
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit or 100,))
        
        results = cursor.fetchall()
        conn.close()
        
        models = []
        for row in results:
            models.append({
                'id': row[0],
                'session_id': row[1],
                'category': row[2],
                'name': row[3],
                'file_url': row[4],
                'created_at': row[5]
            })
        
        return models
    except Exception as e:
        print(f"Error getting models from database: {e}")
        return []

def delete_model_from_db(session_id):
    """
    Delete model from database by session_id and delete related files
    
    Args:
        session_id: Session ID of the model to delete
    
    Returns:
        True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # First, get the file_url before deleting
        cursor.execute('''
            SELECT file_url FROM models
            WHERE session_id = ?
        ''', (session_id,))
        
        result = cursor.fetchone()
        file_url = result[0] if result else None
        
        # Delete from database
        cursor.execute('''
            DELETE FROM models
            WHERE session_id = ?
        ''', (session_id,))
        
        conn.commit()
        conn.close()
        
        # Delete related files
        if file_url:
            delete_model_files(session_id, file_url)
        
        print(f"Model deleted from database: {session_id}")
        return True
    except Exception as e:
        print(f"Error deleting model from database: {e}")
        return False

def delete_model_files(session_id, file_url):
    """
    Delete files associated with a model including:
    - White model files (e.g., {session_id}_white.glb)
    - Textured model files (e.g., {session_id}_textured.glb)
    - Any other files with the same session_id prefix
    
    Args:
        session_id: Session ID of the model
        file_url: Primary file URL from database
    """
    try:
        # Delete files from both temp and uploads directories
        search_dirs = [TEMP_DIR, UPLOADS_DIR]
        deleted_count = 0
        
        for dir_path in search_dirs:
            if not os.path.exists(dir_path):
                continue
                
            try:
                # List all files in directory
                all_files = os.listdir(dir_path)
                
                # Delete all files starting with session_id (includes white model, textured model, etc.)
                for filename in all_files:
                    if filename.startswith(f"{session_id}_"):
                        file_path = os.path.join(dir_path, filename)
                        if os.path.isfile(file_path):
                            try:
                                os.remove(file_path)
                                deleted_count += 1
                                # Identify file type for better logging
                                if "_white" in filename:
                                    file_type = "white model"
                                elif "_textured" in filename:
                                    file_type = "textured model"
                                else:
                                    file_type = "model file"
                                print(f"Deleted {file_type}: {file_path}")
                            except Exception as e:
                                print(f"Error deleting file {file_path}: {e}")
                
            except Exception as e:
                print(f"Error accessing directory {dir_path}: {e}")
        
        print(f"Total {deleted_count} files deleted for session_id: {session_id}")
                
    except Exception as e:
        print(f"Error deleting model files: {e}")

def update_model_name_in_db(session_id, new_name, new_category=None):
    """
    Update model name and category in database
    
    Args:
        session_id: Session ID of the model
        new_name: New name for the model
        new_category: New category for the model (optional)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        if new_category:
            cursor.execute('''
                UPDATE models
                SET name = ?, category = ?
                WHERE session_id = ?
            ''', (new_name, new_category, session_id))
            print(f"Model updated: {session_id}, new name: {new_name}, new category: {new_category}")
        else:
            cursor.execute('''
                UPDATE models
                SET name = ?
                WHERE session_id = ?
            ''', (new_name, session_id))
            print(f"Model name updated: {session_id}, new name: {new_name}")
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating model: {e}")
        return False

# Initialize database on startup
init_database()
os.makedirs(SCENE_SAVE_DIR, exist_ok=True)

# 添加上下文处理器，确保在 HTTPS 环境下生成正确的 URL
@app.context_processor
def inject_scheme():
    """注入 URL scheme，确保在反向代理后正确识别 HTTPS"""
    return {
        'url_scheme': request.scheme if request.is_secure or request.headers.get('X-Forwarded-Proto') == 'https' else 'http'
    }

# 确保静态文件路由正常工作
@app.route('/static/<path:filename>')
def static_files(filename):
    """显式处理静态文件请求"""
    try:
        # 移除前导斜杠并规范化路径，防止路径遍历攻击
        filename = filename.lstrip('/')
        # 防止路径遍历攻击（包含 .. 或绝对路径）
        if '..' in filename or os.path.isabs(filename):
            print(f"Invalid filename (path traversal attempt): {filename}")
            return jsonify({'error': 'Invalid path'}), 403
        
        # 使用Flask的send_from_directory，它已经内置了安全检查
        return send_from_directory(app.static_folder, filename)
    except Exception as e:
        print(f"Error serving static file {filename}: {e}")
        import traceback
        traceback.print_exc()
        # 如果文件不存在，send_from_directory会自动返回404
        # 这里只处理其他类型的错误
        return jsonify({'error': str(e)}), 500

@app.route('/favicon.ico')
def favicon():
    """处理favicon请求"""
    try:
        favicon_path = os.path.join(app.static_folder, 'icon_no_bg.ico')
        if os.path.exists(favicon_path):
            return send_file(favicon_path, mimetype='image/x-icon')
        else:
            return '', 204  # No Content
    except Exception as e:
        print(f"Error serving favicon: {e}")
        return '', 404

@app.route('/')
def index():
    """主页路由"""
    return render_template('index.html')

@app.route('/gen')
def gen():
    """3D模型生成页面路由"""
    return render_template('gen.html')

@app.route('/reveal')
def reveal():
    return render_template('reveal.html')

@app.route('/AR')
def AR():
    return render_template('AR.html')

@app.route('/space')
def space():
    return render_template('space.html')

@app.route('/api/check-queue', methods=['POST'])
def check_queue():
    """检查队列状态"""
    try:
        # 简化的队列检查，实际可以调用 api_server 的队列状态
        response = requests.post(f'{API_BASE_URL}/health')
        if response.status_code == 200:
            return jsonify({'queue_count': 0})
        return jsonify({'queue_count': 0})
    except Exception as e:
        return jsonify({'queue_count': 0, 'error': str(e)})

@app.route('/api/main/text_to_model', methods=['POST'])
def text_to_model():
    """文生模型接口"""
    try:
        data = request.get_json()
        response = requests.post(f'{API_BASE_URL}/api/main/text_to_model', json=data)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/main/single_image_to_model', methods=['POST'])
def single_image_to_model():
    """单图生模型接口"""
    try:
        # 接收JSON格式的请求
        data = request.get_json()
        
        if data is None:
            return jsonify({'status': 'error', 'message': 'Request body must be JSON'}), 400
        
        # 检查必要的字段
        if 'image_base64' not in data:
            return jsonify({'status': 'error', 'message': 'No image_base64 provided'}), 400
        
        # 获取参数，使用前端传递的session_id或生成新的
        image_base64 = data.get('image_base64')
        quality = data.get('quality', 'standard')
        # 使用前端传递的session_id，如果没有则生成新的
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        # 清理和验证base64字符串
        if not image_base64:
            return jsonify({'status': 'error', 'message': 'Empty image_base64 provided'}), 400
        
        # 移除所有空白字符（空格、换行、制表符等）
        image_base64 = ''.join(image_base64.split())
        
        # 确保长度能被4整除，如果不能则添加填充
        padding = len(image_base64) % 4
        if padding:
            image_base64 += '=' * (4 - padding)
        
        # 验证base64格式
        try:
            base64.b64decode(image_base64, validate=True)
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Invalid base64 format: {str(e)}'}), 400
        
        # 准备发送到API的数据
        api_data = {
            'image_base64': image_base64,
            'quality': quality,
            'session_id': session_id
        }
        
        # 获取并传递 name 参数（如果前端提供了）
        name = data.get('name')
        if name:
            api_data['name'] = name
        
        # 发送到API服务器（异步处理，不等待结果）
        requests.post(f'{API_BASE_URL}/api/main/single_image_to_model', json=api_data)
        
        # 返回session_id给前端
        return jsonify({
            'status': 'processing',
            'session_id': session_id,
            'message': 'Image is being processed'
        }), 200
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/main/multi_image_to_model', methods=['POST'])
def multi_image_to_model():
    """多图生模型接口"""
    try:
        data = request.get_json()
        response = requests.post(f'{API_BASE_URL}/api/main/multi_image_to_model', json=data)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/main/images_to_scene', methods=['POST'])
def images_to_scene():
    """多图生场景接口"""
    try:
        data = request.get_json()
        response = requests.post(f'{API_BASE_URL}/api/main/images_to_scene', json=data)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session_files(session_id):
    """获取会话文件列表接口"""
    try:
        files = []
        # 定义搜索目录
        search_dirs = list(dict.fromkeys([TEMP_DIR, UPLOADS_DIR]))
        
        for dir_path in search_dirs:
            if os.path.exists(dir_path):
                all_files = os.listdir(dir_path)
                for filename in all_files:
                    if filename.startswith(f"{session_id}_"):
                        full_path = os.path.join(dir_path, filename)
                        if os.path.isfile(full_path):
                            # Determine file type and label
                            file_type = "model"
                            label = filename
                            if "_white.glb" in filename or "_white.gltf" in filename:
                                label = "白模"
                            elif "_textured.glb" in filename or "_textured.gltf" in filename:
                                label = "最终结果"
                            elif ".png" in filename or ".jpg" in filename or ".jpeg" in filename:
                                file_type = "image"
                                label = "生成图片"
                            
                            files.append({
                                "filename": filename,
                                "path": filename,
                                "type": file_type,
                                "label": label,
                                "size": os.path.getsize(full_path)
                            })
        
        return jsonify({"session_id": session_id, "files": files}), 200
    except Exception as e:
        print(f"Error getting session files: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/database/save', methods=['POST'])
def save_model():
    """保存模型信息到数据库"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        category = data.get('category')  # 'item' or 'scene'
        name = data.get('name', '')
        file_url = data.get('file_url', '')
        
        if not session_id or not category:
            return jsonify({'status': 'error', 'message': 'session_id and category are required'}), 400
        
        save_model_to_db(session_id, category, name, file_url)
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/database/models', methods=['GET'])
def get_models():
    """从数据库获取模型列表"""
    try:
        category = request.args.get('category')  # Optional: 'item' or 'scene'
        limit = request.args.get('limit', type=int)
        
        models = get_models_from_db(category=category, limit=limit)
        return jsonify({'status': 'success', 'models': models}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/database/models/delete', methods=['POST'])
def delete_model():
    """从数据库删除模型"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'status': 'error', 'message': 'session_id is required'}), 400
        
        success = delete_model_from_db(session_id)
        if success:
            return jsonify({'status': 'success'}), 200
        else:
            return jsonify({'status': 'error', 'message': 'Failed to delete model'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/database/models/update', methods=['POST'])
def update_model():
    """更新模型名称和类别"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        new_name = data.get('name', '')
        new_category = data.get('category')
        
        if not session_id:
            return jsonify({'status': 'error', 'message': 'session_id is required'}), 400
        
        success = update_model_name_in_db(session_id, new_name, new_category)
        if success:
            return jsonify({'status': 'success'}), 200
        else:
            return jsonify({'status': 'error', 'message': 'Failed to update model'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/file/temp/<path:file_path>')
def download_temp_file(file_path):
    """从temp目录下载文件"""
    try:
        return send_from_directory(TEMP_DIR, file_path)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/file/uploads/<path:file_path>')
def download_uploads_file(file_path):
    """从uploads目录下载文件"""
    try:
        return send_from_directory(UPLOADS_DIR, file_path)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/file/<path:file_path>')
def download_file(file_path):
    """下载文件接口（自动尝试temp和uploads目录，完全由app.py处理）"""
    try:
        # Try temp directory first
        temp_path = os.path.join(TEMP_DIR, file_path)
        if os.path.exists(temp_path):
            return send_from_directory(TEMP_DIR, file_path)
        
        # Try uploads directory
        uploads_path = os.path.join(UPLOADS_DIR, file_path)
        if os.path.exists(uploads_path):
            return send_from_directory(UPLOADS_DIR, file_path)
        
        # 文件不存在，返回错误（不再fallback到api_server）
        return jsonify({'status': 'error', 'message': 'File not found'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/space/saved-scenes', methods=['GET'])
def list_saved_scenes():
    """获取已保存的场景列表"""
    try:
        os.makedirs(SCENE_SAVE_DIR, exist_ok=True)
        scene_files = []
        for filename in os.listdir(SCENE_SAVE_DIR):
            if not filename.endswith('.json'):
                continue
            file_path = os.path.join(SCENE_SAVE_DIR, filename)
            if not os.path.isfile(file_path):
                continue
            scene_files.append({
                'filename': filename,
                'name': filename[:-5],
                'updated_at': os.path.getmtime(file_path),
                'size': os.path.getsize(file_path)
            })

        scene_files.sort(key=lambda x: x['updated_at'], reverse=True)
        return jsonify({'status': 'success', 'scenes': scene_files}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/space/save', methods=['POST'])
def save_scene_json():
    """保存场景文件到save目录"""
    try:
        data = request.get_json() or {}
        scene_name = (data.get('name') or '').strip()
        scene_data = data.get('scene')

        if not scene_name:
            return jsonify({'status': 'error', 'message': 'name is required'}), 400
        if scene_data is None:
            return jsonify({'status': 'error', 'message': 'scene is required'}), 400

        safe_name = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '_', scene_name)
        if not safe_name:
            safe_name = f"scene_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        os.makedirs(SCENE_SAVE_DIR, exist_ok=True)
        filename = f"{safe_name}.json"
        save_path = os.path.join(SCENE_SAVE_DIR, filename)

        payload = {
            'name': scene_name,
            'saved_at': datetime.now().isoformat(),
            'scene': scene_data
        }
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return jsonify({'status': 'success', 'filename': filename}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/space/load/<path:filename>', methods=['GET'])
def load_scene_json(filename):
    """读取已保存场景JSON"""
    try:
        filename = os.path.basename(filename)
        if not filename.endswith('.json'):
            return jsonify({'status': 'error', 'message': 'Only .json files are allowed'}), 400

        os.makedirs(SCENE_SAVE_DIR, exist_ok=True)
        file_path = os.path.join(SCENE_SAVE_DIR, filename)
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': 'Scene file not found'}), 404

        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        return jsonify({'status': 'success', 'data': content}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/space/delete/<path:filename>', methods=['DELETE'])
def delete_scene_json(filename):
    """删除已保存场景JSON"""
    try:
        filename = os.path.basename(filename)
        if not filename.endswith('.json'):
            return jsonify({'status': 'error', 'message': 'Only .json files are allowed'}), 400

        os.makedirs(SCENE_SAVE_DIR, exist_ok=True)
        file_path = os.path.join(SCENE_SAVE_DIR, filename)
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': 'Scene file not found'}), 404

        os.remove(file_path)
        return jsonify({'status': 'success', 'message': 'Scene deleted'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/main/upload_model', methods=['POST'])
def upload_model():
    """上传3D模型文件"""
    try:
        # Get file from request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get other parameters
        name = request.form.get('name', '')
        session_id = request.form.get('session_id', '')
        filename = request.form.get('filename', file.filename)
        category = request.form.get('category', 'item')  # Default to 'item' if not provided
        
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
        
        # Ensure uploads directory exists
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        
        # Save file to uploads directory
        file_path = os.path.join(UPLOADS_DIR, filename)
        file.save(file_path)
        
        # Construct file URL
        file_url = filename
        
        # Save to database with category
        save_model_to_db(session_id, category, name, file_url)
        
        return jsonify({
            'status': 'success',
            'message': 'File uploaded successfully',
            'file_url': file_url,
            'session_id': session_id
        }), 200
        
    except Exception as e:
        print(f"Error uploading file: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8082)
