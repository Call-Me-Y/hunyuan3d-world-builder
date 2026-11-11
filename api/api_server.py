# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

"""
Model Worker API Server - Refactored version with atomic and main services
"""
import argparse
import asyncio
import base64
import io
import logging
import os
from dotenv import load_dotenv
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections import deque
from io import BytesIO
from typing import Optional, Dict, List

# Add project root directory to Python path
# Current file is at: api/api_server.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 环境变量：优先项目根目录 .env，其次 api/.env（后者可覆盖，便于本地覆盖）
load_dotenv(os.path.join(project_root, ".env"))
load_dotenv(os.path.join(current_dir, ".env"), override=True)

import numpy as np
import torch
import trimesh
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List as TypingList
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import (
    Hunyuan3DDiTFlowMatchingPipeline,
    FloaterRemover,
    DegenerateFaceRemover,
    FaceReducer,
)

# Database support
import sqlite3
from datetime import datetime

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import volcano engine client
try:
    from server.api.volcano_engine_client import VolcanoEngineClient
    VOLCANO_ENGINE_AVAILABLE = True
except ImportError:
    try:
        from volcano_engine_client import VolcanoEngineClient
        VOLCANO_ENGINE_AVAILABLE = True
    except ImportError:
        VOLCANO_ENGINE_AVAILABLE = False
        logger.warning("火山引擎客户端未安装，将使用本地模型")

# Global variables
# White models (untextured) directory
WHITE_MODEL_DIR = os.path.join(project_root, 'temp')
# Textured models directory
TEXTURED_MODEL_DIR = os.path.join(project_root, 'uploads')
# Legacy temp directory for other files
SAVE_DIR = os.path.join(project_root, 'save')
TEMP_DIR = os.path.join(project_root, 'temp')

os.makedirs(WHITE_MODEL_DIR, exist_ok=True)
os.makedirs(TEXTURED_MODEL_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Database path
DB_PATH = os.path.join(project_root, 'database', 'models.db')
DB_DIR = os.path.dirname(DB_PATH)
os.makedirs(DB_DIR, exist_ok=True)

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
    logger.info("Database initialized")

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
        logger.info(f"Saving model to database - session_id: {session_id}, category: {category}, name: '{name}', file_url: {file_url}")
        
        # Validate inputs - ensure name is not None or empty
        if not name or name.strip() == '':
            logger.warning(f"Empty name provided for session_id {session_id}, using default name")
            name = f"{session_id}_{category}"
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if record exists
        cursor.execute('''
            SELECT id, name FROM models WHERE session_id = ? AND category = ?
        ''', (session_id, category))
        existing = cursor.fetchone()
        
        if existing:
            logger.info(f"Record exists for session_id {session_id}, category {category}. Current name: '{existing[1]}'. Updating with new name: '{name}'")
            cursor.execute('''
                UPDATE models SET name = ?, file_url = ?, created_at = ? WHERE session_id = ? AND category = ?
            ''', (name, file_url, datetime.now(), session_id, category))
        else:
            logger.info(f"Creating new record for session_id {session_id}, category {category}, name: '{name}'")
            cursor.execute('''
                INSERT INTO models (session_id, category, name, file_url, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, category, name, file_url, datetime.now()))
        
        conn.commit()
        conn.close()
        logger.info(f"Model successfully saved to database: {session_id}, {category}, name: '{name}', {file_url}")
    except Exception as e:
        logger.error(f"Error saving model to database: {e}")
        import traceback
        traceback.print_exc()

# Initialize database
init_database()

# =============================
# Pydantic Request Models
# =============================

class TextToImageRequest(BaseModel):
    """Text to Image Request Model"""
    text: str = Field(..., description="Text prompt for image generation")
    quality: str = Field("standard", description="Quality: fast/standard/high")
    use_volcano: bool = Field(True, description="Use volcano engine API")
    volcano_access_key: Optional[str] = Field(None, description="Volcano engine access key")
    volcano_secret_key: Optional[str] = Field(None, description="Volcano engine secret key")


class SingleImageToModelRequest(BaseModel):
    """Single Image to Model Request Model"""
    image_base64: str = Field(..., description="Base64 encoded image")
    quality: str = Field("standard", description="Quality: fast/standard/high")
    session_id: Optional[str] = Field(None, description="Session ID")
    main_service_id: Optional[str] = Field(None, description="Main service ID")
    volcano_access_key: Optional[str] = Field(None, description="Volcano engine access key")
    volcano_secret_key: Optional[str] = Field(None, description="Volcano engine secret key")
    name: Optional[str] = Field(None, description="Model name")


class MultiImageToModelRequest(BaseModel):
    """Multi Image to Model Request Model"""
    images_base64: TypingList[str] = Field(..., description="List of base64 encoded images")
    quality: str = Field("standard", description="Quality: fast/standard/high")
    session_id: Optional[str] = Field(None, description="Session ID")
    main_service_id: Optional[str] = Field(None, description="Main service ID")
    name: Optional[str] = Field(None, description="Model name")


class TextToModelRequest(BaseModel):
    """Text to Model Request Model"""
    text: str = Field(..., description="Text prompt")
    quality: str = Field("standard", description="Quality: fast/standard/high")
    session_id: Optional[str] = Field(None, description="Session ID")
    main_service_id: Optional[str] = Field(None, description="Main service ID")
    use_volcano: bool = Field(True, description="Use volcano engine API")
    volcano_access_key: Optional[str] = Field(None, description="Volcano engine access key")
    volcano_secret_key: Optional[str] = Field(None, description="Volcano engine secret key")
    name: Optional[str] = Field(None, description="Model name")


class MeshToTexturedRequest(BaseModel):
    """Mesh to Textured Model Request Model"""
    mesh_base64: str = Field(..., description="Base64 encoded mesh file")
    image_base64: str = Field(..., description="Base64 encoded texture image")
    main_service_id: Optional[str] = Field(None, description="Main service ID")


class ImagesToSceneRequest(BaseModel):
    """Images to Scene Request Model"""
    images_base64: TypingList[str] = Field(..., description="List of base64 encoded images")
    quality: str = Field("standard", description="Quality: fast/standard/high")
    main_service_id: Optional[str] = Field(None, description="Main service ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    name: Optional[str] = Field(None, description="Model name")

# Queue management
task_queue = asyncio.Queue()
task_results = {}
task_status = {}
task_lock = threading.Lock()

# Model workers (lazy loading)
pipeline_t2i = None
pipeline_single = None
pipeline_multi = None
pipeline_tex = None
pipeline_scene = None
rembg = None
volcano_client = None


def get_volcano_client(access_key=None, secret_key=None):
    """
    Get or create volcano engine client with optional credentials
    
    Args:
        access_key: Optional volcano access key
        secret_key: Optional volcano secret key
    
    Returns:
        VolcanoEngineClient or None if not available
    """
    if not VOLCANO_ENGINE_AVAILABLE:
        return None
    
    try:
        # If credentials provided, create a new client
        if access_key and secret_key:
            return VolcanoEngineClient(access_key=access_key, secret_key=secret_key)
        
        # Use global client (initialized from environment)
        global volcano_client
        if volcano_client is None:
            try:
                volcano_client = VolcanoEngineClient()
            except ValueError as e:
                logger.warning(f"Failed to initialize volcano client: {e}")
                return None
        return volcano_client
    
    except Exception as e:
        logger.warning(f"Failed to get volcano client: {e}")
        return None

# Model paths - 使用本地模型路径
MODEL_ROOT = os.getenv('HY3DGEN_MODELS') or os.path.join(project_root, 'Hunyuan3D-2', 'models')
if not os.path.isabs(MODEL_ROOT):
    MODEL_ROOT = os.path.abspath(os.path.join(project_root, MODEL_ROOT))
os.environ.setdefault('HY3DGEN_MODELS', MODEL_ROOT)

MODEL_PATHS = {
    't2i': 'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled',
    'single': '.',  # 本地模型根目录（HY3DGEN_MODELS）
    'single_subfolder': 'hunyuan3d-dit-v2-0-turbo',  # 单图模型
    'multi': '.',  # 本地模型根目录（HY3DGEN_MODELS）
    'multi_subfolder': 'hunyuan3d-dit-v2-mv-turbo',  # 多图模型
    # tex pipeline 会在 model_path 下拼接多个子目录，必须传真实本地目录
    'tex': MODEL_ROOT,
    'tex_subfolder': 'hunyuan3d-paint-v2-0-turbo',  # 贴图模型
    'scene': MODEL_ROOT,
    'scene_subfolder': 'HunyuanWorld-Mirror',  # 场景模型
}


def clear_gpu_cache():
    """Clear GPU cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(f"GPU memory cleared. Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")


def load_image_from_base64(image_base64: str) -> Image.Image:
    """Load image from base64 string"""
    return Image.open(BytesIO(base64.b64decode(image_base64)))


def load_multi_images_from_base64(images_base64: List[str]) -> Dict[str, Image.Image]:
    """Load multiple images from base64 strings
    
    The returned dictionary uses view tags compatible with MVImageProcessorV2:
    - 'front', 'left', 'back', 'right' for multi-view processing
    """
    # Define the view tags that MVImageProcessorV2 expects
    view_tags = ['front', 'left', 'back', 'right']
    
    images = {}
    for idx, img_base64 in enumerate(images_base64):
        if idx < len(view_tags):
            # Use the correct view tag name instead of 'image_0', 'image_1', etc.
            images[view_tags[idx]] = Image.open(BytesIO(base64.b64decode(img_base64)))
    return images


def get_queue_position(task_id: str) -> int:
    """Get queue position for a task"""
    with task_lock:
        current_tasks = list(task_queue._queue)
        try:
            return current_tasks.index(task_id)
        except ValueError:
            return -1


# =============================
# Atomic Services
# =============================

class AtomicServiceHandler:
    """Handler for atomic services"""
    
    @staticmethod
    def generate_multi_view_images_from_single(
        image_base64: str,
        use_volcano: bool = True
    ) -> List[str]:
        """
        Generate multiple view images from a single image using volcano engine
        Returns: List of image base64 strings (top, front, side views)
        """
        try:
            if not use_volcano or not VOLCANO_ENGINE_AVAILABLE:
                logger.warning("火山引擎不可用，跳过多视图生成")
                return [image_base64]  # Return original image
            
            global volcano_client
            if volcano_client is None:
                volcano_client = VolcanoEngineClient()
            
            view_configs = [
                ("俯视图", "俯视角度，从天花板或天空俯视，地图，缩略图，鸟瞰图，保持场景不变"),
                ("正视图", "正视图，正对墙壁或主要建筑物，保持场景不变"),
                ("侧视图", "侧视图，视角旋转90度，从侧面观看，保持场景不变")
            ]
            
            multi_view_images = []
            for view_name, prompt in view_configs:
                try:
                    logger.info(f"Generating {view_name} using volcano engine")
                    result_image = volcano_client.image_to_image(
                        prompt=prompt,
                        image_base64=image_base64,
                        scale=0.6,
                        seed=-1
                    )
                    
                    # Convert to base64
                    buffered = BytesIO()
                    result_image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    multi_view_images.append(img_base64)
                    logger.info(f"Successfully generated {view_name}")
                except Exception as e:
                    logger.warning(f"Failed to generate {view_name}: {e}")
                    multi_view_images.append(image_base64)  # Use original as fallback
            
            return multi_view_images if multi_view_images else [image_base64]
            
        except Exception as e:
            logger.error(f"Error in multi-view generation: {e}")
            return [image_base64]  # Return original image on error
    
    @staticmethod
    def text_to_image_with_bg_removal(
        text: str,
        quality: str = "standard",
        use_volcano: bool = True,
        volcano_access_key: Optional[str] = None,
        volcano_secret_key: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Atomic Service 1: Text to Image with Background Removal
        Input: text, quality, use_volcano, volcano_access_key, volcano_secret_key, session_id
        Output: image base64 string
        """
        try:
            clear_gpu_cache()
            logger.info("Starting text-to-image service")
            
            # Enhance prompt to ensure complete item generation with clear background separation
            enhanced_prompt = f"{text}，生成该物品，确保图片完整包含物品，物品与背景清晰分离，物体避免有透明的部分，背景不要有任何东西或阴影"
            logger.info(f"Enhanced prompt: {enhanced_prompt}")
            
            image = None
            
            # Try using volcano engine API first if enabled
            if use_volcano and VOLCANO_ENGINE_AVAILABLE:
                try:
                    # Use optional credentials or default client
                    volcano_cli = get_volcano_client(volcano_access_key, volcano_secret_key)
                    
                    if volcano_cli:
                        # Generate image using volcano engine with enhanced prompt
                        image = volcano_cli.text_to_image(prompt=enhanced_prompt, width=1024, height=1024)
                        logger.info("Successfully generated image using volcano engine")
                    else:
                        logger.warning("Volcano client not available, falling back to local model")
                except Exception as ve:
                    logger.warning(f"Failed to use volcano engine: {ve}, falling back to local model")
            
            # Fallback to local model if volcano engine failed or disabled
            if image is None:
                global pipeline_t2i
                if pipeline_t2i is None:
                    from hy3dgen.text2image import HunyuanDiTPipeline
                    pipeline_t2i = HunyuanDiTPipeline(MODEL_PATHS['t2i'], device='cuda')
                
                # Generate image from text with enhanced prompt
                image = pipeline_t2i(enhanced_prompt)
            
            # Remove background
            global rembg
            if rembg is None:
                rembg = BackgroundRemover()
            
            image = rembg(image)
            
            # Save image to temp folder if session_id is provided
            if session_id:
                picture_path = os.path.join(TEMP_DIR, f'{session_id}_picture.png')
                image.save(picture_path)
                logger.info(f"Saved generated picture to {picture_path}")
            
            # Convert to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            clear_gpu_cache()
            logger.info("Text-to-image service completed")
            return img_base64
            
        except Exception as e:
            logger.error(f"Error in text-to-image service: {e}")
            clear_gpu_cache()
            raise
    
    @staticmethod
    def single_image_to_shape(
        image_base64: str,
        quality: str = "standard",
        format: str = "glb",
        progress_callback=None,
        generate_multi_view: bool = False
    ) -> str:
        """
        Atomic Service 2: Single Image to White Shape Model
        Input: image (base64), quality, format
        Output: model file path
        
        Args:
            generate_multi_view: If True, generate multiple view images using volcano engine
        """
        try:
            clear_gpu_cache()
            logger.info("Starting single-image-to-shape service")
            
            # Optionally generate multi-view images
            if generate_multi_view:
                multi_view_images = AtomicServiceHandler.generate_multi_view_images_from_single(
                    image_base64, use_volcano=True
                )
                # Use first image (or original if generation failed)
                image_base64 = multi_view_images[0]
            
            # Load image
            image = load_image_from_base64(image_base64)
            
            # Remove background
            global rembg
            if rembg is None:
                rembg = BackgroundRemover()
            if image.mode == "RGB":
                image = rembg(image)
            
            # Setup parameters based on quality
            quality_params = {
                "fast": {"steps": 3, "resolution": 128},
                "standard": {"steps": 5, "resolution": 256},
                "high": {"steps": 10, "resolution": 384}
            }
            params = quality_params.get(quality, quality_params["standard"])
            
            if progress_callback:
                progress_callback("Loading shape generation model...", 10)
            
            # Lazy load single-image model
            global pipeline_single
            if pipeline_single is None:
                pipeline_single = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    MODEL_PATHS['single'],
                    subfolder=MODEL_PATHS['single_subfolder'],
                    use_safetensors=True,
                    device='cuda',
                )
                pipeline_single.enable_flashvdm(mc_algo='mc')
            
            if progress_callback:
                progress_callback("Generating 3D shape...", 30)
            
            # Generate 3D model
            generator = torch.Generator('cuda').manual_seed(1234)
            outputs = pipeline_single(
                image=image,
                num_inference_steps=params["steps"],
                guidance_scale=5.0,
                octree_resolution=params["resolution"],
                generator=generator,
                output_type='trimesh'
            )
            
            if progress_callback:
                progress_callback("Saving model...", 90)
            
            mesh = outputs[0]
            
            # Save mesh to white model directory
            task_id = str(uuid.uuid4())
            save_path = os.path.join(WHITE_MODEL_DIR, f'{task_id}_white.{format}')
            mesh.export(save_path)
            
            if progress_callback:
                progress_callback("Completed", 100)
            
            clear_gpu_cache()
            logger.info("Single-image-to-shape service completed")
            return save_path
            
        except Exception as e:
            logger.error(f"Error in single-image-to-shape service: {e}")
            clear_gpu_cache()
            raise
    
    @staticmethod
    def multi_image_to_shape(
        images_base64: List[str],
        quality: str = "standard",
        format: str = "glb",
        progress_callback=None
    ) -> str:
        """
        Atomic Service 3: Multiple Images to White Shape Model
        Input: images (base64 list), quality, format
        Output: model file path
        """
        try:
            clear_gpu_cache()
            logger.info("Starting multi-image-to-shape service")
            
            # Load images
            images = load_multi_images_from_base64(images_base64)
            
            # Remove background from each image
            global rembg
            if rembg is None:
                rembg = BackgroundRemover()
            
            processed_images = {}
            for idx, (key, img) in enumerate(images.items()):
                if img.mode == "RGB":
                    img = rembg(img)
                processed_images[key] = img
                
                if progress_callback:
                    progress_callback(f"Processing image {idx+1}/{len(images)}...", (idx+1)*10/len(images))
            
            # Setup parameters based on quality
            quality_params = {
                "fast": {"steps": 3, "resolution": 380},
                "standard": {"steps": 5, "resolution": 380},
                "high": {"steps": 10, "resolution": 512}
            }
            params = quality_params.get(quality, quality_params["standard"])
            
            if progress_callback:
                progress_callback("Loading multi-view shape generation model...", 30)
            
            # Lazy load multi-image model
            global pipeline_multi
            if pipeline_multi is None:
                pipeline_multi = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    MODEL_PATHS['multi'],
                    subfolder=MODEL_PATHS['multi_subfolder'],
                    use_safetensors=True,
                    device='cuda',
                )
            
            if progress_callback:
                progress_callback("Generating 3D shape from multi-view images...", 50)
            
            # Generate 3D model
            generator = torch.Generator('cuda').manual_seed(1234)
            outputs = pipeline_multi(
                image=processed_images,
                num_inference_steps=params["steps"],
                octree_resolution=params["resolution"],
                generator=generator,
                output_type='trimesh'
            )
            
            if progress_callback:
                progress_callback("Saving model...", 90)
            
            mesh = outputs[0]
            
            # Save mesh to white model directory
            task_id = str(uuid.uuid4())
            save_path = os.path.join(WHITE_MODEL_DIR, f'{task_id}_white.{format}')
            mesh.export(save_path)
            
            if progress_callback:
                progress_callback("Completed", 100)
            
            clear_gpu_cache()
            logger.info("Multi-image-to-shape service completed")
            return save_path
            
        except Exception as e:
            logger.error(f"Error in multi-image-to-shape service: {e}")
            clear_gpu_cache()
            raise
    
    @staticmethod
    def mesh_texturing(
        mesh_path: str,
        image_base64: str,
        progress_callback=None
    ) -> str:
        """
        Atomic Service 4: Mesh Texturing
        Input: mesh path, image (base64)
        Output: textured model file path
        """
        try:
            clear_gpu_cache()
            logger.info("Starting mesh texturing service")
            
            # Load mesh
            mesh = trimesh.load(mesh_path)
            
            # Load reference image
            image = load_image_from_base64(image_base64)
            
            if progress_callback:
                progress_callback("Loading texture generation model...", 10)
            
            # Load texture model
            global pipeline_tex
            if pipeline_tex is None:
                from hy3dgen.texgen import Hunyuan3DPaintPipeline
                pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(
                    MODEL_PATHS['tex'],
                    subfolder=MODEL_PATHS['tex_subfolder']
                )
            
            # Post-process mesh
            if progress_callback:
                progress_callback("Processing mesh...", 30)
            
            floater_remover = FloaterRemover()
            degenerate_remover = DegenerateFaceRemover()
            face_reducer = FaceReducer()
            
            mesh = floater_remover(mesh)
            mesh = degenerate_remover(mesh)
            mesh = face_reducer(mesh, max_facenum=40000)
            
            if progress_callback:
                progress_callback("Generating texture...", 50)
            
            # Generate texture
            textured_mesh = pipeline_tex(mesh, image)
            
            if progress_callback:
                progress_callback("Saving textured model...", 90)
            
            # Save textured mesh to textured model directory
            task_id = str(uuid.uuid4())
            save_path = os.path.join(TEXTURED_MODEL_DIR, f'{task_id}_textured.glb')
            textured_mesh.export(save_path)
            
            if progress_callback:
                progress_callback("Completed", 100)
            
            clear_gpu_cache()
            logger.info("Mesh texturing service completed")
            return save_path
            
        except Exception as e:
            logger.error(f"Error in mesh texturing service: {e}")
            clear_gpu_cache()
            raise
    
    @staticmethod
    def scene_generation(
        images_base64: List[str],
        quality: str = "standard",
        progress_callback=None
    ) -> str:
        """
        Atomic Service 5: Scene Generation from Multiple Images
        Input: images (base64 list), quality
        Output: scene model file path
        """
        try:
            clear_gpu_cache()
            logger.info("Starting scene generation service")
            
            # Note: Scene generation is more complex and may require HunyuanWorld-Mirror
            # For now, we'll use multi-view generation as a placeholder
            # This should be replaced with actual scene generation implementation
            
            # Save images temporarily
            temp_images = []
            for idx, img_base64 in enumerate(images_base64):
                img = load_image_from_base64(img_base64)
                temp_path = os.path.join(SAVE_DIR, f'temp_scene_img_{idx}.png')
                img.save(temp_path)
                temp_images.append(temp_path)
            
            if progress_callback:
                progress_callback(f"Processing {len(images_base64)} images...", 30)
            
            # For now, use multi-image generation as a simplified scene
            scene_path = AtomicServiceHandler.multi_image_to_shape(
                images_base64,
                quality,
                "glb",
                progress_callback
            )
            
            # Cleanup temp images
            for temp_path in temp_images:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            # Rename to indicate it's a scene (save to textured directory as it's a complete model)
            task_id = str(uuid.uuid4())
            new_path = os.path.join(TEXTURED_MODEL_DIR, f'{task_id}_scene.glb')
            os.rename(scene_path, new_path)
            
            clear_gpu_cache()
            logger.info("Scene generation service completed")
            return new_path
            
        except Exception as e:
            logger.error(f"Error in scene generation service: {e}")
            clear_gpu_cache()
            raise


# =============================
# Main Services (Combined Atomic Services)
# =============================

class MainServiceHandler:
    """Handler for main services"""
    
    @staticmethod
    def text_to_model_with_queue(
        text: str,
        quality: str = "standard",
        main_service_id: str = None,
        session_id: str = None,
        use_volcano: bool = True,
        volcano_access_key: Optional[str] = None,
        volcano_secret_key: Optional[str] = None,
        name: str = None
    ) -> Dict:
        """Main Service 1: Text to Model"""
        if main_service_id is None:
            main_service_id = str(uuid.uuid4())
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        logger.info(f"text_to_model_with_queue called - session_id: {session_id}, name: '{name}'")
        
        queue_position = get_queue_position(main_service_id)
        
        try:
            # Step 1: Text to image with bg removal
            img_base64 = AtomicServiceHandler.text_to_image_with_bg_removal(
                text, quality, use_volcano=use_volcano,
                volcano_access_key=volcano_access_key,
                volcano_secret_key=volcano_secret_key,
                session_id=session_id
            )
            
            # Step 2: Image to white shape and save
            temp_mesh_path = AtomicServiceHandler.single_image_to_shape(img_base64, quality, "glb")
            white_mesh_path = os.path.join(WHITE_MODEL_DIR, f'{session_id}_white.glb')
            shutil.copy(temp_mesh_path, white_mesh_path)
            os.remove(temp_mesh_path)
            logger.info(f"White mesh saved to {white_mesh_path}")
            
            # Clear GPU memory after shape generation
            global pipeline_single
            if pipeline_single is not None:
                del pipeline_single
                pipeline_single = None
            clear_gpu_cache()
            logger.info("Shape generation model unloaded, GPU memory cleared")
            
            # Step 3: Generate texture
            temp_textured = AtomicServiceHandler.mesh_texturing(white_mesh_path, img_base64)
            textured_mesh_path = os.path.join(TEXTURED_MODEL_DIR, f'{session_id}_textured.glb')
            shutil.copy(temp_textured, textured_mesh_path)
            os.remove(temp_textured)
            logger.info(f"Textured mesh saved to {textured_mesh_path}")
            
            # Save to database with provided name or default name
            file_url = f"{session_id}_textured.glb"
            model_name = name if name else f"{session_id}_text_item"
            logger.info(f"About to save to database - session_id: {session_id}, model_name: '{model_name}'")
            save_model_to_db(session_id, "item", model_name, file_url)
            
            return {
                "main_service_id": main_service_id,
                "session_id": session_id,
                "queue_position": queue_position,
                "status": "completed",
                "result": {
                    "white_mesh": f"{session_id}_white.glb",
                    "textured_mesh": f"{session_id}_textured.glb"
                }
            }
        except Exception as e:
            return {
                "main_service_id": main_service_id,
                "session_id": session_id,
                "queue_position": queue_position,
                "error": str(e)
            }
    
    @staticmethod
    def single_image_to_model_with_queue(
        image_base64: str,
        quality: str = "standard",
        main_service_id: str = None,
        session_id: str = None,
        name: str = None
    ) -> Dict:
        """Main Service 2: Single Image to Model"""
        if main_service_id is None:
            main_service_id = str(uuid.uuid4())
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        logger.info(f"single_image_to_model_with_queue called - session_id: {session_id}, name: '{name}'")
        
        queue_position = get_queue_position(main_service_id)
        
        try:
            # Step 1: Image to white shape and save to white model directory
            white_mesh_path = os.path.join(WHITE_MODEL_DIR, f'{session_id}_white.glb')
            temp_mesh_path = AtomicServiceHandler.single_image_to_shape(image_base64, quality, "glb")
            
            # Copy to white model directory
            shutil.copy(temp_mesh_path, white_mesh_path)
            os.remove(temp_mesh_path)  # Remove original temp file
            logger.info(f"White mesh saved to {white_mesh_path}")
            
            # Clear GPU memory after shape generation
            global pipeline_single
            if pipeline_single is not None:
                del pipeline_single
                pipeline_single = None
            clear_gpu_cache()
            logger.info("Shape generation model unloaded, GPU memory cleared")
            
            # Step 2: Generate texture using saved files
            textured_mesh_path = os.path.join(TEXTURED_MODEL_DIR, f'{session_id}_textured.glb')
            temp_textured = AtomicServiceHandler.mesh_texturing(white_mesh_path, image_base64)
            
            # Copy to textured model directory
            shutil.copy(temp_textured, textured_mesh_path)
            os.remove(temp_textured)  # Remove original temp file
            logger.info(f"Textured mesh saved to {textured_mesh_path}")
            
            # Save to database with provided name or default name
            file_url = f"{session_id}_textured.glb"
            model_name = name if name else f"{session_id}_item"
            logger.info(f"About to save to database - session_id: {session_id}, model_name: '{model_name}'")
            save_model_to_db(session_id, "item", model_name, file_url)
            
            return {
                "main_service_id": main_service_id,
                "session_id": session_id,
                "queue_position": queue_position,
                "result": {
                    "white_mesh": f"{session_id}_white.glb",
                    "textured_mesh": f"{session_id}_textured.glb"
                }
            }
        except Exception as e:
            logger.error(f"Error in single_image_to_model: {e}")
            traceback.print_exc()
            return {
                "main_service_id": main_service_id,
                "session_id": session_id,
                "queue_position": queue_position,
                "error": str(e)
            }
    
    @staticmethod
    def multi_image_to_model_with_queue(
        images_base64: List[str],
        quality: str = "standard",
        main_service_id: str = None,
        session_id: str = None,
        name: str = None
    ) -> Dict:
        """Main Service 3: Multiple Images to Model"""
        if main_service_id is None:
            main_service_id = str(uuid.uuid4())
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        logger.info(f"multi_image_to_model_with_queue called - session_id: {session_id}, name: '{name}'")
        
        queue_position = get_queue_position(main_service_id)
        
        try:
            # Step 1: Multi-image to white shape
            mesh_path = AtomicServiceHandler.multi_image_to_shape(images_base64, quality, "glb")
            
            # Clear GPU memory after shape generation
            global pipeline_multi
            if pipeline_multi is not None:
                del pipeline_multi
                pipeline_multi = None
            clear_gpu_cache()
            logger.info("Multi-view shape generation model unloaded, GPU memory cleared")
            
            # Step 2: Generate texture (use first image as reference)
            textured_mesh_path = AtomicServiceHandler.mesh_texturing(mesh_path, images_base64[0])
            
            # Move files to session-specific location
            white_mesh_filename = f"{session_id}_white.glb"
            white_mesh_final = os.path.join(WHITE_MODEL_DIR, white_mesh_filename)
            shutil.copy(mesh_path, white_mesh_final)
            os.remove(mesh_path)
            
            textured_mesh_filename = f"{session_id}_textured.glb"
            textured_mesh_final = os.path.join(TEXTURED_MODEL_DIR, textured_mesh_filename)
            shutil.copy(textured_mesh_path, textured_mesh_final)
            os.remove(textured_mesh_path)
            
            # Save to database with provided name or default name
            model_name = name if name else f"{session_id}_multi_item"
            logger.info(f"About to save to database - session_id: {session_id}, model_name: '{model_name}'")
            save_model_to_db(session_id, "item", model_name, textured_mesh_filename)
            
            return {
                "main_service_id": main_service_id,
                "session_id": session_id,
                "queue_position": queue_position,
                "result": {
                    "white_mesh": white_mesh_filename,
                    "textured_mesh": textured_mesh_filename
                }
            }
        except Exception as e:
            return {
                "main_service_id": main_service_id,
                "session_id": session_id,
                "queue_position": queue_position,
                "error": str(e)
            }
    
    @staticmethod
    def images_to_scene_with_queue(
        images_base64: List[str],
        quality: str = "standard",
        main_service_id: str = None,
        session_id: str = None,
        name: str = None
    ) -> Dict:
        """Main Service 4: Images to Scene"""
        if main_service_id is None:
            main_service_id = str(uuid.uuid4())
        if session_id is None:
            session_id = main_service_id  # Use main_service_id as session_id if not provided
        
        logger.info(f"images_to_scene_with_queue called - session_id: {session_id}, name: '{name}'")
        
        queue_position = get_queue_position(main_service_id)
        
        try:
            # Generate scene from images
            scene_path = AtomicServiceHandler.scene_generation(images_base64, quality)
            
            # Save to database with provided name or default name
            scene_filename = os.path.basename(scene_path)
            model_name = name if name else f"{session_id}_scene"
            logger.info(f"About to save to database - session_id: {session_id}, model_name: '{model_name}'")
            save_model_to_db(session_id, "scene", model_name, scene_filename)
            
            return {
                "main_service_id": main_service_id,
                "session_id": session_id,
                "queue_position": queue_position,
                "result": {
                    "scene_model": scene_path
                }
            }
        except Exception as e:
            return {
                "main_service_id": main_service_id,
                "session_id": session_id,
                "queue_position": queue_position,
                "error": str(e)
            }
    
    @staticmethod
    def mesh_to_textured_with_queue(
        mesh_base64: str,
        image_base64: str,
        main_service_id: str = None
    ) -> Dict:
        """Main Service 5: Mesh to Textured Model"""
        if main_service_id is None:
            main_service_id = str(uuid.uuid4())
        
        queue_position = get_queue_position(main_service_id)
        
        try:
            # Save mesh temporarily
            mesh_data = base64.b64decode(mesh_base64)
            temp_mesh_path = os.path.join(SAVE_DIR, f'temp_mesh_{uuid.uuid4()}.glb')
            with open(temp_mesh_path, 'wb') as f:
                f.write(mesh_data)
            
            # Generate texture
            textured_mesh_path = AtomicServiceHandler.mesh_texturing(temp_mesh_path, image_base64)
            
            # Cleanup
            if os.path.exists(temp_mesh_path):
                os.remove(temp_mesh_path)
            
            # Return just the filename for download endpoint
            textured_mesh_filename = os.path.basename(textured_mesh_path)
            
            # Save to database (using main_service_id as session_id)
            save_model_to_db(main_service_id, "item", f"{main_service_id}_mesh_item", textured_mesh_filename)
            
            return {
                "main_service_id": main_service_id,
                "queue_position": queue_position,
                "result": {
                    "textured_mesh": textured_mesh_filename
                }
            }
        except Exception as e:
            return {
                "main_service_id": main_service_id,
                "queue_position": queue_position,
                "error": str(e)
            }


# =============================
# FastAPI Application
# =============================

app = FastAPI(title="Hunyuan3D-2 API Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/api/session/{session_id}")
async def get_session_files(session_id: str):
    """Get all files for a session ID"""
    try:
        files = []
        # Remove duplicate directories (WHITE_MODEL_DIR and TEMP_DIR are the same)
        search_dirs = list(dict.fromkeys([WHITE_MODEL_DIR, TEXTURED_MODEL_DIR, TEMP_DIR]))
        logger.info(f"Searching for session files with session_id: {session_id}")
        logger.info(f"Search directories: {search_dirs}")
        
        for dir_path in search_dirs:
            logger.info(f"Checking directory: {dir_path}")
            if os.path.exists(dir_path):
                logger.info(f"Directory exists, listing files...")
                all_files = os.listdir(dir_path)
                logger.info(f"Found {len(all_files)} files in {dir_path}")
                for filename in all_files:
                    logger.info(f"Checking file: {filename}, session_id prefix: {session_id}_")
                    if filename.startswith(f"{session_id}_"):
                        logger.info(f"Match found: {filename}")
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
                            logger.info(f"Added file: {filename} ({file_type}: {label})")
        
        logger.info(f"Total files found: {len(files)}")
        return {"session_id": session_id, "files": files}
    except Exception as e:
        logger.error(f"Error getting session files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/file/{file_path:path}")
async def download_file(file_path: str):
    """Download file endpoint"""
    try:
        # Try different possible locations
        possible_paths = [
            file_path,  # Absolute path
            os.path.join(WHITE_MODEL_DIR, file_path),  # White model directory
            os.path.join(TEXTURED_MODEL_DIR, file_path),  # Textured model directory
            os.path.join(TEMP_DIR, file_path),  # Temp directory
            os.path.join(SAVE_DIR, file_path),  # Legacy save directory
        ]
        
        # Also check with just filename
        filename = os.path.basename(file_path)
        if filename != file_path:
            possible_paths.extend([
                os.path.join(WHITE_MODEL_DIR, filename),
                os.path.join(TEXTURED_MODEL_DIR, filename),
                os.path.join(TEMP_DIR, filename),
                os.path.join(SAVE_DIR, filename),
            ])
        
        # Try to find existing file
        for path in possible_paths:
            if os.path.exists(path):
                return FileResponse(path)
        
        raise HTTPException(status_code=404, detail="File not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/atomic/text_to_image")
async def atomic_text_to_image(request: Request):
    """Atomic Service 1: Text to Image with Background Removal"""
    params = await request.json()
    text = params.get("text")
    quality = params.get("quality", "standard")
    use_volcano = params.get("use_volcano", True)  # Default to using volcano engine
    session_id = params.get("session_id")  # Optional session ID
    
    try:
        img_base64 = AtomicServiceHandler.text_to_image_with_bg_removal(
            text, quality, use_volcano=use_volcano, session_id=session_id
        )
        return {"status": "success", "image_base64": img_base64}
    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.post("/api/atomic/single_image_to_shape")
async def atomic_single_image_to_shape(request: Request):
    """Atomic Service 2: Single Image to Shape"""
    params = await request.json()
    image_base64 = params.get("image_base64")
    quality = params.get("quality", "standard")
    format = params.get("format", "glb")
    generate_multi_view = params.get("generate_multi_view", False)  # Enable multi-view generation
    
    try:
        mesh_path = AtomicServiceHandler.single_image_to_shape(
            image_base64, quality, format, generate_multi_view=generate_multi_view
        )
        return FileResponse(mesh_path)
    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.post("/api/atomic/multi_image_to_shape")
async def atomic_multi_image_to_shape(request: Request):
    """Atomic Service 3: Multi Image to Shape"""
    params = await request.json()
    images_base64 = params.get("images_base64")
    quality = params.get("quality", "standard")
    format = params.get("format", "glb")
    
    try:
        mesh_path = AtomicServiceHandler.multi_image_to_shape(images_base64, quality, format)
        return FileResponse(mesh_path)
    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.post("/api/atomic/mesh_texturing")
async def atomic_mesh_texturing(request: Request):
    """Atomic Service 4: Mesh Texturing"""
    params = await request.json()
    mesh_path = params.get("mesh_path")
    image_base64 = params.get("image_base64")
    
    try:
        textured_mesh_path = AtomicServiceHandler.mesh_texturing(mesh_path, image_base64)
        return FileResponse(textured_mesh_path)
    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.post("/api/atomic/scene_generation")
async def atomic_scene_generation(request: Request):
    """Atomic Service 5: Scene Generation"""
    params = await request.json()
    images_base64 = params.get("images_base64")
    quality = params.get("quality", "standard")
    
    try:
        scene_path = AtomicServiceHandler.scene_generation(images_base64, quality)
        return FileResponse(scene_path)
    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.post("/api/main/text_to_model")
async def main_text_to_model(request: TextToModelRequest):
    """Main Service 1: Text to Model"""
    logger.info(f"API endpoint /api/main/text_to_model received request with name: '{request.name}'")
    main_service_id = request.main_service_id or str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())
    
    result = MainServiceHandler.text_to_model_with_queue(
        text=request.text,
        quality=request.quality,
        main_service_id=main_service_id,
        session_id=session_id,
        use_volcano=request.use_volcano,
        volcano_access_key=request.volcano_access_key,
        volcano_secret_key=request.volcano_secret_key,
        name=request.name
    )
    return JSONResponse(content=result)


@app.post("/api/main/single_image_to_model")
async def main_single_image_to_model(request: SingleImageToModelRequest):
    """Main Service 2: Single Image to Model"""
    logger.info(f"API endpoint /api/main/single_image_to_model received request with name: '{request.name}'")
    main_service_id = request.main_service_id or str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())
    
    result = MainServiceHandler.single_image_to_model_with_queue(
        image_base64=request.image_base64,
        quality=request.quality,
        main_service_id=main_service_id,
        session_id=session_id,
        name=request.name
    )
    return JSONResponse(content=result)


@app.post("/api/main/multi_image_to_model")
async def main_multi_image_to_model(request: MultiImageToModelRequest):
    """Main Service 3: Multi Image to Model"""
    logger.info(f"API endpoint /api/main/multi_image_to_model received request with name: '{request.name}'")
    main_service_id = request.main_service_id or str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())
    
    result = MainServiceHandler.multi_image_to_model_with_queue(
        images_base64=request.images_base64,
        quality=request.quality,
        main_service_id=main_service_id,
        session_id=session_id,
        name=request.name
    )
    return JSONResponse(content=result)


@app.post("/api/main/images_to_scene")
async def main_images_to_scene(request: ImagesToSceneRequest):
    """Main Service 4: Images to Scene"""
    logger.info(f"API endpoint /api/main/images_to_scene received request with name: '{request.name}'")
    main_service_id = request.main_service_id or str(uuid.uuid4())
    session_id = request.session_id or main_service_id
    
    result = MainServiceHandler.images_to_scene_with_queue(
        images_base64=request.images_base64,
        quality=request.quality,
        main_service_id=main_service_id,
        session_id=session_id,
        name=request.name
    )
    return JSONResponse(content=result)


@app.post("/api/main/mesh_to_textured")
async def main_mesh_to_textured(request: MeshToTexturedRequest):
    """Main Service 5: Mesh to Textured Model"""
    main_service_id = request.main_service_id or str(uuid.uuid4())
    
    result = MainServiceHandler.mesh_to_textured_with_queue(
        mesh_base64=request.mesh_base64,
        image_base64=request.image_base64,
        main_service_id=main_service_id
    )
    return JSONResponse(content=result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    logger.info(f"Starting API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
