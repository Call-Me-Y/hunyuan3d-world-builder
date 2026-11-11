# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

"""
Enhanced model worker with support for multiple parameters, multi-view images, and queue management.
"""
import argparse
import asyncio
import base64
import gc
import json
import logging
import logging.handlers
import os
import psutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Dict, List, Optional, Union

import torch
import trimesh
import uvicorn
from PIL import Image
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer, \
    MeshSimplifier
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.text2image import HunyuanDiTPipeline

# 设置基于项目目录的模型路径；可通过环境变量覆盖
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'), override=True)
MODEL_ROOT = os.getenv('HY3DGEN_MODELS') or os.path.join(PROJECT_ROOT, 'Hunyuan3D-2', 'models')
if not os.path.isabs(MODEL_ROOT):
    MODEL_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, MODEL_ROOT))
os.environ.setdefault('HY3DGEN_MODELS', MODEL_ROOT)

# 设置PyTorch CUDA内存优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

LOGDIR = '.'
SAVE_DIR = 'gradio_cache'
os.makedirs(SAVE_DIR, exist_ok=True)

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None

# 任务状态枚举
class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# 模型质量枚举
class ModelQuality(str, Enum):
    FAST = "fast"      # 快速模式，使用turbo版本
    BALANCED = "balanced"  # 平衡模式
    HIGH = "high"      # 高质量模式

# 请求模型
class GenerationRequest(BaseModel):
    # 图像相关参数
    images: Optional[Dict[str, str]] = Field(None, description="多视角图像字典，键为视角名称，值为base64编码的图像")
    image: Optional[str] = Field(None, description="单张图像的base64编码")
    text: Optional[str] = Field(None, description="文本提示词（用于文本到图像）")
    
    # 生成参数
    model_quality: ModelQuality = Field(ModelQuality.BALANCED, description="模型质量：fast/balanced/high")
    enable_texture: bool = Field(True, description="是否启用贴图生成")
    has_multiview: bool = Field(False, description="是否有多视角图片")
    return_progress: bool = Field(False, description="是否返回进度信息")
    return_white_model: bool = Field(False, description="是否返回白模（无贴图）")
    
    # 技术参数
    seed: int = Field(1234, description="随机种子")
    octree_resolution: int = Field(128, description="八叉树分辨率")
    num_inference_steps: int = Field(5, description="推理步数")
    guidance_scale: float = Field(5.0, description="引导尺度")
    face_count: int = Field(40000, description="最大面数")
    
    # 输出参数
    output_format: str = Field("glb", description="输出格式：glb/obj/ply")
    
    class Config:
        use_enum_values = True

# 响应模型
class GenerationResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: Optional[str] = None
    progress: Optional[float] = None
    result_files: Optional[Dict[str, str]] = None  # 文件名到base64的映射
    error: Optional[str] = None

# 任务信息
@dataclass
class TaskInfo:
    task_id: str
    status: TaskStatus
    progress: float
    message: str
    result_files: Dict[str, str]
    error: Optional[str] = None
    created_at: float = None
    completed_at: Optional[float] = None

def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def load_image_from_base64(image_base64: str) -> Image.Image:
    """从base64字符串加载图像"""
    return Image.open(BytesIO(base64.b64decode(image_base64)))


def detect_background(image: Image.Image) -> bool:
    """检测图像是否有背景（简单检测：检查alpha通道）"""
    if image.mode == 'RGBA':
        # 检查alpha通道是否有透明像素
        alpha = image.split()[-1]
        return alpha.getextrema()[0] < 255  # 如果有透明像素，说明有背景
    return True  # RGB图像默认有背景


class ModelCacheManager:
    """模型缓存管理器，负责智能加载和卸载模型"""
    
    def __init__(self, device='cuda', max_memory_usage=0.85):
        self.device = device
        self.max_memory_usage = max_memory_usage  # 最大内存使用率
        self.cached_models = {}  # 缓存的模型
        self.model_usage_count = {}  # 模型使用计数
        self.model_last_used = {}  # 模型最后使用时间
        self.lock = threading.Lock()
        
    def get_gpu_memory_usage(self):
        """获取GPU内存使用情况"""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            return allocated_memory / total_memory
        return 0.0
    
    def get_system_memory_usage(self):
        """获取系统内存使用情况"""
        return psutil.virtual_memory().percent / 100.0
    
    def should_cleanup_memory(self):
        """判断是否需要清理内存"""
        gpu_usage = self.get_gpu_memory_usage()
        system_usage = self.get_system_memory_usage()
        return gpu_usage > self.max_memory_usage or system_usage > 0.9
    
    def cleanup_unused_models(self):
        """清理未使用的模型"""
        with self.lock:
            current_time = time.time()
            models_to_remove = []
            
            # 找出超过5分钟未使用的模型
            for model_key, last_used in self.model_last_used.items():
                if current_time - last_used > 300:  # 5分钟
                    models_to_remove.append(model_key)
            
            # 按使用频率排序，优先移除使用次数少的模型
            if not models_to_remove and self.should_cleanup_memory():
                sorted_models = sorted(
                    self.model_usage_count.items(),
                    key=lambda x: x[1]
                )
                # 移除使用次数最少的模型
                for model_key, _ in sorted_models:
                    if model_key in self.cached_models:
                        models_to_remove.append(model_key)
                        break
            
            # 移除模型
            for model_key in models_to_remove:
                if model_key in self.cached_models:
                    logger.info(f"Cleaning up unused model: {model_key}")
                    del self.cached_models[model_key]
                    if model_key in self.model_usage_count:
                        del self.model_usage_count[model_key]
                    if model_key in self.model_last_used:
                        del self.model_last_used[model_key]
            
            # 强制垃圾回收和GPU内存清理
            if models_to_remove:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Cleaned up {len(models_to_remove)} models")
    
    def cleanup_shape_models(self):
        """专门清理形状生成模型，为贴图模型腾出空间"""
        with self.lock:
            models_to_remove = []
            for model_key in self.cached_models.keys():
                if model_key.startswith('shape_'):
                    models_to_remove.append(model_key)
            
            # 移除形状生成模型
            for model_key in models_to_remove:
                if model_key in self.cached_models:
                    logger.info(f"Cleaning up shape model to free memory: {model_key}")
                    del self.cached_models[model_key]
                    if model_key in self.model_usage_count:
                        del self.model_usage_count[model_key]
                    if model_key in self.model_last_used:
                        del self.model_last_used[model_key]
            
            # 强制垃圾回收和GPU内存清理
            if models_to_remove:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Cleaned up {len(models_to_remove)} shape models to free memory for texture generation")
    
    def force_cleanup_all_models(self):
        """强制清理所有模型，释放最大内存"""
        with self.lock:
            logger.info("Force cleaning up all models to free maximum memory")
            self.cached_models.clear()
            self.model_usage_count.clear()
            self.model_last_used.clear()
            
            # 多次垃圾回收和内存清理
            for _ in range(3):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            logger.info("Force cleanup completed - all models removed")
    
    def get_model(self, model_key: str, load_func, *args, **kwargs):
        """获取模型，如果不存在则加载"""
        with self.lock:
            # 检查是否需要清理内存
            if self.should_cleanup_memory():
                self.cleanup_unused_models()
            
            # 如果模型已缓存，直接返回
            if model_key in self.cached_models:
                self.model_usage_count[model_key] = self.model_usage_count.get(model_key, 0) + 1
                self.model_last_used[model_key] = time.time()
                logger.info(f"Using cached model: {model_key}")
                return self.cached_models[model_key]
            
            # 如果内存不足，先清理
            if self.should_cleanup_memory():
                self.cleanup_unused_models()
            
            # 加载新模型
            try:
                logger.info(f"Loading new model: {model_key}")
                model = load_func(*args, **kwargs)
                self.cached_models[model_key] = model
                self.model_usage_count[model_key] = 1
                self.model_last_used[model_key] = time.time()
                return model
            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"CUDA OOM when loading {model_key}, cleaning up and retrying...")
                # 清理所有模型并重试
                self.cleanup_all_models()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 重试加载
                model = load_func(*args, **kwargs)
                self.cached_models[model_key] = model
                self.model_usage_count[model_key] = 1
                self.model_last_used[model_key] = time.time()
                return model
    
    def cleanup_all_models(self):
        """清理所有模型"""
        with self.lock:
            logger.info("Cleaning up all cached models")
            self.cached_models.clear()
            self.model_usage_count.clear()
            self.model_last_used.clear()
    
    def get_cache_info(self):
        """获取缓存信息"""
        with self.lock:
            return {
                'cached_models': list(self.cached_models.keys()),
                'model_usage_count': dict(self.model_usage_count),
                'gpu_memory_usage': self.get_gpu_memory_usage(),
                'system_memory_usage': self.get_system_memory_usage()
            }


class EnhancedModelWorker:
    def __init__(self,
                 model_path=os.path.join(PROJECT_ROOT, 'Hunyuan3D-2'),
                 tex_model_path=os.path.join(PROJECT_ROOT, 'Hunyuan3D-2'),
                 device='cuda',
                 enable_tex=True):
        self.model_path = model_path
        self.tex_model_path = tex_model_path
        self.device = device
        self.enable_tex = enable_tex
        
        logger.info(f"Loading models from {model_path} on device {device}...")
        
        # 初始化模型缓存管理器
        self.model_cache = ModelCacheManager(device=device, max_memory_usage=0.85)
        
        # 初始化背景移除器
        self.rembg = BackgroundRemover()
        
        # 任务存储
        self.tasks: Dict[str, TaskInfo] = {}
        
        # 线程池用于异步处理
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    def _load_shape_pipeline(self, quality: ModelQuality, has_multiview: bool):
        """根据质量要求加载形状生成管道"""
        # 根据质量和多视角选择子文件夹
        if has_multiview:
            if quality == ModelQuality.FAST:
                subfolder = 'hunyuan3d-dit-v2-mv-turbo'
            else:
                subfolder = 'hunyuan3d-dit-v2-mv-turbo'  # 多视角只有turbo版本
        else:
            if quality == ModelQuality.FAST:
                subfolder = 'hunyuan3d-dit-v2-0-turbo'
            elif quality == ModelQuality.HIGH:
                subfolder = 'hunyuan3d-dit-v2-0'
            else:
                subfolder = 'hunyuan3d-dit-v2-0-turbo'
        
        # 创建模型键
        model_key = f"shape_{subfolder}"
        
        def load_pipeline():
            logger.info(f"Loading shape generation pipeline: {subfolder}")
            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                self.model_path,
                subfolder=subfolder,
                use_safetensors=True,
                device=self.device,
            )
            
            # 启用FlashVDM加速（如果支持）
            try:
                if not has_multiview:  # 多视角版本可能不支持FlashVDM
                    pipeline.enable_flashvdm(mc_algo='mc')
            except Exception as e:
                logger.warning(f"Failed to enable FlashVDM: {e}")
            
            return pipeline
        
        # 使用缓存管理器获取模型
        return self.model_cache.get_model(model_key, load_pipeline)
    
    def _load_tex_pipeline(self):
        """加载贴图生成管道"""
        if not self.enable_tex:
            return None
        
        model_key = "texture_hunyuan3d-paint-v2-0-turbo"
        
        def load_tex_pipeline():
            logger.info("Loading texture generation pipeline...")
            return Hunyuan3DPaintPipeline.from_pretrained(
                self.tex_model_path,
                subfolder='hunyuan3d-paint-v2-0-turbo'
            )
        
        # 强制清理所有模型，为贴图模型腾出最大内存
        logger.info("Force cleaning all models before loading texture pipeline")
        self.model_cache.force_cleanup_all_models()
        
        # 使用缓存管理器获取模型
        return self.model_cache.get_model(model_key, load_tex_pipeline)
    
    def _process_images(self, request: GenerationRequest) -> Union[Image.Image, Dict[str, Image.Image]]:
        """处理输入图像"""
        if request.images:
            # 多视角图像处理
            processed_images = {}
            for view_name, image_base64 in request.images.items():
                image = load_image_from_base64(image_base64)
                
                # 检测并处理背景
                if detect_background(image):
                    logger.info(f"Processing background for {view_name} view")
                    image = self.rembg(image)
                
                processed_images[view_name] = image
            return processed_images
            
        elif request.image:
            # 单张图像处理
            image = load_image_from_base64(request.image)
            
            # 检测并处理背景
            if detect_background(image):
                logger.info("Processing background for single image")
                image = self.rembg(image)
            
            return image
            
        elif request.text:
            # 文本到图像（暂未实现）
            raise NotImplementedError("Text to image generation not implemented yet")
            
        else:
            raise ValueError("No input provided: need images, image, or text")
    
    def _get_generation_params(self, request: GenerationRequest) -> Dict:
        """获取生成参数"""
        params = {
            'seed': request.seed,
            'octree_resolution': request.octree_resolution,
            'num_inference_steps': request.num_inference_steps,
            'guidance_scale': request.guidance_scale,
            'mc_algo': 'mc',
            'generator': torch.Generator(self.device).manual_seed(request.seed)
        }
        
        # 根据质量调整参数
        if request.model_quality == ModelQuality.FAST:
            params['num_inference_steps'] = min(params['num_inference_steps'], 5)
            params['octree_resolution'] = min(params['octree_resolution'], 128)
        elif request.model_quality == ModelQuality.HIGH:
            params['num_inference_steps'] = max(params['num_inference_steps'], 20)
            params['octree_resolution'] = max(params['octree_resolution'], 256)
        
        return params
    
    def _update_task_progress(self, task_id: str, progress: float, message: str):
        """更新任务进度"""
        if task_id in self.tasks:
            self.tasks[task_id].progress = progress
            self.tasks[task_id].message = message
            self.tasks[task_id].status = TaskStatus.PROCESSING
    
    async def generate_async(self, request: GenerationRequest) -> str:
        """异步生成3D模型"""
        task_id = str(uuid.uuid4())
        
        # 创建任务信息
        task_info = TaskInfo(
            task_id=task_id,
            status=TaskStatus.PENDING,
            progress=0.0,
            message="任务已创建",
            result_files={},
            created_at=time.time()
        )
        self.tasks[task_id] = task_info
        
        # 在线程池中执行生成任务
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self._generate_sync, task_id, request)
        
        return task_id
    
    def _generate_sync(self, task_id: str, request: GenerationRequest):
        """同步生成3D模型（在线程池中执行）"""
        try:
            self._update_task_progress(task_id, 0.1, "开始处理图像...")
            
            # 处理输入图像
            processed_input = self._process_images(request)
            self._update_task_progress(task_id, 0.2, "图像处理完成")
            
            # 加载形状生成管道
            self._update_task_progress(task_id, 0.3, "加载形状生成模型...")
            pipeline_shape = self._load_shape_pipeline(request.model_quality, request.has_multiview)
            self._update_task_progress(task_id, 0.4, "形状生成模型加载完成")
            
            # 生成3D形状
            self._update_task_progress(task_id, 0.5, "生成3D形状...")
            generation_params = self._get_generation_params(request)
            generation_params['image'] = processed_input
            
            start_time = time.time()
            mesh = pipeline_shape(**generation_params)[0]
            generation_time = time.time() - start_time
            logger.info(f"3D shape generation completed in {generation_time:.2f} seconds")
            
            self._update_task_progress(task_id, 0.7, "3D形状生成完成")
            
            # 保存白模（如果需要）
            result_files = {}
            if request.return_white_model:
                white_model_path = os.path.join(SAVE_DIR, f'{task_id}_white.{request.output_format}')
                mesh.export(white_model_path)
                with open(white_model_path, 'rb') as f:
                    result_files['white_model'] = base64.b64encode(f.read()).decode()
                self._update_task_progress(task_id, 0.8, "白模已保存")
            
            # 生成贴图（如果启用）
            if request.enable_texture and self.enable_tex:
                self._update_task_progress(task_id, 0.8, "生成贴图...")
                
                # 加载贴图生成管道（会自动清理形状模型）
                pipeline_tex = self._load_tex_pipeline()
                if pipeline_tex is not None:
                    # 预处理网格
                    mesh = FloaterRemover()(mesh)
                    mesh = DegenerateFaceRemover()(mesh)
                    mesh = FaceReducer()(mesh, max_facenum=request.face_count)
                    
                    # 在贴图生成前再次清理内存
                    logger.info("Cleaning memory before texture generation")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 生成贴图
                    if request.has_multiview and isinstance(processed_input, dict):
                        # 多视角情况，使用正面图像
                        front_image = processed_input.get('front', list(processed_input.values())[0])
                    else:
                        front_image = processed_input
                    
                    try:
                        textured_mesh = pipeline_tex(mesh, image=front_image)
                        mesh = textured_mesh
                        self._update_task_progress(task_id, 0.9, "贴图生成完成")
                    except torch.cuda.OutOfMemoryError as e:
                        logger.error(f"CUDA OOM during texture generation: {e}")
                        # 强制清理所有模型并重试
                        self.model_cache.force_cleanup_all_models()
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # 重试贴图生成
                        logger.info("Retrying texture generation after memory cleanup")
                        textured_mesh = pipeline_tex(mesh, image=front_image)
                        mesh = textured_mesh
                        self._update_task_progress(task_id, 0.9, "贴图生成完成（重试成功）")
            
            # 保存最终模型
            final_model_path = os.path.join(SAVE_DIR, f'{task_id}_final.{request.output_format}')
            mesh.export(final_model_path)
            
            with open(final_model_path, 'rb') as f:
                result_files['final_model'] = base64.b64encode(f.read()).decode()
            
            # 更新任务状态
            self.tasks[task_id].status = TaskStatus.COMPLETED
            self.tasks[task_id].progress = 1.0
            self.tasks[task_id].message = "生成完成"
            self.tasks[task_id].result_files = result_files
            self.tasks[task_id].completed_at = time.time()
            
            # 清理GPU内存
            torch.cuda.empty_cache()
            
            # 强制清理所有模型以释放内存
            logger.info("Cleaning all models after generation completion")
            self.model_cache.force_cleanup_all_models()
            
        except Exception as e:
            logger.error(f"Generation failed for task {task_id}: {e}")
            traceback.print_exc()
            
            self.tasks[task_id].status = TaskStatus.FAILED
            self.tasks[task_id].error = str(e)
            self.tasks[task_id].message = f"生成失败: {e}"
    
    def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务状态"""
        return self.tasks.get(task_id)
    
    def get_queue_length(self) -> int:
        """获取队列长度"""
        return len([t for t in self.tasks.values() if t.status in [TaskStatus.PENDING, TaskStatus.PROCESSING]])


# 全局变量
worker_id = str(uuid.uuid4())[:6]
logger = build_logger("enhanced_controller", f"{SAVE_DIR}/enhanced_controller.log")
worker = None
model_semaphore = None

# FastAPI应用
app = FastAPI(title="Hunyuan3D-2 Enhanced API", version="2.0.0")

# CORS中间件
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """生成3D模型（同步接口）"""
    logger.info("Received generation request")
    
    try:
        # 检查并发限制
        if model_semaphore is not None:
            await model_semaphore.acquire()
        
        task_id = await worker.generate_async(request)
        
        # 等待任务完成
        while True:
            task_info = worker.get_task_status(task_id)
            if task_info.status == TaskStatus.COMPLETED:
                return GenerationResponse(
                    task_id=task_id,
                    status=task_info.status,
                    message=task_info.message,
                    result_files=task_info.result_files
                )
            elif task_info.status == TaskStatus.FAILED:
                return GenerationResponse(
                    task_id=task_id,
                    status=task_info.status,
                    error=task_info.error,
                    message=task_info.message
                )
            await asyncio.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Generation error: {e}")
        traceback.print_exc()
        return GenerationResponse(
            task_id="",
            status=TaskStatus.FAILED,
            error=str(e),
            message="生成失败"
        )
    finally:
        if model_semaphore is not None:
            model_semaphore.release()

@app.post("/generate_async", response_model=GenerationResponse)
async def generate_async(request: GenerationRequest):
    """异步生成3D模型"""
    logger.info("Received async generation request")
    
    try:
        task_id = await worker.generate_async(request)
        return GenerationResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="任务已提交，请使用状态接口查询进度"
        )
    except Exception as e:
        logger.error(f"Async generation error: {e}")
        traceback.print_exc()
        return GenerationResponse(
            task_id="",
            status=TaskStatus.FAILED,
            error=str(e),
            message="任务提交失败"
        )

@app.get("/status/{task_id}", response_model=GenerationResponse)
async def get_status(task_id: str):
    """获取任务状态"""
    task_info = worker.get_task_status(task_id)
    if task_info is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return GenerationResponse(
        task_id=task_id,
        status=task_info.status,
        message=task_info.message,
        progress=task_info.progress,
        result_files=task_info.result_files if task_info.status == TaskStatus.COMPLETED else None,
        error=task_info.error
    )

@app.get("/queue/status")
async def get_queue_status():
    """获取队列状态"""
    return {
        "queue_length": worker.get_queue_length(),
        "total_tasks": len(worker.tasks),
        "active_tasks": len([t for t in worker.tasks.values() if t.status == TaskStatus.PROCESSING])
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "worker_id": worker_id}

@app.get("/cache/status")
async def get_cache_status():
    """获取模型缓存状态"""
    cache_info = worker.model_cache.get_cache_info()
    return {
        "cache_info": cache_info,
        "queue_length": worker.get_queue_length(),
        "total_tasks": len(worker.tasks)
    }

@app.post("/cache/cleanup")
async def cleanup_cache():
    """手动清理模型缓存"""
    worker.model_cache.cleanup_all_models()
    return {"message": "Cache cleaned up successfully"}

def initialize_worker():
    """初始化工作器"""
    global worker, model_semaphore
    
    # 默认参数
    model_path = 'Hunyuan3D-2'
    tex_model_path = 'Hunyuan3D-2'
    device = 'cuda'
    concurrency = 2
    enable_tex = True
    
    # 创建信号量
    model_semaphore = asyncio.Semaphore(concurrency)
    
    # 初始化工作器
    worker = EnhancedModelWorker(
        model_path=model_path,
        tex_model_path=tex_model_path,
        device=device,
        enable_tex=enable_tex
    )
    
    logger.info(f"Worker initialized with model_path={model_path}, device={device}")

# 初始化工作器
initialize_worker()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Hunyuan3D-2 API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model_path", type=str, default='Hunyuan3D-2')
    parser.add_argument("--tex_model_path", type=str, default='Hunyuan3D-2')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument('--enable_tex', action='store_true', default=True)
    
    args = parser.parse_args()
    logger.info(f"Starting enhanced API server with args: {args}")
    
    # 重新初始化工作器（如果提供了不同的参数）
    if (args.model_path != 'Hunyuan3D-2' or 
        args.tex_model_path != 'Hunyuan3D-2' or 
        args.device != 'cuda' or 
        args.limit_model_concurrency != 2 or 
        not args.enable_tex):
        
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
        worker = EnhancedModelWorker(
            model_path=args.model_path,
            tex_model_path=args.tex_model_path,
            device=args.device,
            enable_tex=args.enable_tex
        )
        logger.info(f"Worker re-initialized with custom args")
    
    # 启动服务器
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
