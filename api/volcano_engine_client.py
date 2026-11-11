#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
火山引擎API客户端
用于文生图和图生图功能
"""

import json
import os
import base64
import datetime
import hashlib
import hmac
import requests
import time
import logging

logger = logging.getLogger(__name__)


class VolcanoEngineClient:
    """火山引擎API客户端"""
    
    def __init__(self, access_key=None, secret_key=None):
        """
        初始化客户端
        
        Args:
            access_key: 火山引擎访问密钥ID（从环境变量VOLCANO_ACCESS_KEY获取）
            secret_key: 火山引擎访问密钥Secret（从环境变量VOLCANO_SECRET_KEY获取）
        """
        self.access_key = access_key or os.getenv('VOLCANO_ACCESS_KEY')
        self.secret_key = secret_key or os.getenv('VOLCANO_SECRET_KEY')
        
        if not self.access_key or not self.secret_key:
            raise ValueError('火山引擎访问密钥未设置，请设置环境变量 VOLCANO_ACCESS_KEY 和 VOLCANO_SECRET_KEY')
        
        # API配置
        self.method = 'POST'
        self.host = 'visual.volcengineapi.com'
        self.region = 'cn-north-1'
        self.endpoint = 'https://visual.volcengineapi.com'
        self.service = 'cv'
    
    def _sign(self, key, msg):
        """计算HMAC-SHA256签名"""
        return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()
    
    def _get_signature_key(self, key, dateStamp, regionName, serviceName):
        """获取签名密钥"""
        kDate = self._sign(key.encode('utf-8'), dateStamp)
        kRegion = self._sign(kDate, regionName)
        kService = self._sign(kRegion, serviceName)
        kSigning = self._sign(kService, 'request')
        return kSigning
    
    def _format_query(self, parameters):
        """格式化查询参数"""
        request_parameters_init = ''
        for key in sorted(parameters):
            request_parameters_init += key + '=' + parameters[key] + '&'
        request_parameters = request_parameters_init[:-1]
        return request_parameters
    
    def _sign_v4_request(self, req_query, req_body):
        """生成V4签名请求"""
        # 生成时间戳
        t = datetime.datetime.utcnow()
        current_date = t.strftime('%Y%m%dT%H%M%SZ')
        datestamp = t.strftime('%Y%m%d')
        
        # 构建规范请求
        canonical_uri = '/'
        canonical_querystring = req_query
        signed_headers = 'content-type;host;x-content-sha256;x-date'
        payload_hash = hashlib.sha256(req_body.encode('utf-8')).hexdigest()
        content_type = 'application/json'
        
        canonical_headers = (
            f'content-type:{content_type}\n'
            f'host:{self.host}\n'
            f'x-content-sha256:{payload_hash}\n'
            f'x-date:{current_date}\n'
        )
        
        canonical_request = (
            f'{self.method}\n'
            f'{canonical_uri}\n'
            f'{canonical_querystring}\n'
            f'{canonical_headers}\n'
            f'{signed_headers}\n'
            f'{payload_hash}'
        )
        
        # 构建签名字符串
        algorithm = 'HMAC-SHA256'
        credential_scope = f'{datestamp}/{self.region}/{self.service}/request'
        string_to_sign = (
            f'{algorithm}\n'
            f'{current_date}\n'
            f'{credential_scope}\n'
            f'{hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()}'
        )
        
        # 计算签名
        signing_key = self._get_signature_key(
            self.secret_key, 
            datestamp, 
            self.region, 
            self.service
        )
        
        signature = hmac.new(
            signing_key, 
            string_to_sign.encode('utf-8'), 
            hashlib.sha256
        ).hexdigest()
        
        # 构建授权头
        authorization_header = (
            f'{algorithm} '
            f'Credential={self.access_key}/{credential_scope}, '
            f'SignedHeaders={signed_headers}, '
            f'Signature={signature}'
        )
        
        headers = {
            'X-Date': current_date,
            'Authorization': authorization_header,
            'X-Content-Sha256': payload_hash,
            'Content-Type': content_type
        }
        
        return headers
    
    def text_to_image(self, prompt, width=1024, height=1024, seed=None):
        """
        使用火山引擎API进行文生图
        
        Args:
            prompt: 文本提示词
            width: 图像宽度
            height: 图像高度
            seed: 随机种子
            
        Returns:
            PIL.Image: 生成的图像
        """
        try:
            logger.info(f"使用火山引擎API生成图像，提示词: {prompt[:50]}...")
            
            # 配置查询参数
            query_params = {
                'Action': 'CVProcess',
                'Version': '2022-08-31',
            }
            formatted_query = self._format_query(query_params)
            
            # 构建请求体
            body_params = {
                "req_key": "high_aes_general_v30l_zt2i",  # 文生图标识
                "prompt": prompt,
                "image_param": {
                    "width": width,
                    "height": height
                }
            }
            
            # 添加可选参数
            if seed:
                body_params["seed"] = seed
            
            formatted_body = json.dumps(body_params)
            
            # 生成签名
            headers = self._sign_v4_request(formatted_query, formatted_body)
            
            # 发送请求
            request_url = f'{self.endpoint}?{formatted_query}'
            
            response = requests.post(
                request_url, 
                headers=headers, 
                data=formatted_body,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('code') == 10000 and 'data' in result:
                    # 处理返回的图像数据
                    data = result['data']
                    
                    # 优先使用base64图像
                    if 'image' in data:
                        from PIL import Image
                        from io import BytesIO
                        image_data = base64.b64decode(data['image'])
                        image = Image.open(BytesIO(image_data))
                        logger.info("火山引擎文生图成功")
                        return image
                    elif 'binary_data_base64' in data and data['binary_data_base64']:
                        from PIL import Image
                        from io import BytesIO
                        image_data = base64.b64decode(data['binary_data_base64'][0])
                        image = Image.open(BytesIO(image_data))
                        logger.info("火山引擎文生图成功")
                        return image
                    else:
                        logger.error("API响应中未找到图像数据")
                        raise ValueError("API响应中未找到图像数据")
                else:
                    error_msg = result.get('message', '未知错误')
                    logger.error(f"火山引擎API错误: {error_msg}")
                    raise ValueError(f"火山引擎API错误: {error_msg}")
            else:
                logger.error(f"HTTP请求失败: {response.status_code}")
                raise ValueError(f"HTTP请求失败: {response.status_code}")
                
        except Exception as e:
            logger.error(f"火山引擎文生图失败: {e}")
            raise
    
    def image_to_image(self, prompt, image_base64, scale=0.6, seed=-1, max_retries=10):
        """
        使用火山引擎API进行图生图
        
        Args:
            prompt: 文本提示词
            image_base64: 输入图片的base64编码
            scale: 编辑强度（0-1）
            seed: 随机种子
            max_retries: 最大重试次数
            
        Returns:
            PIL.Image: 生成的图像
        """
        try:
            logger.info(f"使用火山引擎API进行图生图，提示词: {prompt[:50]}...")
            
            # 提交任务
            task_id = self._submit_im2im_task(prompt, image_base64, scale, seed)
            if not task_id:
                raise ValueError("任务提交失败")
            
            # 查询结果
            retries = 0
            while retries < max_retries:
                time.sleep(2)
                result = self._get_task_result(task_id)
                
                if result and result.get('status') == 'done':
                    # 获取图像数据
                    if 'binary_data_base64' in result and result['binary_data_base64']:
                        from PIL import Image
                        from io import BytesIO
                        image_data = base64.b64decode(result['binary_data_base64'][0])
                        image = Image.open(BytesIO(image_data))
                        logger.info("火山引擎图生图成功")
                        return image
                    elif 'image_urls' in result and result['image_urls']:
                        # 下载图像
                        img_response = requests.get(result['image_urls'][0], timeout=30)
                        from PIL import Image
                        from io import BytesIO
                        image = Image.open(BytesIO(img_response.content))
                        logger.info("火山引擎图生图成功")
                        return image
                
                retries += 1
                if result and result.get('status') not in ['in_queue', 'generating']:
                    break
            
            raise ValueError("查询超时或任务失败")
            
        except Exception as e:
            logger.error(f"火山引擎图生图失败: {e}")
            raise
    
    def _submit_im2im_task(self, prompt, image_base64, scale, seed):
        """提交图生图任务"""
        query_params = {
            'Action': 'CVSync2AsyncSubmitTask',
            'Version': '2022-08-31',
        }
        formatted_query = self._format_query(query_params)
        
        body_params = {
            "req_key": "seededit_v3.0",
            "prompt": prompt,
            "binary_data_base64": [image_base64],
            "seed": seed,
            "scale": scale
        }
        
        formatted_body = json.dumps(body_params, ensure_ascii=False)
        headers = self._sign_v4_request(formatted_query, formatted_body)
        
        request_url = f'{self.endpoint}?{formatted_query}'
        
        response = requests.post(
            request_url, 
            headers=headers, 
            data=formatted_body,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('code') == 10000:
                task_id = result['data']['task_id']
                logger.info(f"任务提交成功，task_id: {task_id}")
                return task_id
        
        logger.error(f"任务提交失败: {response.text}")
        return None
    
    def _get_task_result(self, task_id):
        """查询任务结果"""
        query_params = {
            'Action': 'CVSync2AsyncGetResult',
            'Version': '2022-08-31',
        }
        formatted_query = self._format_query(query_params)
        
        req_json_config = {
            "return_url": True,
            "logo_info": {
                "add_logo": False,
                "position": 0,
                "language": 0,
                "opacity": 1
            }
        }
        
        body_params = {
            "req_key": "seededit_v3.0",
            "task_id": task_id,
            "req_json": json.dumps(req_json_config, ensure_ascii=False)
        }
        
        formatted_body = json.dumps(body_params, ensure_ascii=False)
        headers = self._sign_v4_request(formatted_query, formatted_body)
        
        request_url = f'{self.endpoint}?{formatted_query}'
        
        response = requests.post(
            request_url, 
            headers=headers, 
            data=formatted_body,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('code') == 10000:
                data = result['data']
                status = data.get('status')
                
                if status == 'done':
                    return data
                elif status in ['in_queue', 'generating']:
                    return {'status': status}
        
        return None

