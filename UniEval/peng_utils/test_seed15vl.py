import base64
import json
import os
import sys
import time
import traceback
from io import BytesIO
import requests
from urllib.parse import urlparse

import numpy as np
import torch
from openai import OpenAI

sys.path.append("./peng_utils")
from . import get_image


def read_config(path):
    with open(path) as json_file:
        config = json.load(json_file)
    return config


class TestSEED15VL:
    def __init__(self, device=None):
        # 读取配置文件
        try:
            cfg = read_config("./peng_utils/openai_cfg.json")
            # 查找SEED-1.5-VL配置，如果没有则使用默认配置
            seed_cfg = None
            for api_cfg in cfg.get("apis", []):
                if "seed" in api_cfg.get("name", "").lower() or "doubao" in api_cfg.get("name", "").lower():
                    seed_cfg = api_cfg
                    break
            
            if seed_cfg is None:
                # 如果没有找到专门的配置，使用第一个配置作为模板
                seed_cfg = cfg.get("apis", [{}])[0]
                print("⚠️ 未找到SEED-1.5-VL专用配置，使用默认配置")
        
        except Exception as e:
            print(f"⚠️ 读取配置文件失败: {e}")
            seed_cfg = {}
        
        # 模型配置
        self.model = "doubao-1-5-thinking-vision-pro-250428"  # SEED-1.5-VL模型ID
        
        # API配置
        self.api_key = seed_cfg.get("api_key", os.environ.get("ARK_API_KEY"))
        if not self.api_key:
            raise ValueError("❌ 请设置ARK_API_KEY环境变量或在配置文件中提供api_key")
        
        # 设置环境变量
        os.environ["ARK_API_KEY"] = self.api_key
        
        # 获取base_url
        base_url = seed_cfg.get("base_url", "https://ark.cn-beijing.volces.com/api/v3")
        
        # 代理设置（如果配置文件中有的话）
        if "http_proxy" in seed_cfg and seed_cfg["http_proxy"]:
            os.environ["http_proxy"] = seed_cfg["http_proxy"]
            print(f"🌐 设置HTTP代理: {seed_cfg['http_proxy']}")
        if "https_proxy" in seed_cfg and seed_cfg["https_proxy"]:
            os.environ["https_proxy"] = seed_cfg["https_proxy"]
            print(f"🌐 设置HTTPS代理: {seed_cfg['https_proxy']}")
        
        # 网络连接测试
        self._test_network_connectivity(base_url)
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key,
            timeout=60.0,  # 增加超时时间到60秒
            max_retries=2,  # 减少SDK内部重试次数，让我们的重试逻辑来处理
        )
        
        self.device = device
        print(f"✅ SEED-1.5-VL 初始化成功")
        print(f"📍 模型: {self.model}")
        print(f"🔗 API地址: {base_url}")

    def _test_network_connectivity(self, base_url):
        """测试网络连接"""
        try:
            parsed_url = urlparse(base_url)
            test_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            print(f"🔍 测试网络连接到: {test_url}")
            
            # 先测试是否需要代理
            try:
                # 不使用代理的直接连接测试
                response = requests.get(test_url, timeout=10, proxies={'http': None, 'https': None})
                print(f"✅ 直连成功，状态码: {response.status_code}")
                return
            except:
                print(f"🔄 直连失败，尝试使用代理...")
            
            # 使用代理的连接测试
            response = requests.get(test_url, timeout=10)
            print(f"✅ 代理连接成功，状态码: {response.status_code}")
            
        except requests.exceptions.Timeout:
            print(f"⚠️ 网络连接超时，可能需要配置代理")
            self._suggest_network_solutions()
        except requests.exceptions.ConnectionError as e:
            print(f"⚠️ 网络连接失败: {str(e)}")
            self._suggest_network_solutions()
        except Exception as e:
            print(f"⚠️ 网络测试异常: {e}")
            self._suggest_network_solutions()

    def _suggest_network_solutions(self):
        """建议网络解决方案"""
        print("\n💡 网络问题可能的解决方案:")
        print("1. 🌐 检查代理设置是否正确")
        print("2. 🔧 尝试更换网络环境（如切换到手机热点）")
        print("3. ⚡ 确认防火墙允许访问 ark.cn-beijing.volces.com")
        print("4. 🔑 联系网络管理员或使用VPN")
        print("5. 📞 检查API服务状态：https://ark.cn-beijing.volces.com")
        
        # 显示当前代理设置
        http_proxy = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY')
        https_proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
        
        if http_proxy or https_proxy:
            print(f"\n🌐 当前代理设置:")
            if http_proxy:
                print(f"   HTTP: {http_proxy}")
            if https_proxy:
                print(f"   HTTPS: {https_proxy}")
        else:
            print(f"\n⚠️ 未检测到代理设置，如果在公司网络，可能需要配置代理")

    def move_to_device(self, device):
        """兼容接口，API调用不需要设备管理"""
        self.device = device
        return

    def generate_image(self, image_list_list, question_list):
        """批量生成图像响应"""
        print("🚀 SEED-1.5-VL generate begins")
        answers = []
        
        for i, (image_list, question) in enumerate(zip(image_list_list, question_list)):
            try:
                print(f"🔧 处理第{i+1}个样本...")
                
                # 处理图像列表
                images = []
                for j, image in enumerate(image_list):
                    try:
                        # 转换为PIL图像
                        img = get_image(np.array(image))
                        
                        # 转换为base64
                        buffered = BytesIO()
                        img.save(buffered, format="JPEG")
                        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        images.append(img_b64)
                        
                        print(f"   📷 图像{j+1}处理完成")
                    except Exception as e:
                        print(f"   ⚠️ 图像{j+1}处理失败: {e}")
                        continue
                
                if not images:
                    print(f"   ⚠️ 第{i+1}个样本没有有效图像，跳过")
                    answers.append("No valid images provided.")
                    continue
                
                # 调用API生成回答
                answer = self.answer(images, question)
                answers.append(answer)
                print(f"✅ 第{i+1}个样本生成成功")
                
            except Exception as e:
                error_msg = f"Error processing sample {i+1}: {str(e)}"
                answers.append(error_msg)
                print(f"❌ 第{i+1}个样本生成失败: {e}")
                traceback.print_exc()

        print("🎉 SEED-1.5-VL generate finish")
        return answers

    def answer(self, base64_image_list, context, max_tokens=500):
        """调用SEED-1.5-VL API生成回答"""
        try:
            # 构建消息内容
            content = []
            
            # 添加图像
            for i, img_b64 in enumerate(base64_image_list):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }
                })
            
            # 添加文本问题
            content.append({
                "type": "text", 
                "text": context
            })
            
            # 调用API
            print(f"🔧 调用SEED-1.5-VL API，图像数量: {len(base64_image_list)}")
            
            # 重试机制
            max_retries = 5  # 增加重试次数
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        wait_time = min(2 ** attempt, 30)  # 指数退避，但最大等待30秒
                        print(f"🔄 重试第{attempt}次，等待{wait_time}秒...")
                        time.sleep(wait_time)
                    
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": content
                            }
                        ],
                        max_tokens=max_tokens,
                        temperature=0.1,  # 降低随机性，提高一致性
                        timeout=90,  # 单次请求超时时间90秒
                    )
                    
                    # 提取回答
                    answer = response.choices[0].message.content
                    print(f"✅ API调用成功，回答长度: {len(answer)}")
                    return answer
                    
                except Exception as e:
                    error_type = type(e).__name__
                    print(f"❌ API调用失败 (尝试{attempt+1}/{max_retries}): {error_type} - {e}")
                    
                    # 特殊处理不同类型的错误
                    if "timeout" in str(e).lower() or "ConnectTimeout" in error_type:
                        print(f"   🌐 网络超时错误，建议检查代理设置或网络连接")
                    elif "ConnectError" in error_type:
                        print(f"   🌐 网络连接错误，请检查代理配置")
                    
                    if attempt == max_retries - 1:
                        # 最后一次重试失败，返回详细错误信息
                        return f"SEED-1.5-VL API调用失败 ({error_type}): {str(e)}"
                    continue
        
        except Exception as e:
            error_msg = f"SEED-1.5-VL API调用失败: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return error_msg

    def generate_image_batch(self, image_list_list, question_list, **kwargs):
        """批量推理接口（兼容接口）"""
        return self.generate_image(image_list_list, question_list)

    def clear_cache(self):
        """清理缓存（兼容接口，API调用无需缓存管理）"""
        pass


# 测试代码
if __name__ == "__main__":
    print("🧪 测试SEED-1.5-VL")
    
    try:
        # 创建测试实例
        model = TestSEED15VL()
        
        # 创建测试图像（随机图像）
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # 测试问题
        test_question = "这是什么？请描述一下这张图片。"
        
        # 调用测试
        result = model.generate_image([[test_image]], [test_question])
        
        print(f"🎯 测试结果: {result}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        traceback.print_exc() 