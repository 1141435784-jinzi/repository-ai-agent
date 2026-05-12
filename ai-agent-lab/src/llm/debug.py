#!/usr/bin/env python3
"""
调试模型可用性问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
import json

def debug_ollama_models():
    """调试Ollama模型"""
    OLLAMA_BASE_URL = "http://localhost:11434"
    
    try:
        # 直接调用Ollama API
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        print(f"API响应状态: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"原始响应: {json.dumps(data, indent=2, ensure_ascii=False)}")
            
            models = data.get("models", [])
            print(f"\n解析的模型列表 ({len(models)}个):")
            for model in models:
                print(f"  名称: {model.get('name')}")
                print(f"  大小: {model.get('size', 0):,} bytes")
                print(f"  修改时间: {model.get('modified_at', '')}")
                print()
        else:
            print(f"API错误: {response.text}")
            
    except Exception as e:
        print(f"调试失败: {e}")

def check_config():
    """检查配置"""
    from config import OLLAMA_MODELS, OLLAMA_DEFAULT_MODEL
    
    print("配置检查:")
    print(f"  OLLAMA_MODELS: {OLLAMA_MODELS}")
    print(f"  OLLAMA_DEFAULT_MODEL: {OLLAMA_DEFAULT_MODEL}")
    
    # 检查每个模型
    for model in OLLAMA_MODELS:
        print(f"\n检查模型: {model}")
        try:
            response = requests.post(
                "http://localhost:11434/api/show",
                json={"name": model},
                timeout=5
            )
            if response.status_code == 200:
                print(f"  ✅ 模型可用")
            else:
                print(f"  ❌ 模型不可用: HTTP {response.status_code}")
        except Exception as e:
            print(f"  ❌ 检查失败: {e}")

if __name__ == "__main__":
    print("Ollama模型调试")
    print("=" * 50)
    
    debug_ollama_models()
    print("\n" + "=" * 50)
    check_config()