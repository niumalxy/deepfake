"""
超分辨率工具简单测试脚本

这个脚本可以直接运行，测试超分辨率功能（本地和远程模式）
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from PIL import Image
from segment_agent.skills.tools.sr_tool import (
    super_resolution,
    super_resolution_save,
    get_local_sr_model,
    get_remote_sr_model
)

def test_local_mode():
    """测试本地模式"""
    print("=== 测试本地模式 ===")
    
    test_image_path = "test.png"
    
    if not os.path.exists(test_image_path):
        print(f"警告: 测试图像 {test_image_path} 不存在")
        print("请将测试图像命名为 test.png 并放在项目根目录下")
        return False
    
    try:
        img = Image.open(test_image_path)
        print(f"原始图像尺寸: {img.size}")
        
        # 测试本地模式
        sr_img = super_resolution(img, scale=2, mode="local")
        print(f"本地超分辨率图像尺寸: {sr_img.size}")
        
        output_path = "test_sr_local_output.png"
        sr_img.save(output_path)
        print(f"本地超分辨率图像已保存到 {output_path}")
        
        print("本地模式测试成功！")
        return True
        
    except Exception as e:
        print(f"本地模式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_remote_mode():
    """测试远程模式（需要配置 API）"""
    print("\n=== 测试远程模式 ===")
    
    test_image_path = "test.png"
    
    if not os.path.exists(test_image_path):
        print(f"警告: 测试图像 {test_image_path} 不存在")
        return False
    
    # 注意：这里使用示例 URL，实际使用时需要替换为真实的 API 端点
    api_url = os.environ.get("SR_API_URL", "http://localhost:8000/api/super-resolution")
    api_key = os.environ.get("SR_API_KEY", "")
    
    print(f"API URL: {api_url}")
    print(f"API Key: {'已配置' if api_key else '未配置'}")
    
    try:
        img = Image.open(test_image_path)
        print(f"原始图像尺寸: {img.size}")
        
        # 测试远程模式
        sr_img = super_resolution(
            img,
            scale=2,
            mode="remote",
            api_url=api_url,
            api_key=api_key,
            timeout=10
        )
        print(f"远程超分辨率图像尺寸: {sr_img.size}")
        
        output_path = "test_sr_remote_output.png"
        sr_img.save(output_path)
        print(f"远程超分辨率图像已保存到 {output_path}")
        
        print("远程模式测试成功！")
        return True
        
    except Exception as e:
        print(f"远程模式测试失败（这是预期的，如果没有配置真实的 API）: {e}")
        print("提示: 如果要测试远程模式，请设置环境变量 SR_API_URL 和 SR_API_KEY")
        return False

def test_direct_model_classes():
    """测试直接使用模型类"""
    print("\n=== 测试直接使用模型类 ===")
    
    test_image_path = "test.png"
    
    if not os.path.exists(test_image_path):
        print(f"警告: 测试图像 {test_image_path} 不存在")
        return False
    
    try:
        img = Image.open(test_image_path)
        
        # 测试本地模型类
        local_model = get_local_sr_model(model_type="EDSR")
        sr_img = local_model.super_resolution(img, scale=2)
        sr_img.save("test_sr_direct_local.png")
        print("直接使用本地模型类测试成功！")
        
        # 测试远程模型类（会降级到插值，因为没有真实 API）
        remote_model = get_remote_sr_model(
            api_url="http://localhost:8000/api/super-resolution",
            api_key=""
        )
        sr_img = remote_model.super_resolution(img, scale=2)
        sr_img.save("test_sr_direct_remote.png")
        print("直接使用远程模型类测试成功（降级到插值）！")
        
        return True
        
    except Exception as e:
        print(f"直接使用模型类测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_save_functions():
    """测试保存函数"""
    print("\n=== 测试保存函数 ===")
    
    test_image_path = "test.png"
    
    if not os.path.exists(test_image_path):
        print(f"警告: 测试图像 {test_image_path} 不存在")
        return False
    
    try:
        img = Image.open(test_image_path)
        
        # 测试本地保存函数
        super_resolution_save(
            img,
            "test_sr_save_local.png",
            scale=2,
            mode="local"
        )
        print("本地保存函数测试成功！")
        
        # 测试远程保存函数（会降级到插值）
        super_resolution_save(
            img,
            "test_sr_save_remote.png",
            scale=2,
            mode="remote",
            api_url="http://localhost:8000/api/super-resolution",
            api_key=""
        )
        print("远程保存函数测试成功（降级到插值）！")
        
        return True
        
    except Exception as e:
        print(f"保存函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试超分辨率工具...\n")
    
    results = []
    
    results.append(("本地模式", test_local_mode()))
    results.append(("远程模式", test_remote_mode()))
    results.append(("直接使用模型类", test_direct_model_classes()))
    results.append(("保存函数", test_save_functions()))
    
    print("\n" + "="*50)
    print("测试结果汇总:")
    print("="*50)
    
    for name, success in results:
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{name}: {status}")
    
    print("\n测试完成！")
