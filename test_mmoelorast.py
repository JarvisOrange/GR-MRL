#!/usr/bin/env python3
"""
测试 MMOELoraST 模块的导入和基本功能
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_mmoelorast_import():
    """测试 MMOELoraST 模块的导入"""
    try:
        from Model.MLoRA.peft.tuners.mmoelorast import (
            MMOELoraSTLinear, 
            MMOELoraSTConfig,
            Expert,
            Gate,
            MMOELinearA,
            MMOELinearB
        )
        print("✓ MMOELoraST 模块导入成功")
        return True
    except Exception as e:
        print(f"✗ MMOELoraST 模块导入失败: {e}")
        return False

def test_expert_creation():
    """测试 Expert 类的创建"""
    try:
        from Model.MLoRA.peft.tuners.mmoelorast import Expert
        
        expert = Expert(in_features=64, out_features=32)
        x = torch.randn(2, 64)
        output = expert(x)
        
        assert output.shape == (2, 32), f"Expected shape (2, 32), got {output.shape}"
        print("✓ Expert 类创建和推理成功")
        return True
    except Exception as e:
        print(f"✗ Expert 类测试失败: {e}")
        return False

def test_gate_creation():
    """测试 Gate 类的创建"""
    try:
        from Model.MLoRA.peft.tuners.mmoelorast import Gate
        
        # 创建临时的 pattern 文件
        pattern_embed = torch.randn(8, 128)
        torch.save(pattern_embed, 'temp_pattern.pt')
        
        gate = Gate(
            input_size=64, 
            expert_num=8, 
            gate_embed_dim=128, 
            gate_embed_path='temp_pattern.pt',
            expert_top_k=2
        )
        
        x = torch.randn(2, 10, 64)
        output = gate(x)
        
        assert output.shape == (2, 8), f"Expected shape (2, 8), got {output.shape}"
        print("✓ Gate 类创建和推理成功")
        
        # 清理临时文件
        os.remove('temp_pattern.pt')
        return True
    except Exception as e:
        print(f"✗ Gate 类测试失败: {e}")
        # 清理临时文件
        if os.path.exists('temp_pattern.pt'):
            os.remove('temp_pattern.pt')
        return False

def test_mmoelinear_creation():
    """测试 MMOELinearA 和 MMOELinearB 类的创建"""
    try:
        from Model.MLoRA.peft.tuners.mmoelorast import MMOELinearA, MMOELinearB
        
        # 测试 MMOELinearA
        mmoe_a = MMOELinearA(in_features=64, out_features=32, expert_num=4)
        x = torch.randn(2, 64)
        outputs_a = mmoe_a(x)
        
        assert len(outputs_a) == 4, f"Expected 4 outputs, got {len(outputs_a)}"
        assert all(out.shape == (2, 8) for out in outputs_a), "Output shapes mismatch"
        
        # 测试 MMOELinearB
        mmoe_b = MMOELinearB(in_features=32, out_features=16, expert_num=4)
        outputs_b = mmoe_b(outputs_a)
        
        assert len(outputs_b) == 4, f"Expected 4 outputs, got {len(outputs_b)}"
        assert all(out.shape == (2, 16) for out in outputs_b), "Output shapes mismatch"
        
        print("✓ MMOELinearA 和 MMOELinearB 类创建和推理成功")
        return True
    except Exception as e:
        print(f"✗ MMOELinear 类测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试 MMOELoraST 模块...")
    
    tests = [
        test_mmoelorast_import,
        test_expert_creation,
        test_gate_creation,
        test_mmoelinear_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！MMOELoraST 模块工作正常。")
    else:
        print("⚠️  部分测试失败，请检查相关代码。")

if __name__ == "__main__":
    main() 