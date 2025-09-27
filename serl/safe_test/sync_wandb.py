#!/usr/bin/env python3
"""
同步并检查WandB运行状态
"""

import os
import subprocess
import glob

def sync_all_wandb_runs():
    """同步所有本地WandB运行到云端"""
    
    print("🔄 检查并同步WandB运行...")
    
    wandb_dir = "/home/xyz/Desktop/xluo/RL_safe/serl/safe_test/wandb"
    
    if not os.path.exists(wandb_dir):
        print("❌ WandB目录不存在")
        return
    
    # 查找所有运行目录
    run_dirs = []
    for item in os.listdir(wandb_dir):
        full_path = os.path.join(wandb_dir, item)
        if os.path.isdir(full_path) and (item.startswith("run-") or item.startswith("offline-run-")):
            run_dirs.append((item, full_path))
    
    print(f"📁 找到 {len(run_dirs)} 个运行目录:")
    for name, path in run_dirs:
        print(f"  - {name}")
    
    # 同步每个运行
    synced_runs = []
    for name, path in run_dirs:
        try:
            print(f"\n🔄 同步 {name}...")
            result = subprocess.run(
                ["wandb", "sync", path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print(f"✅ {name} 同步成功")
                # 从输出中提取URL
                if "Syncing:" in result.stderr:
                    lines = result.stderr.split('\n')
                    for line in lines:
                        if "Syncing:" in line and "https://" in line:
                            url = line.split("Syncing: ")[1].split(" ...")[0]
                            synced_runs.append((name, url))
                            break
            else:
                print(f"⚠️  {name} 同步失败: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {name} 同步超时")
        except Exception as e:
            print(f"❌ {name} 同步出错: {e}")
    
    # 显示结果
    print(f"\n📊 同步结果:")
    print("=" * 60)
    
    if synced_runs:
        print("✅ 成功同步的运行:")
        for name, url in synced_runs:
            print(f"  🔗 {name}: {url}")
        
        print(f"\n🌐 查看所有运行:")
        print("  https://wandb.ai/")
        
        # 提取项目列表
        projects = set()
        for name, url in synced_runs:
            if "/runs/" in url:
                project_part = url.split("/runs/")[0]
                if "/" in project_part:
                    project = project_part.split("/")[-1]
                    projects.add(project)
        
        print(f"\n📂 涉及的项目:")
        for project in sorted(projects):
            print(f"  - {project}")
            
    else:
        print("❌ 没有成功同步任何运行")
    
    return len(synced_runs)

def check_wandb_status():
    """检查WandB状态"""
    print("\n🔍 检查WandB状态...")
    
    try:
        result = subprocess.run(["wandb", "status"], capture_output=True, text=True)
        print("WandB状态:")
        print(result.stdout)
    except Exception as e:
        print(f"❌ 无法检查WandB状态: {e}")

def main():
    """主函数"""
    print("🚀 WandB同步工具")
    print("=" * 50)
    
    # 检查WandB状态
    check_wandb_status()
    
    # 同步所有运行
    synced_count = sync_all_wandb_runs()
    
    print(f"\n📈 总结: 成功同步了 {synced_count} 个运行")
    
    if synced_count > 0:
        print("\n💡 提示:")
        print("1. 访问 https://wandb.ai/ 查看所有运行")
        print("2. 如果看不到最新数据，请刷新页面")
        print("3. 检查项目名称是否正确")

if __name__ == "__main__":
    main()
