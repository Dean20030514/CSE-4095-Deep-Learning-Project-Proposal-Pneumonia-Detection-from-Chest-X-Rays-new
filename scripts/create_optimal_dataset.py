"""
创建最优数据集:整合data和merged_dataset的优点
- 从merged_dataset获取完整的唯一图像(含test集)
- 使用85/10/5划分(train/val/test)
- 完全去重
- 分层采样保持类别比例
"""
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict
import random

def compute_file_hash(filepath):
    """计算文件MD5哈希"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def collect_and_deduplicate():
    """从两个数据集收集并去重"""
    print('='*70)
    print('步骤1: 收集所有唯一图像')
    print('='*70)
    
    unique_images = {}  # hash -> (filepath, class)
    
    # 从merged_dataset收集(包含test集)
    print('\n扫描 merged_dataset/chest_xray/ ...')
    merged_base = Path('merged_dataset/chest_xray')
    for split in ['train', 'val', 'test']:
        for cls in ['NORMAL', 'PNEUMONIA']:
            img_dir = merged_base / split / cls
            if not img_dir.exists():
                continue
            
            files = list(img_dir.glob('*.jpeg'))
            print(f'  {split}/{cls}: {len(files)} 个文件')
            
            for img_file in files:
                try:
                    file_hash = compute_file_hash(img_file)
                    if file_hash not in unique_images:
                        unique_images[file_hash] = (img_file, cls)
                except Exception as e:
                    print(f'    警告: 无法读取 {img_file}: {e}')
    
    # 从data收集(以防有额外图像)
    print('\n扫描 data/ ...')
    data_base = Path('data')
    for split in ['train', 'val']:
        for cls in ['NORMAL', 'PNEUMONIA']:
            img_dir = data_base / split / cls
            if not img_dir.exists():
                continue
            
            files = list(img_dir.glob('*.jpeg'))
            added = 0
            for img_file in files:
                try:
                    file_hash = compute_file_hash(img_file)
                    if file_hash not in unique_images:
                        unique_images[file_hash] = (img_file, cls)
                        added += 1
                except Exception as e:
                    print(f'    警告: 无法读取 {img_file}: {e}')
            
            if added > 0:
                print(f'  {split}/{cls}: +{added} 个新图像')
    
    # 按类别组织
    normal_images = []
    pneumonia_images = []
    
    for file_hash, (filepath, cls) in unique_images.items():
        if cls == 'NORMAL':
            normal_images.append(filepath)
        else:
            pneumonia_images.append(filepath)
    
    print(f'\n去重结果:')
    print(f'  NORMAL: {len(normal_images)} 张')
    print(f'  PNEUMONIA: {len(pneumonia_images)} 张')
    print(f'  总计: {len(unique_images)} 张唯一图像')
    
    return normal_images, pneumonia_images

def stratified_split(normal_images, pneumonia_images, 
                     train_ratio=0.85, val_ratio=0.10, test_ratio=0.05, seed=42):
    """分层采样划分为train/val/test"""
    print(f'\n{"="*70}')
    print('步骤2: 三分划分数据集 (85/10/5)')
    print('='*70)
    
    random.seed(seed)
    random.shuffle(normal_images)
    random.shuffle(pneumonia_images)
    
    n_normal = len(normal_images)
    n_pneumonia = len(pneumonia_images)
    
    # 计算划分点
    normal_train_end = int(n_normal * train_ratio)
    normal_val_end = normal_train_end + int(n_normal * val_ratio)
    
    pneumonia_train_end = int(n_pneumonia * train_ratio)
    pneumonia_val_end = pneumonia_train_end + int(n_pneumonia * val_ratio)
    
    splits = {
        'train': {
            'NORMAL': normal_images[:normal_train_end],
            'PNEUMONIA': pneumonia_images[:pneumonia_train_end]
        },
        'val': {
            'NORMAL': normal_images[normal_train_end:normal_val_end],
            'PNEUMONIA': pneumonia_images[pneumonia_train_end:pneumonia_val_end]
        },
        'test': {
            'NORMAL': normal_images[normal_val_end:],
            'PNEUMONIA': pneumonia_images[pneumonia_val_end:]
        }
    }
    
    print('\n划分结果:')
    for split, classes in splits.items():
        normal_count = len(classes['NORMAL'])
        pneumonia_count = len(classes['PNEUMONIA'])
        total = normal_count + pneumonia_count
        percentage = total / (n_normal + n_pneumonia) * 100
        print(f'  {split:5s}: {total:4d} 张 ({percentage:5.1f}%) - '
              f'NORMAL: {normal_count:4d}, PNEUMONIA: {pneumonia_count:4d}')
    
    return splits

def copy_files(splits, output_dir='data_final'):
    """复制文件到新目录"""
    print(f'\n{"="*70}')
    print('步骤3: 创建最终数据集')
    print('='*70)
    
    output_path = Path(output_dir)
    
    if output_path.exists():
        print(f'\n删除旧目录: {output_path}')
        shutil.rmtree(output_path)
    
    print(f'创建目录: {output_path}\n')
    
    total_copied = 0
    for split, classes in splits.items():
        for cls, files in classes.items():
            target_dir = output_path / split / cls
            target_dir.mkdir(parents=True, exist_ok=True)
            
            print(f'复制 {split}/{cls}: {len(files)} 张图像', end='')
            
            for src_file in files:
                dst_file = target_dir / src_file.name
                counter = 1
                while dst_file.exists():
                    dst_file = target_dir / f'{src_file.stem}_{counter}{src_file.suffix}'
                    counter += 1
                
                shutil.copy2(src_file, dst_file)
                total_copied += 1
            
            print(' ✓')
    
    print(f'\n总计复制: {total_copied} 张图像')
    return output_path

def verify_dataset(output_path):
    """验证数据集完整性"""
    print(f'\n{"="*70}')
    print('步骤4: 验证数据集')
    print('='*70)
    
    print('\n最终统计:')
    total = 0
    for split in ['train', 'val', 'test']:
        for cls in ['NORMAL', 'PNEUMONIA']:
            path = output_path / split / cls
            if path.exists():
                count = len(list(path.glob('*.jpeg')))
                print(f'  {split}/{cls}: {count}')
                total += count
    
    size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / (1024**3)
    print(f'\n  总计: {total} 张图像')
    print(f'  大小: {size:.2f} GB')
    
    # 检查是否有重复
    print('\n检查内部重复...')
    hashes = set()
    duplicates = 0
    for img in output_path.rglob('*.jpeg'):
        h = compute_file_hash(img)
        if h in hashes:
            duplicates += 1
        else:
            hashes.add(h)
    
    if duplicates == 0:
        print('  ✓ 无重复文件')
    else:
        print(f'  ⚠️  发现 {duplicates} 个重复文件!')
    
    return total

def cleanup_old_datasets():
    """清理旧数据集"""
    print(f'\n{"="*70}')
    print('步骤5: 清理旧数据集')
    print('='*70)
    
    old_dirs = ['data', 'merged_dataset']
    
    print('\n即将删除:')
    for dir_name in old_dirs:
        path = Path(dir_name)
        if path.exists():
            size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / (1024**3)
            print(f'  - {dir_name}/ ({size:.2f} GB)')
    
    response = input('\n确认删除旧数据集? (yes/no): ').strip().lower()
    
    if response == 'yes':
        for dir_name in old_dirs:
            path = Path(dir_name)
            if path.exists():
                print(f'  删除 {dir_name}/ ...', end='')
                shutil.rmtree(path)
                print(' ✓')
        print('\n旧数据集已清理!')
        return True
    else:
        print('\n保留旧数据集')
        return False

def rename_final_dataset():
    """将data_final重命名为data"""
    print(f'\n{"="*70}')
    print('步骤6: 重命名为data')
    print('='*70)
    
    if Path('data_final').exists():
        Path('data_final').rename('data')
        print('\n✓ data_final/ → data/')
    else:
        print('\n⚠️  data_final/ 不存在')

def main():
    print('╔' + '='*68 + '╗')
    print('║' + ' '*20 + '创建最优数据集' + ' '*32 + '║')
    print('╚' + '='*68 + '╝\n')
    
    print('目标: 整合data和merged_dataset的优点')
    print('  ✓ 完全去重')
    print('  ✓ 包含独立测试集')
    print('  ✓ 科学的85/10/5划分')
    print('  ✓ 分层采样保持类别比例')
    print('  ✓ 固定随机种子(可复现)\n')
    
    # 1. 收集去重
    normal_images, pneumonia_images = collect_and_deduplicate()
    
    # 2. 三分划分
    splits = stratified_split(normal_images, pneumonia_images,
                             train_ratio=0.85, val_ratio=0.10, test_ratio=0.05,
                             seed=42)
    
    # 3. 复制文件
    output_path = copy_files(splits, 'data_final')
    
    # 4. 验证
    total = verify_dataset(output_path)
    
    # 5. 清理旧数据集
    cleaned = cleanup_old_datasets()
    
    # 6. 重命名
    if cleaned:
        rename_final_dataset()
    
    # 最终报告
    print('\n' + '╔' + '='*68 + '╗')
    print('║' + ' '*25 + '完成!' + ' '*37 + '║')
    print('╚' + '='*68 + '╝\n')
    
    if cleaned:
        print('最优数据集已创建: data/')
    else:
        print('最优数据集已创建: data_final/')
        print('(旧数据集保留,请手动删除后重命名data_final为data)')
    
    print('\n数据集特点:')
    print('  ✓ 完全去重,无冗余文件')
    print('  ✓ 85% train (训练集)')
    print('  ✓ 10% val (验证集,用于early stopping和超参数调优)')
    print('  ✓ 5% test (测试集,用于最终评估)')
    print('  ✓ 分层采样,类别比例一致')
    print('  ✓ 随机种子42,完全可复现')
    
    print('\n使用方法:')
    print('  python -m src.train --data_root data --save_dir runs/experiment')
    print('  python -m src.eval --checkpoint runs/experiment/best.pt --data_root data --split test')

if __name__ == '__main__':
    main()
