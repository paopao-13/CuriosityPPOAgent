"""VRAM 监控工具"""
import torch


def get_vram_usage():
    """返回 (allocated_mb, peak_mb)，无 CUDA 时返回 (0, 0)"""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.memory_allocated() / 1e6
    peak = torch.cuda.max_memory_allocated() / 1e6
    return allocated, peak


def reset_peak_vram():
    """重置峰值显存统计"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def empty_cache():
    """清空 CUDA 缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def vram_summary():
    """返回格式化的 VRAM 字符串"""
    allocated, peak = get_vram_usage()
    return f"VRAM: allocated={allocated:.0f}MB, peak={peak:.0f}MB"
