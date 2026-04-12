# 查看结果：tmux attach -t 0
"""
ERA5 2020 高空数据 NC → NPY 转换脚本
======================================
严格对应 data_loader_npyfiles.py 预期的目录/文件结构：

  root_dir/
    2020/
      2020-01-01/
        00-00-00-z-1000.0.npy   [721, 1440] float32
        00-00-00-q-1000.0.npy
        ...
        06-00-00-t-50.0.npy
        ...
        18-00-00-v-50.0.npy
      2020-01-02/
        ...

输入 NC 文件命名规则（高空）：
  2020_01_01-06.nc, 2020_01_07-12.nc, 2020_01_13-18.nc, ...

变量: z, q, u, v, t  (higher_features)
气压层: 1000,925,850,700,600,500,400,300,250,200,150,100,50 hPa
时次: 00:00, 06:00, 12:00, 18:00 UTC（严格只保存这四个时次）
分辨率: ERA5 原始 0.25° (721×1440)，不做任何插值/裁剪

运行日志写入: <OUT_DIR>/nc2npy_2020.log
"""

import os
import glob
import logging
import traceback
from datetime import datetime, timezone

import numpy as np
import xarray as xr
from tqdm import tqdm


# ─────────────────────────── 配置区 ───────────────────────────
# NC 文件所在目录（高空数据，不含 single 子目录）
NC_DIR  = "./data"

# 输出 root_dir（须与 vamoe.yaml 中 root_dir 一致）
OUT_DIR = "./data/era5_npy"

YEAR = 2020

# 与 data_loader_npyfiles.py 完全一致
HIGHER_FEATURES  = ['z', 'q', 'u', 'v', 't']
PRESSURE_LEVELS  = [1000.0, 925.0, 850.0, 700.0, 600.0, 500.0,
                    400.0, 300.0, 250.0, 200.0, 150.0, 100.0, 50.0]
TARGET_HOURS     = {0, 6, 12, 18}   # 只保存这四个时次

# 气压坐标候选名（兼容新旧版 ERA5）
LEVEL_DIM_CANDIDATES = ['pressure_level', 'level', 'plev', 'lev', 'isobaricInhPa']
TIME_DIM_CANDIDATES  = ['valid_time', 'time', 'initial_time']
# ──────────────────────────────────────────────────────────────


# ─────────────────────── 日志初始化 ──────────────────────────
def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("nc2npy")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # 文件 handler（DEBUG 及以上全记录）
    fh = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 控制台 handler（INFO 及以上）
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger
# ──────────────────────────────────────────────────────────────


def detect_dim(ds: xr.Dataset, candidates: list) -> str | None:
    for name in candidates:
        if name in ds.dims:
            return name
    return None


def get_pressure_vals(ds: xr.Dataset, level_dim: str) -> np.ndarray:
    vals = ds.coords[level_dim].values.astype(float)
    if vals.max() > 2000:          # Pa → hPa
        vals = vals / 100.0
    return vals


def t64_to_py(t64) -> datetime:
    """numpy datetime64 → Python datetime (UTC, naive)"""
    ts = np.datetime64(t64, "ns")
    epoch = np.datetime64(0, "ns")
    one_s = np.timedelta64(1, "s") / np.timedelta64(1, "ns")
    unix_s = int((ts - epoch) / one_s)
    return datetime.utcfromtimestamp(unix_s)


def process_nc_file(nc_path: str, log: logging.Logger,
                    stats: dict) -> None:
    """处理单个压力层 NC 文件，将其中 00/06/12/18 时次切片存为 npy。"""
    fname = os.path.basename(nc_path)
    log.info(f"━━━ 开始处理: {fname}")

    try:
        ds = xr.open_dataset(nc_path, engine="netcdf4")
    except Exception as e:
        log.error(f"无法打开文件 {fname}: {e}")
        stats["failed_files"] += 1
        return

    # ── 识别维度 ──
    time_dim  = detect_dim(ds, TIME_DIM_CANDIDATES)
    level_dim = detect_dim(ds, LEVEL_DIM_CANDIDATES)

    if not time_dim:
        log.error(f"{fname}: 未找到时间维度，已有维度={list(ds.dims)}")
        ds.close(); stats["failed_files"] += 1; return
    if not level_dim:
        log.error(f"{fname}: 未找到气压层维度，已有维度={list(ds.dims)}")
        ds.close(); stats["failed_files"] += 1; return

    log.debug(f"{fname}: 时间维度={time_dim}, 气压层维度={level_dim}")

    # ── 检查变量 ──
    missing_vars = [v for v in HIGHER_FEATURES if v not in ds.data_vars]
    if missing_vars:
        log.warning(f"{fname}: 缺少变量 {missing_vars}，跳过。"
                    f"文件中变量={list(ds.data_vars)}")
        ds.close(); stats["failed_files"] += 1; return

    pressure_vals = get_pressure_vals(ds, level_dim)
    log.debug(f"{fname}: 气压层={pressure_vals.tolist()}")

    time_vals = ds.coords[time_dim].values
    n_times   = len(time_vals)
    log.info(f"{fname}: 共 {n_times} 个时次")

    skipped_times  = 0
    written_files  = 0
    skipped_files  = 0
    error_files    = 0

    with tqdm(total=n_times, desc=f"  {fname[:30]}", unit="时次",
              leave=False, colour="yellow", dynamic_ncols=True) as pbar:

        for ti, t64 in enumerate(time_vals):
            dt   = t64_to_py(t64)
            hour = dt.hour

            if hour not in TARGET_HOURS:
                skipped_times += 1
                pbar.update(1)
                continue

            # 输出目录: OUT_DIR/2020/2020-01-01/
            date_str = f"{dt.year}-{dt.month:02d}-{dt.day:02d}"
            day_dir  = os.path.join(OUT_DIR, str(dt.year), date_str)
            os.makedirs(day_dir, exist_ok=True)

            # 文件名前缀: 00-00-00-
            prefix = f"{hour:02d}-00-00-"

            for var in HIGHER_FEATURES:
                for plev in PRESSURE_LEVELS:
                    npy_name = f"{prefix}{var}-{plev}.npy"
                    npy_path = os.path.join(day_dir, npy_name)

                    if os.path.exists(npy_path):
                        skipped_files += 1
                        continue

                    try:
                        # 找最近气压层索引
                        plev_idx = int(np.argmin(np.abs(pressure_vals - plev)))

                        data = ds[var].isel(
                            **{time_dim: ti, level_dim: plev_idx}
                        ).values.astype(np.float32)

                        # 验证 shape（ERA5 0.25° 应为 721×1440）
                        if data.ndim != 2:
                            log.warning(f"  {npy_name}: 预期2D数据，实际shape={data.shape}")
                        if data.shape[0] not in (720, 721) or data.shape[1] != 1440:
                            log.warning(f"  {npy_name}: 非预期shape={data.shape}")

                        np.save(npy_path, data)
                        written_files += 1

                    except Exception as e:
                        log.error(f"  保存 {npy_name} 失败: {e}\n"
                                  f"    {traceback.format_exc(limit=2)}")
                        error_files += 1

            pbar.update(1)

    ds.close()

    stats["written"]  += written_files
    stats["skipped"]  += skipped_files
    stats["errors"]   += error_files

    log.info(f"{fname}: 完成 | 新建={written_files} 跳过={skipped_files} "
             f"错误={error_files} 非目标时次={skipped_times}")


def find_nc_files(nc_dir: str, year: int) -> list:
    """查找所有高空压力层 NC 文件（排除 single 子目录）。"""
    pattern = os.path.join(nc_dir, f"{year}_*.nc")
    files   = sorted(glob.glob(pattern))
    # 排除 single 目录下的文件（surface data）
    files   = [f for f in files if "single" not in f.replace("\\", "/")]
    return files


def main():
    log_path = os.path.join(OUT_DIR, "nc2npy_2020.log")
    log = setup_logger(log_path)

    log.info("=" * 62)
    log.info(f"ERA5 {YEAR} 高空数据 NC → NPY 转换任务启动")
    log.info(f"输入目录 : {NC_DIR}")
    log.info(f"输出目录 : {OUT_DIR}")
    log.info(f"日志文件 : {log_path}")
    log.info(f"变量     : {HIGHER_FEATURES}")
    log.info(f"气压层   : {PRESSURE_LEVELS}")
    log.info(f"目标时次 : {sorted(TARGET_HOURS)} UTC")
    log.info("=" * 62)

    nc_files = find_nc_files(NC_DIR, YEAR)

    if not nc_files:
        log.error(f"未找到任何 {YEAR}_*.nc 文件，请检查 NC_DIR: {NC_DIR}")
        return

    log.info(f"共找到 {len(nc_files)} 个高空 NC 文件:")
    for f in nc_files:
        log.info(f"  {os.path.basename(f)}")

    os.makedirs(OUT_DIR, exist_ok=True)

    stats = {"written": 0, "skipped": 0, "errors": 0, "failed_files": 0}

    with tqdm(total=len(nc_files), desc="总体进度", unit="文件",
              colour="cyan", dynamic_ncols=True) as pbar_total:
        for nc_path in nc_files:
            process_nc_file(nc_path, log, stats)
            pbar_total.update(1)

    log.info("=" * 62)
    log.info(f"转换完成！")
    log.info(f"  新建 npy 文件 : {stats['written']}")
    log.info(f"  已存在跳过   : {stats['skipped']}")
    log.info(f"  写入错误     : {stats['errors']}")
    log.info(f"  NC 文件失败  : {stats['failed_files']}")
    log.info(f"  输出目录     : {OUT_DIR}")
    log.info("=" * 62)

    # ── 快速验证：输出一个样本文件信息 ──
    log.info("快速验证:")
    for root, dirs, files in os.walk(OUT_DIR):
        npys = [f for f in files if f.endswith(".npy")]
        if npys:
            sample_path = os.path.join(root, sorted(npys)[0])
            try:
                arr = np.load(sample_path)
                log.info(f"  样本: {sample_path}")
                log.info(f"  shape={arr.shape}  dtype={arr.dtype}"
                         f"  min={arr.min():.4f}  max={arr.max():.4f}")
            except Exception as e:
                log.warning(f"  验证读取失败: {e}")
            break


if __name__ == "__main__":
    main()
