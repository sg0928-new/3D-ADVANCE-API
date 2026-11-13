# 性能与资源占用

所有数字均由 `benchmarks/run_bench.py` 在本机实测得出，不是估算。
复现：`python benchmarks/run_bench.py --md outputs/bench.md --json outputs/bench.json`。

测试机：Windows 本地、Python 3.13.16、numpy 2.5.4、Intel64 Family 6 Model 183
（单进程，无 BLAS 多线程调优；库本身不使用 scipy/numba/GPU）。

## 1. 吞吐

| 分组 | 基准 | 数值 | 单位 | 说明 |
|---|---|---|---|---|
| points | voxel_downsample | 5 884 | k pts/s | 1,000,000 点 → 1 cm 级体素 |
| points | estimate_normals (PCA) | 135.0 | k pts/s | 999,513 点，半径 0.10 m |
| points | remove_statistical_outliers | 311.0 | k pts/s | 同规模，k=16 |
| points | farthest_point_sample | 45.4 | k pts/s | 从 1M 中取 2048 点 |
| registration | icp 8k↔8k, 30 次迭代 | 6 165 | ms | 无 KD-tree，哈希网格 kNN |
| io | PLY binary write | 50 353 | k pts/s | 500k 点，7.6 MB |
| io | PLY binary read | 36 927 | k pts/s | 同上 |
| render | 点渲染 960×640×2x | 458 | k pts/s | 46,124 点，1.9 px 喷溅 |
| render | 网格渲染 960×640×2x | 38.9 | k faces/s | 238,852 面，density 1.5 |
| recon | tsdf_fuse（喷溅） | 13.9 | M voxels/s | 5.99 M 体素，29.6 万被观测 |
| recon | reconstruct 端到端 | 147.4 | k faces/s | 47.6 万面 / 3.23 s |
| lidar | 模拟 64 线扫描 | 390.3 | k rays/s | 10.35 万回波 |
| lidar | range image 投影 | 20 126 | k pts/s | 10.35 万点 |
| lidar | analyze_scan | 1 007 | k pts/s | 地面+聚类+OBB 一次算完 |

## 2. 内存（逐步骤实测 RSS）

用独立探针逐步测量（`tests/_debug_mem.py` 的同类方法），读数是 *进程驻留内存*，
不是分配量估算。

| 步骤 | 增量 | 说明 |
|---|---|---|
| voxel_downsample 1M 点 | +23 MB | 只需 key 数组与分组 index，分块处理 |
| estimate_normals 1M 点 | +23 MB 常驻，峰值 +350 MB | 邻域对缓存（4M pair budget） |
| remove_statistical_outliers 1M 点 | +5 MB | 一次半径查询 + 分块 top-k |
| tsdf_fuse 6.0 M 体素 | +30 MB | 场 24 MB（float32）＋观测掩膜 |
| reconstruct 端到端（6.0 M 体素） | 常驻 160 MB | 含平滑、提面、网格 |
| 网格渲染 476 k 面 | 峰值 ≈ 300 MB | 8192 面分块 + 25 万喷溅分块 + 逐像素最小深度归并 |
| 基准全流程进程峰值 | 618 MB | 1M 点 + 重建 + 渲染一气跑完；单独探针序列（含 1M 法线）峰值 700 MB |

三条硬约束写在代码里，不是靠调用者自觉：

1. **体素总量上限 24 M（≈96 MB 场）**。请求更细的体素时自动粗化，实际体素值
   在 `stats["voxel"]` 里返回——不会静默 OOM。
2. **邻域对预算 `PAIR_BUDGET = 4,000,000`**（`adv3d/_backend.py`），超过就把
   查询切块，峰值内存与点数解耦。
3. **渲染分块**：三角形按 8192 面、喷溅按 25 万碟片分批；超过单面上限的
   大三角先做一次四分裂，所以一个铺满屏幕的面不会退化成稀疏点阵。

   代价与取舍：分块之后不能再靠“整幅按深度排序”实现遮挡，所以每个喷溅块内部先
   用 `lexsort` 把样本按 (像素, 深度) 归并成“每像素最近的一个”，再与缓冲里已有的
   深度比较后才写入。这让跨块遮挡严格正确（早期版本按块顺序写色，交叉处会出现
   亮斑），但把网格渲染从 ~52 k faces/s 拉到 ~39 k faces/s——我们选正确性。

## 3. 设计取向（为什么能这样）

| 选择 | 替代方案 | 理由 |
|---|---|---|
可分离的**哈希网格**（`GridIndex`）取代 KD-tree | scipy.spatial.cKDTree | 零依赖；批量邻域在 C 层一次 `searchsorted` 完成；索引 8 B/点 |
**TSDF + Surface Nets** 取代 Marching Cubes | 256 张查找表 | 表少、代码短、天然水密；对偶面片比 MC 更平滑 |
**屏幕空间自适应三角形喷溅** 取代逐三角形扫描线 | 纯 Python 光栅循环 | 采样数按投影面积分配，一次向量化写缓冲，无逐面 Python 循环 |
**全部 numpy 向量化 + 显式分块** 取代 numba/C 扩展 | Cython、pybind11 | 单 wheel 到处能跑；峰值内存可控是显式设计，不靠运气 |
**程序化合成数据** 取代数据集下载 | Open3D / ModelNet | 示例与基准零下载、可复现、可作为回归基线 |

## 4. 已知瓶颈（如实记录）

| 项目 | 现状 | 原因 | 建议 |
|---|---|---|---|
| `icp` 8k×8k / 30 迭代 | 6.2 s | 每次迭代重做一次半径 kNN（无增量索引） | 上采样到 4k 点，或把 `max_points` 调小；迭代 10–15 次通常足够 |
| `estimate_normals` 1M 点 | 7.5 s，峰值 +350 MB | 邻域对的 lexsort 合并 | 先体素降采样（6 M pts/s，代价极低）；半径别超过平均间距 3 倍 |
| 重建细节（1.4 cm 体素） | 表面有体素级“肋纹” | 表面网顶点是格点分辨率下的对偶点，14 cm 管径在 1.4 cm 体素下只有 ~63 个体素绕一圈 | 更细体素（受 24 M 上限）、多一次体素模糊（`voxel_smooth=3`）、或 `smooth_laplacian` 迭代提高 |
| 网格渲染 | ~39 k faces/s | 纯 numpy 喷溅，且每个像素做精确最小深度归并（见下） | 大网格先 `decimate_voxel`；或降 supersample=1（约 3–4 倍） |
| LAS 写入 | 仅 1.2 格式 0/2 | 未实现 1.4 的 LASzip | 需要 1.4 请自行转 PLY |

## 5. 复现与对比建议

```powershell
python tests\smoke_core.py       # 索引 / 体素 / 法线 / ICP / 网格算子
python tests\smoke_io.py         # 六种格式往返 + 500k PLY 计时
python tests\smoke_render.py     # 原语符号体积断言 + 四联渲染
python tests\smoke_pipeline.py   # 重建偏差断言 + LiDAR 报告断言
python benchmarks\run_bench.py   # 上表
```

比对本机与服务器时只需比较 **每秒点数**（points/ms），因为绝对毫秒受缓存、
温度和后台负载影响很大。上表数字来自一台普通消费级机器，可作为下界参考。
