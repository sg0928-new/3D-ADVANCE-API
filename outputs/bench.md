| group | benchmark | value | unit | note |
|---|---|---|---|---|
| points | voxel_downsample | 5883.6 | k pts/s | 1,000,000 pts |
| points | estimate_normals (PCA) | 135.0 | k pts/s | 999,513 pts |
| points | remove_statistical_outliers | 311.0 | k pts/s | 999,513 pts, k=16 |
| points | farthest_point_sample | 45.4 | k pts/s | 2048 out of 1M |
| registration | icp 8k<->8k, 30 it | 6164.9 | ms | 6-DoF, kd-free |
| io | PLY binary write | 50352.5 | k pts/s | 7.6 MB |
| io | PLY binary read | 36927.3 | k pts/s | 7.6 MB |
| render | point render 960x640x2 | 457.9 | k pts/s | 46,124 pts, 1.9 px |
| render | mesh render 960x640x2 | 38.9 | k faces/s | 238,852 faces, density 1.5 |
| recon | tsdf_fuse (splatting) | 13.9 | M vox/s | 5,994,120 voxels, 296,583 observed |
| recon | reconstruct end-to-end | 147.4 | k faces/s | 475,876 faces in 3229 ms |
| lidar | simulate 64-beam scan | 390.3 | k rays/s | 103,509 returns |
| lidar | range image | 20126.2 | k pts/s | 103,509 pts |
| lidar | analyze_scan | 1007.2 | k pts/s | ground + clusters + boxes |

Peak resident memory of the whole run: **618 MB**.