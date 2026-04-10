# 螺旋桨参数化建模方法 — Python & MATLAB 代码包

## 目录结构

```
parametric_codes/
├── python/                         # Python 实现 (含可视化)
│   ├── p01_cst.py                  # CST 方法
│   ├── p02_parsec.py               # PARSEC 方法
│   ├── p03_hicks_henne.py          # Hicks-Henne 凸包函数
│   ├── p04_bspline.py              # B样条/NURBS
│   ├── p05_ffd.py                  # 自由形变 (FFD)
│   ├── p06_spanwise_betz_adkins.py # 展向分布 + Betz-Prandtl + Adkins-Liebeck
│   ├── cst_demo.png                # CST 可视化结果
│   ├── parsec_demo.png             # PARSEC 可视化结果
│   ├── hicks_henne_demo.png        # Hicks-Henne 可视化结果
│   ├── bspline_demo.png            # B样条 可视化结果
│   ├── ffd_demo.png                # FFD 可视化结果
│   └── spanwise_betz_adkins_demo.png # 展向/Betz/Adkins 可视化结果
│
└── matlab/                         # MATLAB 实现
    ├── m01_cst.m                   # CST 方法
    ├── m02_parsec.m                # PARSEC 方法
    ├── m03_hicks_henne.m           # Hicks-Henne 凸包函数
    ├── m04_bspline.m               # B样条/NURBS
    ├── m05_ffd.m                   # 自由形变 (FFD)
    └── m06_spanwise_betz_adkins.m  # 展向分布 + Betz-Prandtl + Adkins-Liebeck
```

## 方法对应表

| 编号 | 方法 | Python 文件 | MATLAB 文件 | 参考文献 |
|------|------|------------|-------------|---------|
| 1 | CST | p01_cst.py | m01_cst.m | Kulfan, J. Aircraft, 2008 |
| 2 | PARSEC | p02_parsec.py | m02_parsec.m | Sobieczky, 1999 |
| 3 | Hicks-Henne | p03_hicks_henne.py | m03_hicks_henne.m | Hicks & Henne, J. Aircraft, 1978 |
| 4 | B-Spline | p04_bspline.py | m04_bspline.m | Piegl & Tiller, 1997 |
| 5 | FFD | p05_ffd.py | m05_ffd.m | Sederberg & Parry, 1986 |
| 6 | 展向分布 + Betz + Adkins | p06_*.py | m06_*.m | 多篇经典文献 |

## 运行方法

### Python
```bash
pip install numpy matplotlib scipy
python p01_cst.py          # 生成 cst_demo.png
python p02_parsec.py       # 生成 parsec_demo.png
# ... 依此类推
```

### MATLAB
直接在 MATLAB 中打开并运行每个 .m 文件即可，所有函数都定义在脚本内部（local functions），无需额外依赖。

## 每个文件的内容

每个代码文件都包含：
1. **文件头注释**: 方法来源、核心思想、参考文献
2. **核心函数**: 可独立调用的参数化函数
3. **演示代码**: 4~6个子图，展示不同使用场景
4. **中文注释**: 关键步骤均有中文说明
