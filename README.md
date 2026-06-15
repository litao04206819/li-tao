# 锂离子电池建模（PyBaMM）— 从放电仿真到参数辨识

> 基于 [PyBaMM](https://pybamm.org) 的锂离子电池物理建模教程，由浅入深的 6 个 Jupyter Notebook，
> 覆盖 SPM / SPMe / DFN 三类模型、放电倍率、动态工况、热与老化、参数辨识。

本项目使用 PyBaMM（Python Battery Mathematical Modelling）构建锂离子电池模型，
以 Jupyter Notebook 形式由浅入深演示从基础放电仿真到参数辨识的完整链路。

> 说明：PyBaMM 为纯 Python 物理建模库，无 MATLAB 等价实现，因此本项目仅提供 Python / Notebook 版本。

## 安装

```bash
pip install -r requirements.txt
# 验证
python -c "import pybamm; print(pybamm.__version__)"
```

PyBaMM 默认自带 CasADi / IDAKLU 求解器，无需额外编译；首次运行会自动加载内置参数集。

## Notebook 列表

| Notebook | 主题 | 主要内容 |
| --- | --- | --- |
| `nb01_basics.ipynb` | 入门与三模型对比 | SPM / SPMe / DFN 单次放电；电压曲线与求解耗时对比；三者物理差异 |
| `nb02_parameters_crate.ipynb` | 参数集与放电倍率 | 内置参数集；修改关键参数；扫描多 C-rate；截止电压条件 |
| `nb03_experiments_drivecycle.ipynb` | 工况与动态负载 | CCCV 多步循环；CSV drive cycle 插值电流；恒/变功率负载 |
| `nb04_thermal.ipynb` | 热模型 | isothermal vs lumped 温升；环境温度/散热影响；产热来源 |
| `nb05_degradation_aging.ipynb` | 老化与容量衰减 | SEI / 析锂 / 颗粒开裂子模型；多循环容量保持率；老化指标 |
| `nb06_parameter_identification.ipynb` | 参数辨识 | 合成数据 → 扰动初值 → scipy/PyBOP 优化 → 拟合对比 |

## 运行方式

逐个交互运行：

```bash
jupyter lab          # 或 jupyter notebook
```

命令行端到端执行（验证可无错跑通）：

```bash
jupyter nbconvert --to notebook --execute nb01_basics.ipynb
```

## 模型说明

- **SPM**（Single Particle Model，单颗粒模型）：每个电极一个代表性颗粒，最快、精度最低，适合系统级/控制仿真。
- **SPMe**（SPM with electrolyte，含电解液单颗粒模型）：在 SPM 基础上加入电解液动力学，精度与速度折中。
- **DFN**（Doyle-Fuller-Newman，又称 Newman P2D）：完整电化学模型，精度最高、计算量最大。

默认参数集采用 `Chen2020`（LG M50，NMC811 / 石墨-硅，文献常用、数据完整）。可在 notebook 中替换为
`Marquis2019`、`OKane2022`、`Ai2020`（含 LFP/其他化学体系）等。

## 数据

`data/` 目录存放小体积示例数据（drive cycle、合成实验数据 CSV），随仓库提交以保证 notebook 离线可复现；
部分数据由 notebook 运行时生成。
