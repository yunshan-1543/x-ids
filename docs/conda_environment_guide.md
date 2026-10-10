# Conda 环境整理指南

## 当前盘点结论

- Conda 安装位置: `D:\miniconda3`
- 当前项目建议环境: `ids`
- `ids` Python: `3.10.20`
- `ids` 关键依赖检查: `pip check` 未发现 broken requirements
- 当前环境列表: `base`, `claude_work`, `codex_work`, `ids`, `testenv`

## 主要风险点

1. `conda` 不在当前 PowerShell PATH 中,直接执行 `conda ...` 会失败。建议使用完整路径:

   ```powershell
   D:\miniconda3\condabin\conda.bat ...
   ```

2. 当前 `auto_activate_base=True`,新终端容易自动进入 `base`,之后误把项目依赖装到 `base`。

3. 当前 `channel_priority=flexible`,并且清华镜像与 `defaults` 同时存在。长期混装时,同一依赖可能来自不同 channel,增加求解和二进制兼容风险。

4. `ids` 环境导出时出现过 `conda-meta\setuptools-83.0.0-...json` 文件句柄占用提示。通常是仍有 Python、Jupyter、uvicorn 或编辑器进程在使用该环境。整理前先关闭相关进程,必要时重启电脑。

## 推荐整理策略

- `base`: 只保留 conda 自身和通用管理工具,不要安装项目依赖。
- `ids`: 本项目唯一运行环境,用于数据清洗、训练、FastAPI 服务、Scapy 抓包。
- `codex_work` / `claude_work`: 保留给智能体或其他工具链,不要混用本项目。
- `testenv`: 如果确认不用,再由用户本人删除。

## 用户本人执行的整理命令

先关闭正在运行的 `uvicorn`、Jupyter、Python 训练脚本和占用 `ids` 的终端。

关闭 base 自动激活:

```powershell
D:\miniconda3\condabin\conda.bat config --set auto_activate_base false
```

把 channel priority 改为 strict:

```powershell
D:\miniconda3\condabin\conda.bat config --set channel_priority strict
```

查看当前 channel:

```powershell
D:\miniconda3\condabin\conda.bat config --show channels
```

如果决定统一使用清华镜像,建议移除裸 `defaults`,避免同源重复:

```powershell
D:\miniconda3\condabin\conda.bat config --remove channels defaults
```

清理 conda 包缓存:

```powershell
D:\miniconda3\condabin\conda.bat clean --all
```

检查项目环境:

```powershell
D:\miniconda3\condabin\conda.bat activate ids
python --version
python -m pip check
python -c "import fastapi, uvicorn, xgboost, numpy, pandas, shap, scapy, sklearn, torch; print('ids env ok')"
```

如果确认 `testenv` 不再使用,再删除:

```powershell
D:\miniconda3\condabin\conda.bat env remove -n testenv
```

## 项目运行固定命令

每次运行本项目都先激活 `ids`:

```powershell
D:\miniconda3\condabin\conda.bat activate ids
cd E:\project1\detector
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
```

实时抓包还需要 Windows 系统安装 Npcap,并以管理员权限启动 PowerShell。
