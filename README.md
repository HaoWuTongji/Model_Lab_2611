# ModelLab · 个人建模实验室

面向个人学习与科研的本地数学规划工作台。打开即可使用，无账号、密码、教师/学生角色或课程管理。

模型输入 → 修订确认 → 确定性编译 → HiGHS 求解 → 独立残差校验 → 版本与代码导出。

## 启动

本机依赖与前端产物已准备，直接在本目录执行：

```bash
make start
```

打开 http://127.0.0.1:8000 ，或双击 `启动 ModelLab.command`。不需要初始化账号。

从旧版升级时停止旧服务，再重新启动一次。旧练习、图片、确认和历史版本保留，在同一个个人工作台中可继续编辑。旧课程/账号数据库记录仅保留作为历史关联，不再提供登录或管理入口。

## 界面连接大模型

左侧进入 **模型设置**：

1. 选择“兼容 Chat Completions 的云端 API”或“本机模型服务”。
2. 填写 API 地址（供应商 Base URL，通常以 `/v1` 结尾）、模型名称和 API Key。
3. 勾选“该模型支持图像输入”。供应商支持 JSON 模式时可另外启用。
4. 点击“保存连接”，即时生效，无需编辑 `.env` 或重启 API/worker。
5. 可点击“测试连接”发送一条简短文本请求，验证地址、模型和凭据；图像能力仍需实际识别验证。

密钥输入框不回显已保存值，留空保留现有密钥；输入新值替换。更换 API 地址必须重新输入对应密钥，避免把原密钥发往另一服务。点击“清除连接”删除当前连接和密钥。本机回环地址的服务可不填写密钥。

连接保存在 `.runtime/model-service.json`（或 MODELLAB_DATA_DIR 指定目录），文件权限为 0600。密钥不写入浏览器存储，不返回到读取设置的 API，不进入任务参数、日志或模型导出包。它是本机配置文件，并非加密密码库；备份数据目录时一并妥善保管。

旧 `.env` 模型配置仅在尚未通过界面保存配置时作为兼容回退。界面保存或清除后以界面配置为准。识别任务排队后若连接发生变化，任务会提示重新提交，不会静默改用另一个服务。

## 使用模型与案例

- **建模工作区**：新建个人练习，手动输入公式、变量范围和数据，也可以上传手写图片。
- **参考案例**：内置整数生产计划、连续松弛、3×3 指派、自由变量、不可行模型、无界模型。可查看数学表达与数据，点击“以此开始建模”创建独立副本。
- **模型设置**：配置视觉大模型服务。未接入模型时仍可手动编辑和确认。

上传图片后先保存版本，再点击“识别手写模型”。识别结果必须预览、核对疑点并采纳为新版本，不会覆盖已有编辑。

公式支持 `3*x + 5*y`、`sum(i,I,x[i])`、`sum(i,I,sum(j,J,cost[i,j]*x[i,j]))`。解析预览后点击“应用此表达式”。JSON 修改后点击“应用 JSON”，再保存新版本。空白变量界限表示未确认；显式 null /“无界”才表示自由方向。

确认覆盖目标、全部约束、量词、变量类型、上下界和实例数据。修改后需要重新确认。结果只证明当前数学模型的可行性/最优性，不证明它符合业务题意。

## 求解环境

默认使用**本机求解**，无需 Docker，也无需大模型 API。确认模型后直接点击“运行已确认模型”，由独立 Python 子进程调用 HiGHS；页面会显示“本机求解已就绪”。

保留默认 30 秒求解限时、额外 10 秒硬超时、任务取消、进程清理与独立结果校验。每任务使用独立工作目录和最小环境，不向求解进程传递模型 API Key。worker 退出时，通过父进程生命期管道回收子进程；重新启动后悬挂任务标记为 interrupted。

本机模式执行固定模板，不接受用户 Python。它是个人本地工具的进程分离，不具备容器的文件系统、网络或内存隔离；本机模式限制求解线程数、CPU 时间及输出文件大小，未强制 1 GiB 内存上限。

如需要原有 Docker 隔离模式，可在 `.env` 设置 `SOLVER_MODE=docker`，安装并启动 Docker 后运行 `make runner`，再重启平台。默认 `SOLVER_MODE=local`，两种模式都不会在 API 请求进程内直接求解。

本地可信 CLI 可直接运行真实 HiGHS，不需要大模型或 Docker：

```bash
make cli
.venv/bin/python -m modellab.cli fixtures/production_lp.json --solve --out .runtime/lp
.venv/bin/python -m modellab.cli fixtures/assignment.json --data fixtures/assignment_data.json --solve --out .runtime/assignment
```

## 新机器安装与开发

需要 Python 3.12（推荐 uv 管理）、Node 22；默认本机求解无需 Docker。已实测 Apple Silicon；目标 Linux / Mac Intel 应重新验收。

```bash
make setup-app     # 锁定依赖安装 + 前端构建
make runner        # 可选：仅 Docker 模式需要
make start         # API + worker + 构建后的前端
```

开发模式三个终端：`make dev`、`make worker`、`make frontend`。Vite 在 127.0.0.1:5173，代理 API。API 与 worker 使用相同项目目录和配置。

后端锁为 `uv.lock`，前端锁为 `frontend/package-lock.json`，runner 依赖为 `runner/requirements.txt`。当前仅支持单机单人使用；所有可访问此本地服务的人使用同一工作台。默认绑定 127.0.0.1，不作为多人公开服务部署。

## 数据、复现与测试

数据默认在 `.runtime/`。停止服务后备份整个数据目录即可保留模型、图片、历史与连接设置。`MODELLAB_DATA_DIR` 可指定其他目录。

确认模型后可下载 ZIP：原图、题意、模型、数据、IR、`solve.py`、来源映射、审计和锁定求解依赖。有成功运行时还包含结果与配置。

在解压目录使用 Python 3.12：

```bash
pip install -r requirements.txt
python solve.py
```

预先安装依赖后可离线复现。

```bash
make test
make build
make test-browser
make schema
```

模拟识别仅用于开发：设置 `ALLOW_MOCK=1` 后重新启动。模拟 provider 始终返回固定生产模型，不读取图片，不代表真实识别效果。真实 LLM 测试仅在显式配置环境凭据时运行，可能产生 API 用量。

[本机求解验证](docs/LOCAL_SOLVER.md) · [个人版变更与验证](docs/PERSONAL_MODE.md) · [原 M1–M3 验证记录](docs/TEST_REPORT.md) · [实施决策](docs/DECISIONS.md)
