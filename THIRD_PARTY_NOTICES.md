# 第三方组件与来源

ModelLab 为独立实现，没有复制 OptiMUS 或 ORLM 源码、权重或研究数据集。
本地 `参考项目/OptiMUS-main/README.md` 仅用于理解固定阶段与结构化状态的参考方向，未将该下载目录当作已核实 commit 的代码依赖。
因此当前没有需要记录的移植文件或上游 commit；以后若移植，应补充 commit、许可、文件与修改记录。

主要依赖：React / Vite / KaTeX / FastAPI / Pydantic / HiGHS / NumPy / Pillow / uvicorn / httpx / argon2-cffi / Playwright。
实际发行包的 LICENSE/NOTICE 以各锁定依赖自带内容为准，Python 依赖锁为 uv.lock，前端依赖锁为 frontend/package-lock.json。
KaTeX 字体在构建产物中随包本地分发，保留 node_modules/katex/LICENSE 及字体许可。

- HiGHS 官方：https://highs.dev/
- HiGHS Python API：https://ergo-code.github.io/HiGHS/dev/interfaces/python/example-py/
- OptiMUS：https://github.com/teshnizi/OptiMUS
- Chat Completions 图像协议参考：https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create

图像适配器使用可配置 HTTP 协议；没有默认付费服务、默认模型或隐式 API 调用。
