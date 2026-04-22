# AI Code Complete

一个面向 VS Code 的代码补全原型仓库，当前由两个部分组成：

- `ai-code-complete/`：VS Code 扩展，负责索引工作区、触发补全、展示上下文与应用建议
- `CodeRAG-main/`：Python/Flask 后端，负责代码检索、上下文融合和本地模型推理



## 仓库结构

```text
.
├─ README.md
├─ ai-code-complete/       # VS Code 扩展
└─ CodeRAG-main/           # CodeRAG 检索与推理后端
```

## 功能概览

### VS Code 扩展

扩展位于 `ai-code-complete/`，已实现：

- 工作区文件索引与增量更新
- 基于标识符频率的本地补全候选
- 基于简单 BM25 和 dataflow 的混合检索上下文提取
- 调用 Flask 后端进行检索增强补全
- 在编辑器中预览补全文本
- 支持查看检索上下文、应用补全、取消补全

默认命令和快捷键：

- `AI Code Complete: Rebuild Workspace Index`
  - `Ctrl+Shift+R`
- `AI Code Complete: Trigger Completion`
  - `Ctrl+Alt+T`
- `Tab`
  - 在存在建议时应用补全

### CodeRAG 后端

后端位于 `CodeRAG-main/`，已实现：

- 解析当前工作区并构建缓存到 `.ai-code-complete/dataflow-cache/`
- 通过 `DataflowRetriever` 做 dataflow 检索
- 通过 BM25 做稀疏检索
- 融合并重排两路检索结果
- 组装 prompt 后调用本地 Hugging Face 模型推理
- 通过 Flask 暴露 `/health`、`/index`、`/suggest` 接口

当前默认加载模型：

- `Salesforce/codegen-350M-mono`

## 运行要求

### 扩展侧

- Node.js
- npm
- VS Code 1.105.0 及以上

### 后端侧

- Python 3.10+
- 推荐使用 `uv`
- 建议具备可用的 Python 深度学习环境
- 首次运行模型时需要下载 Hugging Face 模型与依赖

## 快速开始

### 1. 启动 Python 后端

进入后端目录并安装依赖：

```bash
cd CodeRAG-main
uv sync
```
激活虚拟环境：

```bash
.venv\Scripts\Activate.ps1
```   

启动服务：


```bash
python flask_server.py --host 127.0.0.1 --port 5050
```

服务默认地址：

- `http://127.0.0.1:5050`

### 2. 启动 VS Code 扩展

进入扩展目录并安装依赖：

```bash
cd ai-code-complete
npm install
```

然后用 VS Code 打开 `ai-code-complete/`，按 `F5` 启动 Extension Development Host。

### 3. 在测试工作区中验证

1. 打开任意代码仓库作为工作区
2. 确保后端服务已启动
3. 在扩展设置中确认后端地址为默认值：

```text
aiCodeComplete.backendSuggestUrl = http://127.0.0.1:5050/suggest
aiCodeComplete.backendIndexUrl   = http://127.0.0.1:5050/index
```

4. 执行 `AI Code Complete: Rebuild Workspace Index`
5. 在代码文件中使用 `Ctrl+Alt+T` 触发补全

## 扩展配置

扩展支持以下主要配置项：

- `aiCodeComplete.include`
- `aiCodeComplete.exclude`
- `aiCodeComplete.maxFiles`
- `aiCodeComplete.maxSuggestions`
- `aiCodeComplete.maxTokenLength`
- `aiCodeComplete.maxFileSizeKB`
- `aiCodeComplete.dataflowTopK`
- `aiCodeComplete.dataflowContextLines`
- `aiCodeComplete.enableLmCompletion`
- `aiCodeComplete.lmApiKey`
- `aiCodeComplete.lmBaseUrl`
- `aiCodeComplete.lmModel`
- `aiCodeComplete.lmMaxTokens`
- `aiCodeComplete.backendEnabled`
- `aiCodeComplete.backendSuggestUrl`
- `aiCodeComplete.backendIndexUrl`
- `aiCodeComplete.backendRequestTimeoutMs`
- `aiCodeComplete.enableLogging`

说明：

- 当前后端默认走本地 Flask 服务
- 扩展配置里保留了 `lmApiKey`、`lmBaseUrl`、`lmModel` 等字段，但当前后端实现主要使用本地模型推理
- `OPENAI_API_KEY` 可作为扩展侧备用环境变量来源，但是否实际生效取决于你如何改造后端

## 日志查看

1. 打开 `View -> Output`
2. 在右上角通道下拉中选择 `AI Code Complete`

## 后端接口

### `GET /health`

用于健康检查。

### `POST /index`

请求体示例：

```json
{
  "workspace_path": "D:/path/to/your/project",
  "force_rebuild": true
}
```

用途：

- 为目标工作区建立或刷新检索缓存

### `POST /suggest`

请求体示例：

```json
{
  "workspace_path": "D:/path/to/your/project",
  "file_path": "D:/path/to/your/project/src/app.py",
  "source_code": "光标前的源码内容",
  "enable_completion": true,
  "max_tokens": 192
}
```

返回内容包含：

- `retrieved_contexts`
- `completion`
- `retrieved_contexts_count`
- `completion_chars`

## 开发说明

### 扩展目录

- 入口文件：`ai-code-complete/extension.js`
- 测试目录：`ai-code-complete/test/`

常用命令：

```bash
cd ai-code-complete
npm run lint
npm test
```

### 后端目录

- 入口文件：`CodeRAG-main/flask_server.py`
- 配置文件：`CodeRAG-main/config/config.toml`
- 脚本目录：`CodeRAG-main/scripts/`

在当前仓库集成方式下，VS Code 扩展通常直接通过 Flask API 调用后端。

## 当前限制

- 根目录是聚合仓库，不是单一 package
- 扩展目前是原型实现，核心逻辑集中在单个 `extension.js`
- 后端默认模型较小，补全质量和速度都仍有优化空间
- 首次建立索引和首次加载模型会比较慢
- Windows 环境下部分深度学习依赖可能需要额外处理
- 仓库内已有一些注释和日志文本存在编码问题，不影响 README，但后续建议统一编码

## 后续可补充的方向

- 将扩展代码拆分为多个模块
- 为后端补充更明确的依赖说明和启动脚本
- 增加真实示例截图或演示 GIF
- 补充 API 协议、缓存结构和检索流程说明
- 增加一键联调脚本



