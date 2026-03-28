# AI Code Complete

`ai-code-complete` 是一个面向当前工作区的轻量自动补全扩展。  
它会扫描仓库内代码文件，提取标识符并在输入时提供补全候选。
采用前后端分离架构：  
- 前端：VS Code 插件（触发建议、展示候选、Tab 接受补全）  
- 后端：Flask + `CodeRAG-main`（dataflow 检索上下文 + 生成补全代码）

## 功能
- 基于仓库代码词汇进行补全建议（不是固定词典）
- 自动索引当前工作区文件（前端本地索引 + 后端 CodeRAG 图索引）
- 基于 `CodeRAG-main` dataflow 检索工作区相关代码片段
- `Ctrl+Space` 手动触发时展示检索增强补全与 LM 推理补全代码
- 支持通过 `Tab` 直接插入 LM 生成补全（inline suggestion）
- 支持手动命令重建索引：`AI Code Complete: Rebuild Workspace Index`
- 文件保存后自动增量更新索引

## 使用方式

1. 启动 Flask 后端（在 `CodeRAG-main` 目录）：
   - `pip install flask`
   - `python flask_server.py --host 127.0.0.1 --port 5050`
2. 在 VS Code 中启动扩展开发主机（`F5`）。
3. 打开任意项目文件并输入至少 2 个字符。
4. 使用 `Ctrl+Space` 触发建议，插件调用后端获取检索上下文与补全代码。
5. 看到 LM ghost text 后按 `Tab` 直接接受补全。
6. 若新增大量文件，执行命令 `AI Code Complete: Rebuild Workspace Index` 同步前后端索引。

## 配置项

- `aiCodeComplete.include`：索引包含文件的 glob
- `aiCodeComplete.exclude`：索引排除文件的 glob
- `aiCodeComplete.maxFiles`：最多索引文件数
- `aiCodeComplete.maxSuggestions`：单次返回补全数量
- `aiCodeComplete.maxTokenLength`：可索引的最大 token 长度
- `aiCodeComplete.maxFileSizeKB`：忽略超过该体积的文件
- `aiCodeComplete.dataflowTopK`：dataflow 检索返回的 top-k 片段数
- `aiCodeComplete.dataflowContextLines`：检索 query 使用的光标前上下文行数
- `aiCodeComplete.enableLmCompletion`：是否启用 LM 推理补全
- `aiCodeComplete.lmApiKey`：LM API Key（为空时回退 `OPENAI_API_KEY`）
- `aiCodeComplete.lmBaseUrl`：LM 接口地址
- `aiCodeComplete.lmModel`：LM 模型名
- `aiCodeComplete.lmMaxTokens`：LM 最大输出 token
- `aiCodeComplete.backendEnabled`：是否启用 Flask 后端
- `aiCodeComplete.backendSuggestUrl`：后端 `/suggest` 地址
- `aiCodeComplete.backendIndexUrl`：后端 `/index` 地址
- `aiCodeComplete.backendRequestTimeoutMs`：后端请求超时
- `aiCodeComplete.enableLogging`：是否输出调试日志到 Output 面板

## 日志查看

1. 打开 `View -> Output`
2. 在右上角通道下拉中选择 `AI Code Complete`
3. 你可以看到：
   - 补全触发方式（自动触发 / `Ctrl+Space` 手动触发）
   - 当前文件、前缀和返回候选数量
   - 索引重建、增量更新、跳过大文件等信息

## 已知限制

- 后端服务未启动时，插件会回退到前端本地候选
- `CodeRAG-main` 的 dataflow 主体针对 Python 最优，其他语言效果取决于上下文质量
- 超大仓库建议调小 `maxFiles` 或收紧 `include` 规则
