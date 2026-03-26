# AI Code Complete

`ai-code-complete` 是一个面向当前工作区的轻量自动补全扩展。  
它会扫描仓库内代码文件，提取标识符并在输入时提供补全候选。

## 功能

- 自动索引当前工作区文件（可配置包含/排除规则）
- 基于仓库代码词汇进行补全建议（不是固定词典）
- 支持手动命令重建索引：`AI Code Complete: Rebuild Workspace Index`
- 文件保存后自动增量更新索引

## 使用方式

1. 在 VS Code 中启动扩展开发主机（`F5`）。
2. 打开任意项目文件并输入至少 2 个字符。
3. 使用 `Ctrl+Space` 触发建议，查看来自工作区的补全项。
4. 如果新增大量文件，执行命令面板中的 `AI Code Complete: Rebuild Workspace Index`。

## 配置项

- `aiCodeComplete.include`：索引包含文件的 glob
- `aiCodeComplete.exclude`：索引排除文件的 glob
- `aiCodeComplete.maxFiles`：最多索引文件数
- `aiCodeComplete.maxSuggestions`：单次返回补全数量
- `aiCodeComplete.maxTokenLength`：可索引的最大 token 长度
- `aiCodeComplete.maxFileSizeKB`：忽略超过该体积的文件
- `aiCodeComplete.enableLogging`：是否输出调试日志到 Output 面板

## 日志查看

1. 打开 `View -> Output`
2. 在右上角通道下拉中选择 `AI Code Complete`
3. 你可以看到：
   - 补全触发方式（自动触发 / `Ctrl+Space` 手动触发）
   - 当前文件、前缀和返回候选数量
   - 索引重建、增量更新、跳过大文件等信息

## 已知限制

- 当前为本地词汇索引补全，不调用远程大模型
- 未做语义理解，主要适合变量/函数名等标识符补全
- 超大仓库建议调小 `maxFiles` 或收紧 `include` 规则
