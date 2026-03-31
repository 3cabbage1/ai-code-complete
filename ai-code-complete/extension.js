const vscode = require('vscode');

const DEFAULT_MAX_FILES = 300;
const DEFAULT_MAX_SUGGESTIONS = 40;
const DEFAULT_MAX_TOKEN_LENGTH = 48;
const DEFAULT_MAX_FILE_SIZE_KB = 256;
const DEFAULT_DATAFLOW_CONTEXT_LINES = 120;
const DEFAULT_DATAFLOW_TOP_K = 8;

function createLogger() {
	const output = vscode.window.createOutputChannel('AI Code Complete');
	return {
		output,
		info(message) {
			if (!getConfig().enableLogging) {
				return;
			}
			output.appendLine(`[${new Date().toISOString()}] ${message}`);
		}
	};
}

class WorkspaceDataflowIndex {
	constructor() {
		this.fileData = new Map();
		this.tokenFrequency = new Map();
		this.symbolToSnippets = new Map();
		this.snippetStore = new Map();
		this.nextSnippetId = 1;
		this.isBuilding = false;
	}

	clear() {
		this.fileData.clear();
		this.tokenFrequency.clear();
		this.symbolToSnippets.clear();
		this.snippetStore.clear();
		this.nextSnippetId = 1;
	}

	getTokenCandidates(prefix, limit) {
		if (!prefix) {
			return [];
		}

		const lowerPrefix = prefix.toLowerCase();
		const matches = [];

		for (const [token, score] of this.tokenFrequency.entries()) {
			if (!token.toLowerCase().startsWith(lowerPrefix)) {
				continue;
			}
			matches.push({ token, score });
		}

		matches.sort((a, b) => {
			if (b.score !== a.score) {
				return b.score - a.score;
			}
			return a.token.localeCompare(b.token);
		});

		return matches.slice(0, limit).map((item) => item.token);
	}

	getDataflowContext(document, position, topK, contextLines) {
		const source = document.getText();
		const lines = source.split(/\r?\n/);
		const cursorLine = position.line;
		const begin = Math.max(0, cursorLine - contextLines);
		const contextText = lines.slice(begin, cursorLine + 1).join('\n');
		const symbols = extractIdentifiers(contextText, 2);
		const symbolSet = new Set(symbols);

		const scored = [];
		for (const snippet of this.snippetStore.values()) {
			let score = 0;
			for (const sym of snippet.defines) {
				if (symbolSet.has(sym)) {
					score += 5;
				}
			}
			for (const sym of snippet.uses) {
				if (symbolSet.has(sym)) {
					score += 2;
				}
			}
			for (const imp of snippet.imports) {
				if (contextText.includes(imp)) {
					score += 3;
				}
			}

			if (snippet.filePath === document.uri.fsPath) {
				score += 1;
			}

			if (score > 0) {
				scored.push({ snippet, score });
			}
		}

		scored.sort((a, b) => {
			if (b.score !== a.score) {
				return b.score - a.score;
			}
			if (a.snippet.filePath !== b.snippet.filePath) {
				return a.snippet.filePath.localeCompare(b.snippet.filePath);
			}
			return a.snippet.line - b.snippet.line;
		});

		return scored.slice(0, topK).map((item) => item.snippet);
	}

	async rebuild(logger) {
		if (this.isBuilding) {
			logger?.info('Index rebuild skipped because another rebuild is running.');
			return;
		}

		this.isBuilding = true;
		try {
			this.clear();
			const config = getConfig();
			logger?.info(`Index rebuild started. include=${config.includeGlob} exclude=${config.excludeGlob} maxFiles=${config.maxFiles}`);
			const files = await vscode.workspace.findFiles(
				config.includeGlob,
				config.excludeGlob,
				config.maxFiles
			);
			logger?.info(`Found ${files.length} files for indexing.`);

			for (const uri of files) {
				await this.indexUri(uri, logger);
			}
			logger?.info(`Index rebuild completed. tokens=${this.tokenFrequency.size}`);
		} finally {
			this.isBuilding = false;
		}
	}

	async indexUri(uri, logger) {
		try {
			const config = getConfig();
			this.removeFile(uri);

			const stat = await vscode.workspace.fs.stat(uri);
			if (stat.size > config.maxFileSizeKB * 1024) {
				logger?.info(`Skip large file: ${uri.fsPath}`);
				return;
			}

			const data = await vscode.workspace.fs.readFile(uri);
			const content = Buffer.from(data).toString('utf8');
			const tokens = extractTokens(content, config.maxTokenLength);
			for (const token of tokens) {
				this.tokenFrequency.set(token, (this.tokenFrequency.get(token) || 0) + 1);
			}

			const snippets = buildFileSnippets(uri.fsPath, content);
			const snippetIds = [];
			for (const snippet of snippets) {
				const snippetId = this.nextSnippetId++;
				snippetIds.push(snippetId);
				this.snippetStore.set(snippetId, snippet);

				for (const symbol of snippet.defines) {
					if (!this.symbolToSnippets.has(symbol)) {
						this.symbolToSnippets.set(symbol, new Set());
					}
					this.symbolToSnippets.get(symbol).add(snippetId);
				}
			}

			this.fileData.set(uri.toString(), {
				tokens,
				snippetIds
			});
		} catch (error) {
			logger?.info(`Index failed for ${uri.fsPath}: ${error?.message || 'unknown error'}`);
		}
	}

	removeFile(uri) {
		const key = uri.toString();
		const existing = this.fileData.get(key);
		if (!existing) {
			return;
		}

		for (const token of existing.tokens) {
			const next = (this.tokenFrequency.get(token) || 0) - 1;
			if (next <= 0) {
				this.tokenFrequency.delete(token);
			} else {
				this.tokenFrequency.set(token, next);
			}
		}

		for (const snippetId of existing.snippetIds) {
			const snippet = this.snippetStore.get(snippetId);
			if (snippet) {
				for (const symbol of snippet.defines) {
					const refs = this.symbolToSnippets.get(symbol);
					if (refs) {
						refs.delete(snippetId);
						if (refs.size === 0) {
							this.symbolToSnippets.delete(symbol);
						}
					}
				}
			}
			this.snippetStore.delete(snippetId);
		}

		this.fileData.delete(key);
	}
}

function getConfig() {
	const config = vscode.workspace.getConfiguration('aiCodeComplete');
	return {
		includeGlob: config.get('include', '**/*.{js,jsx,ts,tsx,py,go,java,cpp,c,h,cs,php,rb,rs,swift,kt,m,mm,scala,sql,md,json,yaml,yml}'),
		excludeGlob: config.get('exclude', '**/{node_modules,.git,dist,build,out,.next,.venv,venv,target,coverage}/**'),
		maxFiles: config.get('maxFiles', DEFAULT_MAX_FILES),
		maxSuggestions: config.get('maxSuggestions', DEFAULT_MAX_SUGGESTIONS),
		maxTokenLength: config.get('maxTokenLength', DEFAULT_MAX_TOKEN_LENGTH),
		maxFileSizeKB: config.get('maxFileSizeKB', DEFAULT_MAX_FILE_SIZE_KB),
		enableLogging: config.get('enableLogging', true),
		dataflowTopK: config.get('dataflowTopK', DEFAULT_DATAFLOW_TOP_K),
		dataflowContextLines: config.get('dataflowContextLines', DEFAULT_DATAFLOW_CONTEXT_LINES),
		enableLmCompletion: config.get('enableLmCompletion', true),
		lmApiKey: config.get('lmApiKey', ''),
		lmBaseUrl: config.get('lmBaseUrl', 'https://api.openai.com/v1/chat/completions'),
		lmModel: config.get('lmModel', 'gpt-4o-mini'),
		lmMaxTokens: config.get('lmMaxTokens', 192),
		backendEnabled: config.get('backendEnabled', true),
		backendSuggestUrl: config.get('backendSuggestUrl', 'http://127.0.0.1:5050/suggest'),
		backendIndexUrl: config.get('backendIndexUrl', 'http://127.0.0.1:5050/index'),
		backendRequestTimeoutMs: config.get('backendRequestTimeoutMs', 20000)
	};
}

function extractTokens(content, maxTokenLength) {
	const results = new Set(extractIdentifiers(content, 2));
	for (const token of Array.from(results)) {
		if (token.length > maxTokenLength || isLikelyKeyword(token)) {
			results.delete(token);
		}
	}
	return results;
}

function extractIdentifiers(content, minLength = 3) {
	const results = [];
	const regex = /\b[A-Za-z_][A-Za-z0-9_]{1,}\b/g;
	let match = regex.exec(content);
	while (match) {
		const token = match[0];
		if (token.length >= minLength && !isLikelyKeyword(token)) {
			results.push(token);
		}
		match = regex.exec(content);
	}
	return results;
}

function isLikelyKeyword(token) {
	const keywords = new Set([
		'function', 'class', 'const', 'let', 'var', 'return', 'if', 'else', 'switch', 'case',
		'for', 'while', 'do', 'break', 'continue', 'import', 'export', 'from', 'default',
		'public', 'private', 'protected', 'static', 'async', 'await', 'true', 'false', 'null',
		'undefined', 'this', 'super', 'new', 'try', 'catch', 'finally', 'throw', 'extends',
		'implements', 'interface', 'enum', 'package', 'module', 'yield', 'with', 'and', 'or',
		'not', 'def', 'lambda', 'pass', 'None', 'self'
	]);
	return keywords.has(token);
}

function buildFileSnippets(filePath, content) {
	const lines = content.split(/\r?\n/);
	const snippets = [];

	for (let i = 0; i < lines.length; i += 1) {
		const line = lines[i];
		if (!line || line.trim().length < 2) {
			continue;
		}

		const defines = extractDefines(line);
		const uses = extractIdentifiers(line, 2);
		const imports = extractImports(line);
		if (defines.length === 0 && uses.length === 0 && imports.length === 0) {
			continue;
		}

		snippets.push({
			filePath,
			line: i + 1,
			text: line.trim(),
			defines,
			uses,
			imports
		});
	}

	return snippets;
}

function extractDefines(line) {
	const defs = [];
	const patterns = [
		/\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\b/g,
		/\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b/g,
		/\b(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\b/g,
		/\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\b/g
	];

	for (const pattern of patterns) {
		let match = pattern.exec(line);
		while (match) {
			defs.push(match[1]);
			match = pattern.exec(line);
		}
	}

	return defs;
}

function extractImports(line) {
	const imports = [];
	const patterns = [
		/\bimport\s+([A-Za-z0-9_./-]+)\b/g,
		/\bfrom\s+([A-Za-z0-9_./-]+)\s+import\b/g,
		/\brequire\(['"]([^'"]+)['"]\)/g
	];

	for (const pattern of patterns) {
		let match = pattern.exec(line);
		while (match) {
			imports.push(match[1]);
			match = pattern.exec(line);
		}
	}
	return imports;
}

function buildContextBlock(snippets) {
	return snippets.map((snippet) => `[${snippet.filePath}:${snippet.line}] ${snippet.text}`).join('\n');
}

async function postJsonWithTimeout(url, body, timeoutMs) {
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(), timeoutMs);
	try {
		const response = await fetch(url, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body),
			signal: controller.signal
		});
		const payload = await response.json().catch(() => ({}));
		return { ok: response.ok, payload, status: response.status };
	} finally {
		clearTimeout(timer);
	}
}

async function suggestViaBackend(document, position, config, logger) {
	if (!config.backendEnabled) {
		return { contexts: [], completion: '', error: 'backend disabled' };
	}

	const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);
	if (!workspaceFolder) {
		return { contexts: [], completion: '', error: 'no workspace folder' };
	}

	const beforeCursor = document.getText(new vscode.Range(new vscode.Position(0, 0), position));
	const apiKey = config.lmApiKey || process.env.OPENAI_API_KEY || '';

	try {
		const body = {
			workspace_path: workspaceFolder.uri.fsPath,
			file_path: document.uri.fsPath,
			source_code: beforeCursor,
			enable_completion: config.enableLmCompletion,
			api_key: apiKey,
			base_url: config.lmBaseUrl,
			model: config.lmModel,
			max_tokens: config.lmMaxTokens
		};
		
		logger.info('='.repeat(80));
		logger.info('QUERY SENT TO BACKEND (source_code before cursor):');
		logger.info('='.repeat(80));
		logger.info(beforeCursor);
		logger.info('='.repeat(80));
		
		const response = await postJsonWithTimeout(config.backendSuggestUrl, body, config.backendRequestTimeoutMs);
		if (!response.ok || !response.payload?.ok) {
			return {
				contexts: [],
				completion: '',
				error: response.payload?.error || `backend http ${response.status}`
			};
		}
		
		const contexts = response.payload.retrieved_contexts || [];
		const completion = response.payload.completion || '';
		
		logger.info('='.repeat(80));
		logger.info(`RETRIEVED CONTEXTS FROM BACKEND (count: ${contexts.length}):`);
		logger.info('='.repeat(80));
		contexts.forEach((ctx, idx) => {
			logger.info(`--- Context ${idx + 1} ---`);
			logger.info(ctx);
		});
		logger.info('='.repeat(80));
		
		logger.info('='.repeat(80));
		logger.info('COMPLETION CODE FROM BACKEND:');
		logger.info('='.repeat(80));
		logger.info(completion || '(empty)');
		logger.info('='.repeat(80));
		
		return {
			contexts,
			completion,
			error: ''
		};
	} catch (error) {
		return { contexts: [], completion: '', error: error?.message || 'backend request failed' };
	}
}

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
	const tokenIndex = new WorkspaceDataflowIndex();
	const logger = createLogger();
	const inlineCache = new Map();
	const inFlightBackend = new Map();

	logger.info('Extension activated.');

	const syncBackendIndex = async (forceRebuild = false) => {
		const config = getConfig();
		if (!config.backendEnabled) {
			logger.info('Backend index sync skipped: backend is disabled.');
			return;
		}
		const folders = vscode.workspace.workspaceFolders || [];
		const first = folders[0];
		if (!first) {
			logger.info('Backend index sync failed: no workspace folder found.');
			return;
		}
		
		logger.info('='.repeat(80));
		logger.info('STARTING BACKEND INDEX SYNC');
		logger.info(`Force rebuild: ${forceRebuild}`);

		logger.info('='.repeat(80));
		
		const response = await postJsonWithTimeout(
			config.backendIndexUrl,
			{
				workspace_path: first.uri.fsPath,
				force_rebuild: forceRebuild
			},
			config.backendRequestTimeoutMs
		);
		
		logger.info('='.repeat(80));
		logger.info('BACKEND INDEX SYNC RESPONSE');
		logger.info('='.repeat(80));
		logger.info(`Response OK: ${response.ok}`);
		
		if (!response.ok || !response.payload?.ok) {
			logger.info(`Backend index sync failed: ${response.payload?.error || `http ${response.status}`}`);
			return;
		}
		
		logger.info(`Backend index synced successfully`);
		logger.info(`Project name: ${response.payload.project_name}`);
		logger.info(`Cache directory: ${response.payload.cache_dir}`);
		
		// 打印后端返回的完整 JSON 响应（限制大小）
		const jsonResponse = JSON.stringify(response.payload, null, 2);
		const maxJsonLength = 5000; // 限制日志长度
		if (jsonResponse.length > maxJsonLength) {
			logger.info(`Backend response JSON (truncated): ${jsonResponse.substring(0, maxJsonLength)}...`);
			logger.info(`Total JSON size: ${jsonResponse.length} characters`);
		} else {
			logger.info(`Backend response JSON: ${jsonResponse}`);
		}
		
		logger.info('='.repeat(80));
	};

	const refreshIndexCommand = vscode.commands.registerCommand('ai-code-complete.rebuildIndex', async () => {
		logger.info('='.repeat(80));
		logger.info('MANUAL REBUILD COMMAND TRIGGERED');
		logger.info('='.repeat(80));
		logger.info(`Triggered by: Keyboard shortcut (Ctrl+Shift+R / Cmd+Shift+R)`);
		logger.info(`Timestamp: ${new Date().toISOString()}`);
		logger.info('='.repeat(80));
		
		const startTime = Date.now();
		
		await vscode.window.withProgress(
			{ location: vscode.ProgressLocation.Notification, title: 'AI Code Complete: rebuilding index...' },
			async () => {
				await tokenIndex.rebuild(logger);
				await syncBackendIndex(true);
			}
		);
		vscode.window.showInformationMessage('AI Code Complete: index rebuilt.');
	});

	const provider = vscode.languages.registerCompletionItemProvider(
		{ scheme: 'file' },
		{
			async provideCompletionItems(document, position, _token, completionContext) {
				const range = document.getWordRangeAtPosition(position, /[A-Za-z_][A-Za-z0-9_]*/);
				const prefix = range ? document.getText(range) : '';
				if (!prefix || prefix.length < 2) {
					return [];
				}

				const triggerLabel = completionContext.triggerKind === vscode.CompletionTriggerKind.Invoke
					? 'manual (Ctrl+Space or API invoke)'
					: completionContext.triggerKind === vscode.CompletionTriggerKind.TriggerCharacter
						? `triggerCharacter (${completionContext.triggerCharacter || '?'})`
						: 'triggerForIncompleteCompletions';

				const config = getConfig();
				const snippets = tokenIndex.getDataflowContext(document, position, config.dataflowTopK, config.dataflowContextLines);
				const candidates = tokenIndex.getTokenCandidates(prefix, config.maxSuggestions);
				logger.info(`Completion requested: mode=${triggerLabel} file=${document.fileName} prefix="${prefix}" resultCount=${candidates.length}`);

				const result = candidates.map((token, index) => {
					const item = new vscode.CompletionItem(token, vscode.CompletionItemKind.Text);
					item.sortText = String(index).padStart(4, '0');
					item.insertText = token;
					item.detail = 'From workspace index';
					return item;
				});

				// Ctrl+Space / 手动触发时，优先尝试生成 LM 补全并展示
					if (completionContext.triggerKind === vscode.CompletionTriggerKind.Invoke) {
						const requestKey = `${document.uri.toString()}::${position.line}:${position.character}`;

						// 后端推理可能很慢（CPU 上 codegen-350M 常见 >20s），不要阻塞建议 UI。
						// 即使有相同位置的请求在处理中，也允许新的请求触发
						// 这样用户可以连续调用代码补全功能
						const p = (async () => {
							logger.info(`Starting new backend completion request: ${requestKey}`);
							const backendResult = await suggestViaBackend(document, position, config, logger);
							if (backendResult.error) {
								logger.info(`Backend suggest failed: ${backendResult.error}`);
								return;
							}
							const lmText = backendResult.completion;
							const lmLen = lmText?.length || 0;
							logger.info(`Backend completion received (background). chars=${lmLen}`);
							if (lmLen > 0) {
						inlineCache.set(requestKey, lmText);
						logger.info(`LM completion ready (background). chars=${lmLen} contexts=${backendResult.contexts.length}`);
						// 显示补全选择对话框
						void showCompletionDialog(document, position, lmText, backendResult.contexts);
					}
						})().finally(() => {
							// 无论成功失败，都从缓存中删除
							inFlightBackend.delete(requestKey);
							logger.info(`Backend request completed and removed from cache: ${requestKey}`);
						});
						// 立即添加到缓存，避免重复处理
						inFlightBackend.set(requestKey, p);

						// 尝试短等待：如果后端足够快，则把 LM 项也放进 Ctrl+Space 列表
						const fastTimeoutMs = 3000;
						let lmTextFast = '';
						let timedOut = true;
						await Promise.race([
							inFlightBackend.get(requestKey).then(() => {
								timedOut = false;
								lmTextFast = inlineCache.get(requestKey) || '';
							}),
							new Promise((resolve) => setTimeout(resolve, fastTimeoutMs))
						]);

						if (!timedOut && lmTextFast.length === 0) {
							logger.info('Backend finished quickly, but completion was empty.');
						}

						if (lmTextFast) {
							const lmItem = new vscode.CompletionItem('LM: inferred completion', vscode.CompletionItemKind.Snippet);
							lmItem.sortText = '0000';
							lmItem.insertText = lmTextFast;
							lmItem.detail = 'Generated from backend (dataflow) context';
							lmItem.documentation = new vscode.MarkdownString([
								'**Dataflow Retrieved Context**',
								'',
								'```',
								buildContextBlock(snippets).slice(0, 4000),
								'```',
								'',
								'**Inferred completion**',
								'',
								'```',
								lmTextFast,
								'```'
							].join('\n'));
							result.unshift(lmItem);
						}
					}

				return result;
			}
		}
	);

	const inlineProvider = vscode.languages.registerInlineCompletionItemProvider(
		{ scheme: 'file' },
		{
			provideInlineCompletionItems(document, position) {
				const cacheKey = `${document.uri.toString()}::${position.line}:${position.character}`;
				const text = inlineCache.get(cacheKey);
				if (!text) {
					return { items: [] };
				}

				const item = new vscode.InlineCompletionItem(text, new vscode.Range(position, position));
				item.filterText = text;
				item.insertText = text;
				return { items: [item] };
			}
		}
	);

	const onSave = vscode.workspace.onDidSaveTextDocument(async (document) => {
		if (document.uri.scheme !== 'file') {
			return;
		}
		logger.info(`Document saved, incremental index update: ${document.fileName}`);
		await tokenIndex.indexUri(document.uri, logger);
	});

	const onDelete = vscode.workspace.onDidDeleteFiles(async (event) => {
		for (const file of event.files) {
			tokenIndex.removeFile(file);
			logger.info(`File removed from index: ${file.fsPath}`);
		}
	});

	const onConfigChanged = vscode.workspace.onDidChangeConfiguration(async (event) => {
		if (event.affectsConfiguration('aiCodeComplete')) {
			logger.info('Configuration changed, rebuilding index.');
			await tokenIndex.rebuild(logger);
		}
	});

	// 监听文档内容改变，当用户手动更改代码时清除预览
	const onDocumentChange = vscode.workspace.onDidChangeTextDocument((event) => {
		// 检查是否有活动的建议
		if (completionSuggestions.size === 0) return;

		const editor = vscode.window.activeTextEditor;
		if (!editor) return;

		// 检查是否是当前活动编辑器的文档改变
		const documentUri = editor.document.uri.toString();
		if (event.document.uri.toString() !== documentUri) return;

		// 检查改变的内容是否是用户输入（通过 contentChanges）
		if (event.contentChanges.length > 0) {
			// 用户手动更改了代码，清除预览
			logger.info('Document content changed, clearing completion previews');
			clearAllPreviews();
		}
	});

	// 补全建议数据存储
	const completionSuggestions = new Map();

	// 存储装饰器，用于清除预览
	const decorationTypes = new Map();

	// 设置上下文变量，用于控制 Tab 键行为
	vscode.commands.executeCommand('setContext', 'ai-code-complete.hasSuggestion', false);

	// 清除所有预览装饰器和建议的辅助函数
	const clearAllPreviews = () => {
		// 清除所有装饰器
		for (const [key, { decorationType, editor }] of decorationTypes.entries()) {
			editor.setDecorations(decorationType, []);
			decorationType.dispose();
		}
		decorationTypes.clear();

		// 清除所有建议
		completionSuggestions.clear();

		// 重置上下文变量
		vscode.commands.executeCommand('setContext', 'ai-code-complete.hasSuggestion', false);
	};

	// 注册命令 - 使用当前光标位置
	const applyCompletionCommand = vscode.commands.registerCommand('ai-code-complete.applyCompletion', async () => {
		const editor = vscode.window.activeTextEditor;
		if (!editor) return;

		const position = editor.selection.active;
		const key = `${editor.document.uri.toString()}`;
		const suggestions = completionSuggestions.get(key) || [];
		const suggestion = suggestions.find(s => s.line === position.line && s.character === position.character);

		if (suggestion) {
			await editor.edit(editBuilder => {
				editBuilder.insert(position, suggestion.completion);
			});
			logger.info('Completion applied by user');

			// 清除所有预览装饰器和建议
			clearAllPreviews();
		}
	});

	// Tab 键按下时自动应用补全
	const tabKeyHandler = vscode.commands.registerCommand('ai-code-complete.tabKey', async () => {
		const editor = vscode.window.activeTextEditor;
		if (!editor) {
			return vscode.commands.executeCommand('tab');
		}

		const position = editor.selection.active;
		const key = `${editor.document.uri.toString()}`;
		const suggestions = completionSuggestions.get(key) || [];
		const suggestion = suggestions.find(s => s.line === position.line && s.character === position.character);

		if (suggestion) {
			// 应用补全
			await editor.edit(editBuilder => {
				editBuilder.insert(position, suggestion.completion);
			});
			logger.info('Completion applied via Tab key');

			// 清除所有预览装饰器和建议
			clearAllPreviews();
		} else {
			// 如果没有补全建议，执行默认的 Tab 行为
			return vscode.commands.executeCommand('tab');
		}
	});

	const viewContextCommand = vscode.commands.registerCommand('ai-code-complete.viewContext', () => {
		const editor = vscode.window.activeTextEditor;
		if (!editor) return;

		const position = editor.selection.active;
		const key = `${editor.document.uri.toString()}`;
		const suggestions = completionSuggestions.get(key) || [];
		const suggestion = suggestions.find(s => s.line === position.line && s.character === position.character);

		if (suggestion) {
			const contextText = suggestion.contexts.map((ctx, idx) => `<h3>${idx + 1}. Context ${idx + 1} </h3>
<p>${ctx}</p>`).join('\n\n');
			const panel = vscode.window.createWebviewPanel(
				'codeCompleteContext',
				'AI Code Complete Context',
				vscode.ViewColumn.Beside,
				{ enableScripts: false }
			);
			panel.webview.html = `
				<!DOCTYPE html>
				<html>
				<head>
					<meta charset="UTF-8">
					<title>AI Code Complete Context</title>
					<style>
						body { font-family: monospace; white-space: pre-wrap; padding: 2px; }
						.context { margin-bottom: 5px; padding: 2px; border: 1px solid #ddd; }
						/*title,h2,h3{text-align: center;}*/
						title,h2,h3,p{margin-left: 10px;}
					</style>
				</head>
				<body>
					<h2>Context Used for Completion</h2>
					${contextText.replace(/\n/g, '<br>')}
				</body>
				</html>
			`;
			logger.info('Context view opened by user');
		}
	});

	const cancelCompletionCommand = vscode.commands.registerCommand('ai-code-complete.cancelCompletion', () => {
		// 清除所有预览装饰器和建议
		clearAllPreviews();
		logger.info('Completion cancelled by user');
	});

	// 显示补全建议
	const showCompletionDialog = async (document, position, completionText, contexts) => {
		const key = `${document.uri.toString()}`;
		const suggestions = completionSuggestions.get(key) || [];

		// 添加新的补全建议
		suggestions.push({
			line: position.line,
			character: position.character,
			completion: completionText,
			contexts: contexts
		});

		completionSuggestions.set(key, suggestions);

		// 设置上下文变量，表示有活动的建议
		vscode.commands.executeCommand('setContext', 'ai-code-complete.hasSuggestion', true);

		// 在光标上方显示补全代码预览
		const editor = vscode.window.activeTextEditor;
		if (editor) {
			// 创建一个临时的装饰器来显示补全预览
			const decorationType = vscode.window.createTextEditorDecorationType({
				before: {
					contentText: completionText,
					color: '#888888',
					fontStyle: 'italic'
				},
				rangeBehavior: vscode.DecorationRangeBehavior.ClosedOpen
			});

			const decorations = [{
				range: new vscode.Range(position, position),
				hoverMessage: 'AI Code Complete Suggestion'
			}];

			editor.setDecorations(decorationType, decorations);

			// 保存装饰器，用于后续清除
			const suggestionKey = `${position.line}:${position.character}`;
			decorationTypes.set(`${key}:${suggestionKey}`, { decorationType, editor });

			// 30秒后自动清理装饰器
			setTimeout(() => {
				const saved = decorationTypes.get(`${key}:${suggestionKey}`);
				if (saved) {
					saved.editor.setDecorations(saved.decorationType, []);
					saved.decorationType.dispose();
					decorationTypes.delete(`${key}:${suggestionKey}`);
				}
			}, 30000);
		}

		logger.info('Completion suggestion displayed');
	};

	context.subscriptions.push(
		refreshIndexCommand, 
		provider, 
		applyCompletionCommand,
		viewContextCommand,
		cancelCompletionCommand,
		tabKeyHandler,
		onSave, 
		onDelete, 
		onConfigChanged, 
		onDocumentChange,
		logger.output
	);

	void tokenIndex.rebuild(logger);
	void syncBackendIndex(false);
}

function deactivate() {}

module.exports = {
	activate,
	deactivate
};
