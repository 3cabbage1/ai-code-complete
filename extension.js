const vscode = require('vscode');

const DEFAULT_MAX_FILES = 300;
const DEFAULT_MAX_SUGGESTIONS = 40;
const DEFAULT_MAX_TOKEN_LENGTH = 48;
const DEFAULT_MAX_FILE_SIZE_KB = 256;

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

class WorkspaceTokenIndex {
	constructor() {
		this.tokenFrequency = new Map();
		this.fileTokens = new Map();
		this.isBuilding = false;
	}

	clear() {
		this.tokenFrequency.clear();
		this.fileTokens.clear();
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
				await this.indexUri(uri, true, logger);
			}
			logger?.info(`Index rebuild completed. tokens=${this.tokenFrequency.size}`);
		} finally {
			this.isBuilding = false;
		}
	}

	async indexUri(uri, isFullRebuild = false, logger) {
		try {
			const config = getConfig();
			if (!isFullRebuild) {
				this.removeFileTokens(uri);
			}

			const stat = await vscode.workspace.fs.stat(uri);
			if (stat.size > config.maxFileSizeKB * 1024) {
				logger?.info(`Skip large file: ${uri.fsPath}`);
				return;
			}

			const data = await vscode.workspace.fs.readFile(uri);
			const content = Buffer.from(data).toString('utf8');
			const tokens = extractTokens(content, config.maxTokenLength);
			this.fileTokens.set(uri.toString(), tokens);
			for (const token of tokens) {
				this.tokenFrequency.set(token, (this.tokenFrequency.get(token) || 0) + 1);
			}
		} catch (error) {
			logger?.info(`Index failed for ${uri.fsPath}: ${error?.message || 'unknown error'}`);
		}
	}

	removeFileTokens(uri) {
		const key = uri.toString();
		const existing = this.fileTokens.get(key);
		if (!existing) {
			return;
		}

		for (const token of existing) {
			const next = (this.tokenFrequency.get(token) || 0) - 1;
			if (next <= 0) {
				this.tokenFrequency.delete(token);
			} else {
				this.tokenFrequency.set(token, next);
			}
		}

		this.fileTokens.delete(key);
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
		enableLogging: config.get('enableLogging', true)
	};
}

function extractTokens(content, maxTokenLength) {
	const results = new Set();
	const regex = /\b[A-Za-z_][A-Za-z0-9_]{2,}\b/g;
	let match = regex.exec(content);

	while (match) {
		const token = match[0];
		if (token.length <= maxTokenLength && !isLikelyKeyword(token)) {
			results.add(token);
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

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
	const tokenIndex = new WorkspaceTokenIndex();
	const logger = createLogger();

	logger.info('Extension activated.');

	const refreshIndexCommand = vscode.commands.registerCommand('ai-code-complete.rebuildIndex', async () => {
		logger.info('Manual rebuild command triggered.');
		await vscode.window.withProgress(
			{ location: vscode.ProgressLocation.Notification, title: 'AI Code Complete: rebuilding index...' },
			async () => {
				await tokenIndex.rebuild(logger);
			}
		);
		vscode.window.showInformationMessage('AI Code Complete: index rebuilt.');
	});

	const provider = vscode.languages.registerCompletionItemProvider(
		{ scheme: 'file' },
		{
			provideCompletionItems(document, position, _token, completionContext) {
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
				const candidates = tokenIndex.getTokenCandidates(prefix, config.maxSuggestions);
				logger.info(`Completion requested: mode=${triggerLabel} file=${document.fileName} prefix="${prefix}" resultCount=${candidates.length}`);
				return candidates.map((token, index) => {
					const item = new vscode.CompletionItem(token, vscode.CompletionItemKind.Text);
					item.sortText = String(index).padStart(4, '0');
					item.insertText = token;
					item.detail = 'From workspace index';
					return item;
				});
			}
		}
	);

	const onSave = vscode.workspace.onDidSaveTextDocument(async (document) => {
		if (document.uri.scheme !== 'file') {
			return;
		}
		logger.info(`Document saved, incremental index update: ${document.fileName}`);
		await tokenIndex.indexUri(document.uri, false, logger);
	});

	const onDelete = vscode.workspace.onDidDeleteFiles(async (event) => {
		for (const file of event.files) {
			tokenIndex.removeFileTokens(file);
			logger.info(`File removed from index: ${file.fsPath}`);
		}
	});

	const onConfigChanged = vscode.workspace.onDidChangeConfiguration(async (event) => {
		if (event.affectsConfiguration('aiCodeComplete')) {
			logger.info('Configuration changed, rebuilding index.');
			await tokenIndex.rebuild(logger);
		}
	});

	context.subscriptions.push(refreshIndexCommand, provider, onSave, onDelete, onConfigChanged, logger.output);

	void tokenIndex.rebuild(logger);
}

function deactivate() {}

module.exports = {
	activate,
	deactivate
};
