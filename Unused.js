#!/usr/bin/env node
/* scripts/unused-vscode-like.cjs
 * VS Code-like references via tsserver protocol
 * - Scan JS/JSX files
 * - Collect variable/import/function/class declarations (可按需扩展)
 * - Ask tsserver "references" at identifier position
 * - Decide unused: only definition ref, no other refs
 * - Stream results to txt while scanning
 */

const fs = require("fs");
const path = require("path");
const cp = require("child_process");
const fg = require("fast-glob");
const chokidar = require("chokidar");
const ts = require("typescript");

// ---------------------- Config ----------------------
const CWD = process.cwd();
const OUT_TXT = path.resolve(CWD, "unused-vars-vscode-like.txt");

// 默认只扫 src；你也可以改成 "**/*.{js,jsx}"
const INCLUDE = process.env.INCLUDE
  ? [process.env.INCLUDE]
  : ["src/**/*.{js,jsx}", "**/*.{js,jsx}"];

const EXCLUDE = [
  "**/node_modules/**",
  "**/dist/**",
  "**/build/**",
  "**/coverage/**",
  "**/.next/**",
  "**/out/**",
  "**/*.min.js",
];

// 是否 watch（实时监控变更并重跑）
const WATCH = process.argv.includes("--watch");

// 是否把 export 的声明也当作可能 unused（默认也统计，但标注）
const INCLUDE_EXPORTED = process.argv.includes("--include-exported");

// 限流：避免 tsserver 请求爆炸（大项目建议调小，比如 200）
const MAX_DECLS_PER_FILE = Number(process.env.MAX_DECLS_PER_FILE || 2000);
// ----------------------------------------------------

function writeHeader(stream, filesCount) {
  stream.write(
    `# VS Code-like unused scan via tsserver\n` +
      `# time: ${new Date().toISOString()}\n` +
      `# cwd: ${CWD}\n` +
      `# files: ${filesCount}\n` +
      `# output: ${OUT_TXT}\n\n`
  );
}

function posToLineCol(text, offset) {
  // 1-based line/col like VS Code
  let line = 1, col = 1;
  for (let i = 0; i < offset; i++) {
    const ch = text.charCodeAt(i);
    if (ch === 10) { // \n
      line++;
      col = 1;
    } else {
      col++;
    }
  }
  return { line, col };
}

function isProbablyGenerated(filePath) {
  const p = filePath.replaceAll("\\", "/");
  return (
    p.includes("/dist/") ||
    p.includes("/build/") ||
    p.includes("/coverage/") ||
    p.includes("/.next/") ||
    p.includes("/out/")
  );
}

// ---- tsserver protocol helpers (Content-Length framing) ----
function createTsServerClient() {
  const tsserverPath = require.resolve("typescript/lib/tsserver.js");
  const child = cp.spawn(process.execPath, [tsserverPath], {
    stdio: ["pipe", "pipe", "pipe"],
  });

  let seq = 0;
  let buffer = Buffer.alloc(0);

  /** @type {Map<number, {resolve: Function, reject: Function}>} */
  const pending = new Map();

  function send(request) {
    return new Promise((resolve, reject) => {
      const s = ++seq;
      request.seq = s;
      request.type = "request";
      const json = JSON.stringify(request);
      const payload = `Content-Length: ${Buffer.byteLength(json, "utf8")}\r\n\r\n${json}`;
      pending.set(s, { resolve, reject });
      child.stdin.write(payload, "utf8");
    });
  }

  function onData(chunk) {
    buffer = Buffer.concat([buffer, chunk]);

    while (true) {
      const headerEnd = buffer.indexOf("\r\n\r\n");
      if (headerEnd === -1) return;

      const header = buffer.slice(0, headerEnd).toString("utf8");
      const m = header.match(/Content-Length:\s*(\d+)/i);
      if (!m) {
        // malformed, drop
        buffer = buffer.slice(headerEnd + 4);
        continue;
      }
      const len = Number(m[1]);
      const total = headerEnd + 4 + len;
      if (buffer.length < total) return;

      const body = buffer.slice(headerEnd + 4, total).toString("utf8");
      buffer = buffer.slice(total);

      let msg;
      try {
        msg = JSON.parse(body);
      } catch {
        continue;
      }

      if (msg.type === "response" && typeof msg.request_seq === "number") {
        const p = pending.get(msg.request_seq);
        if (p) {
          pending.delete(msg.request_seq);
          if (msg.success) p.resolve(msg);
          else p.reject(msg);
        }
      }
      // event messages are ignored for this script
    }
  }

  child.stdout.on("data", onData);
  child.stderr.on("data", () => {
    // ignore, tsserver can be chatty
  });

  async function close() {
    try {
      await send({ command: "shutdown", arguments: {} });
    } catch {}
    child.kill();
  }

  return { send, close };
}

// ---- AST collection (JS) ----
// Collect declaration identifiers that represent "variables" in the sense of unused detection.
function collectDeclsFromJs(filePath, text) {
  const sf = ts.createSourceFile(
    filePath,
    text,
    ts.ScriptTarget.Latest,
    true,
    filePath.endsWith(".jsx") ? ts.ScriptKind.JSX : ts.ScriptKind.JS
  );

  /** @type {Array<{name: string, offset: number, kind: string, exported: boolean}>} */
  const decls = [];

  function addIdent(id, kind, exported) {
    if (!id || !id.text) return;
    decls.push({
      name: id.text,
      offset: id.getStart(sf, false),
      kind,
      exported,
    });
  }

  function isNodeExported(node) {
    // Covers `export const ...`, `export function ...`, `export class ...`
    const mods = node.modifiers;
    if (!mods) return false;
    return mods.some((m) => m.kind === ts.SyntaxKind.ExportKeyword);
  }

  function collectBindingName(nameNode, kind, exported) {
    if (!nameNode) return;
    if (ts.isIdentifier(nameNode)) {
      addIdent(nameNode, kind, exported);
      return;
    }
    // destructuring patterns: collect identifiers inside
    const visit = (n) => {
      if (ts.isIdentifier(n)) addIdent(n, kind, exported);
      ts.forEachChild(n, visit);
    };
    visit(nameNode);
  }

  function visit(node) {
    // variable decl: const/let/var
    if (ts.isVariableStatement(node)) {
      const exported = isNodeExported(node);
      for (const decl of node.declarationList.declarations) {
        collectBindingName(decl.name, "variable", exported);
      }
    }

    // function decl
    if (ts.isFunctionDeclaration(node) && node.name) {
      addIdent(node.name, "function", isNodeExported(node));
    }

    // class decl
    if (ts.isClassDeclaration(node) && node.name) {
      addIdent(node.name, "class", isNodeExported(node));
    }

    // import bindings (ESM)
    if (ts.isImportDeclaration(node) && node.importClause) {
      const clause = node.importClause;

      if (clause.name) addIdent(clause.name, "import", false);

      if (clause.namedBindings) {
        if (ts.isNamespaceImport(clause.namedBindings)) {
          addIdent(clause.namedBindings.name, "import", false);
        } else if (ts.isNamedImports(clause.namedBindings)) {
          for (const spec of clause.namedBindings.elements) {
            // import { a as b } => b is local binding
            addIdent(spec.name, "import", false);
          }
        }
      }
    }

    ts.forEachChild(node, visit);
  }

  visit(sf);
  return decls;
}

// ---- main scan logic ----
async function runScanOnce() {
  const client = createTsServerClient();

  // Configure similar to VS Code defaults (approx)
  await client.send({
    command: "configure",
    arguments: {
      hostInfo: "vscode-like-unused-script",
      preferences: {
        quotePreference: "auto",
      },
    },
  });

  const files = fg.sync(INCLUDE, { cwd: CWD, absolute: true, ignore: EXCLUDE });

  // reset output file (realtime write)
  const stream = fs.createWriteStream(OUT_TXT, { flags: "w" });
  writeHeader(stream, files.length);

  // Tell tsserver we are in an "inferred project"
  // We open files so tsserver can build a project graph like VS Code does.
  // (In JS projects, this is the common VS Code behavior.)
  let totalUnused = 0;

  for (let i = 0; i < files.length; i++) {
    const filePath = files[i];
    if (isProbablyGenerated(filePath)) continue;

    let text;
    try {
      text = fs.readFileSync(filePath, "utf8");
    } catch {
      continue;
    }

    // open in tsserver
    await client.send({
      command: "open",
      arguments: {
        file: filePath,
        fileContent: text,
        scriptKindName: filePath.endsWith(".jsx") ? "JSX" : "JS",
      },
    });

    const decls = collectDeclsFromJs(filePath, text).slice(0, MAX_DECLS_PER_FILE);

    for (const d of decls) {
      if (!INCLUDE_EXPORTED && d.exported) continue;

      const { line, col } = posToLineCol(text, d.offset);

      // tsserver "references" expects 1-based line/offset
      let resp;
      try {
        resp = await client.send({
          command: "references",
          arguments: {
            file: filePath,
            line,
            offset: col,
            // includeDeclaration default is true in tsserver;
            // VS Code usually shows declaration separately/filtered.
            includeDeclaration: true,
          },
        });
      } catch {
        continue;
      }

      const refs = (resp.body && resp.body.refs) ? resp.body.refs : [];

      // VS Code-like unused check:
      // - if only reference is the declaration itself => unused
      // tsserver returns refs with isDefinition flags.
      const nonDefRefs = refs.filter((r) => !r.isDefinition);

      if (nonDefRefs.length === 0) {
        totalUnused++;

        const exportedTag = d.exported ? " exported" : "";
        const lineStr =
          `${filePath}:${line}:${col}  ${d.kind}${exportedTag}  ${d.name}\n`;
        stream.write(lineStr);
      }
    }

    // close file to reduce tsserver memory (optional)
    await client.send({ command: "close", arguments: { file: filePath } });

    // progress
    if ((i + 1) % 25 === 0) {
      stream.write(`\n# progress: ${i + 1}/${files.length}, unused so far: ${totalUnused}\n\n`);
    }
  }

  stream.write(`\n# done. total unused: ${totalUnused}\n`);
  stream.end();

  await client.close();

  console.log(`✅ done. wrote ${totalUnused} items -> ${OUT_TXT}`);
}

async function main() {
  if (!WATCH) {
    await runScanOnce();
    return;
  }

  console.log("👀 watch mode on. Will rescan on changes...");
  const watcher = chokidar.watch(INCLUDE, { ignored: EXCLUDE, ignoreInitial: true });

  let timer = null;
  const trigger = () => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      runScanOnce().catch((e) => console.error(e));
    }, 300);
  };

  watcher.on("add", trigger);
  watcher.on("change", trigger);
  watcher.on("unlink", trigger);

  // initial run
  await runScanOnce();
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
