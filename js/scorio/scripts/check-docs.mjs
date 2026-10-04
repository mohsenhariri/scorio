import { mkdtemp, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { format, isDeepStrictEqual } from "node:util";
import ts from "typescript";

const packageRoot = fileURLToPath(new URL("../", import.meta.url));
const docsRoot = path.join(packageRoot, "docs");
const buildRoot = path.join(docsRoot, "_build");

function expectedOutputs(code) {
  const source = ts.createSourceFile("example.ts", code, ts.ScriptTarget.ES2020, true);
  const expected = [];
  function visit(node) {
    if (ts.isCallExpression(node) && node.expression.getText(source) === "console.log") {
      const lineEnd = code.indexOf("\n", node.end);
      const comment = code.slice(node.end, lineEnd < 0 ? code.length : lineEnd).match(/\/\/\s*=>\s*(.+)/);
      expected.push(comment ? JSON.parse(comment[1]) : undefined);
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
  return expected;
}

function matchesOutput(actual, expected) {
  if (typeof actual === "number" && typeof expected === "number") {
    return Math.abs(actual - expected) <= 1e-10 * Math.max(1, Math.abs(expected));
  }
  if (Array.isArray(actual) && Array.isArray(expected)) {
    return actual.length === expected.length && actual.every((value, i) => matchesOutput(value, expected[i]));
  }
  return isDeepStrictEqual(actual, expected);
}

async function checkExamples() {
  // Keep snippets inside the package so "scorio" resolves to this checkout's
  // exports, including the CommonJS entry points, rather than a registry copy.
  const temporary = await mkdtemp(path.join(packageRoot, ".docs-examples-"));
  const examples = [];
  let checkedOutputs = 0;
  try {
    const documents = (await readdir(docsRoot)).filter((name) => name.endsWith(".md")).sort();
    for (const document of documents) {
      const markdown = await readFile(path.join(docsRoot, document), "utf8");
      const blocks = markdown.matchAll(/^```(ts|typescript|js|javascript|cjs)\r?\n([\s\S]*?)^```/gm);
      let block = 0;
      for (const match of blocks) {
        block += 1;
        const language = match[1];
        const code = match[2];
        const extension = language === "cjs" ? ".cjs" : language.startsWith("t") ? ".ts" : ".mjs";
        const source = path.join(temporary, `${path.parse(document).name}-${block}${extension}`);
        const line = markdown.slice(0, match.index).split("\n").length;
        await writeFile(source, code);
        examples.push({ source, code, language, location: `${document}:${line}` });
      }
    }
    if (!examples.length) throw new Error("No documentation examples found.");

    const program = ts.createProgram({
      rootNames: examples.filter((example) => example.language !== "cjs").map((example) => example.source),
      options: {
        target: ts.ScriptTarget.ES2020,
        module: ts.ModuleKind.NodeNext,
        moduleResolution: ts.ModuleResolutionKind.NodeNext,
        strict: true,
        noUncheckedIndexedAccess: true,
        allowJs: true,
        checkJs: true,
        noEmit: true,
        skipLibCheck: true,
        types: [],
        lib: ["lib.es2020.d.ts", "lib.dom.d.ts"],
      },
    });
    const diagnostics = ts.getPreEmitDiagnostics(program);
    if (diagnostics.length) {
      throw new Error(ts.formatDiagnosticsWithColorAndContext(diagnostics, {
        getCanonicalFileName: (name) => name,
        getCurrentDirectory: () => packageRoot,
        getNewLine: () => "\n",
      }));
    }

    for (const example of examples) {
      let executable = example.source;
      if (example.language.startsWith("t")) {
        executable = example.source.replace(/\.ts$/, ".mjs");
        const compiled = ts.transpileModule(example.code, {
          compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.ESNext },
        });
        await writeFile(executable, compiled.outputText);
      }
      const output = [];
      const log = console.log;
      console.log = (...values) => output.push(values);
      try {
        // Each file has its own module scope. Importing a .cjs snippet still
        // executes its require() calls through the package's CommonJS exports.
        await import(pathToFileURL(executable).href);
      } catch (error) {
        throw new Error(`${example.location} failed`, { cause: error });
      } finally {
        console.log = log;
      }
      expectedOutputs(example.code).forEach((expected, i) => {
        if (expected === undefined) return;
        const values = output[i];
        const actual = values?.length === 1 ? values[0] : values;
        if (!matchesOutput(actual, expected)) {
          throw new Error(`${example.location}: expected ${format(expected)}, got ${format(actual)}`);
        }
        checkedOutputs += 1;
      });
      if (process.env.SCORIO_DOCS_VERBOSE === "1") {
        console.log(`${example.location}\n${output.map((values) => format(...values)).join("\n")}`);
      }
    }
    console.log(`Checked ${examples.length} runnable documentation examples (ESM, TypeScript, and CommonJS).`);
    console.log(`Verified ${checkedOutputs} documented outputs, allowing for floating-point rounding.`);
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

async function htmlFiles(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const filename = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await htmlFiles(filename));
    else if (entry.name.endsWith(".html")) files.push(filename);
  }
  return files;
}

function decodeEntities(value) {
  return value.replace(/&amp;/g, "&").replace(/&quot;/g, '"').replace(/&#39;/g, "'")
    .replace(/&lt;/g, "<").replace(/&gt;/g, ">");
}

async function checkLinks() {
  const pages = new Map();
  for (const filename of await htmlFiles(buildRoot)) {
    const html = await readFile(filename, "utf8");
    const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((match) => decodeEntities(match[1])));
    pages.set(filename, { html, ids });
  }
  if (!pages.has(path.join(buildRoot, "index.html"))) {
    throw new Error("Build the documentation with npm run docs before checking links.");
  }

  const failures = [];
  const existence = new Map();
  let links = 0;
  for (const [filename, { html }] of pages) {
    const relative = path.relative(buildRoot, filename).split(path.sep).join("/");
    const base = new URL(relative, "https://docs.invalid/");
    for (const match of html.matchAll(/<(?:a|link|script|img)\b[^>]*?\b(?:href|src)="([^"]+)"[^>]*>/g)) {
      const href = decodeEntities(match[1]);
      if (/^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(href)) continue;
      const url = new URL(href, base);
      let target = path.join(buildRoot, decodeURIComponent(url.pathname));
      if (url.pathname.endsWith("/")) target = path.join(target, "index.html");
      if (!existence.has(target)) {
        existence.set(target, await stat(target).then((info) => info.isFile()).catch(() => false));
      }
      const fragment = decodeURIComponent(url.hash.slice(1));
      if (!existence.get(target) || (fragment && pages.has(target) && !pages.get(target).ids.has(fragment))) {
        failures.push(`${relative}: ${href}`);
      }
      links += 1;
    }
  }
  if (failures.length) throw new Error(`Broken documentation links:\n${[...new Set(failures)].join("\n")}`);
  console.log(`Checked ${links} local links and assets across ${pages.size} HTML pages.`);
}

await checkExamples();
await checkLinks();
