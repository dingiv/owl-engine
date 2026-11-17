#!/usr/bin/env node
/**
 * 实战门：用本地 `pi` agent 打真实编码任务，逐请求验 TTFT。
 *
 * 判据（任一不过 = exit 1）：
 *  1. 每个 LLM 请求（owl 的一个 turn）的 TTFT ≤ `--ttft-max`（缺省 5000ms）；
 *  2. 运行期间没有 turn 被弃置 / PoolExhausted / abort 回执超时；
 *  3. agent 正常退出且产物存在（默认 index.html ≥ 2KB、含 canvas）；
 *  4. 结束后 GPU 不残留自旋（util 连续 3 次 ≥90% 且日志不再增长）。
 *
 * TTFT 的真源 = owl server 的 turn 指标行（毫秒，与 OpenAI 客户端解析无关）：
 *   [chat] turn Some(N) 指标: ttft=...ms; prefill=...; decode=...; cache=a/b
 *
 * 用法：
 *   node tools/agent-ttft-probe.mjs
 *   node tools/agent-ttft-probe.mjs --ttft-max 3000 --timeout 900000
 *   node tools/agent-ttft-probe.mjs --log /tmp/owl-3080-fixed.log --keep
 *
 * 依赖：本机 `pi`（~/.pi/agent/models.json 里要有 --provider/--model）、
 *       owl server 已起在 models.json 的 baseUrl 上。
 */

import { spawn } from "node:child_process";
import { copyFileSync, createReadStream, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

// ── 参数 ────────────────────────────────────────────────────────────────────
const args = process.argv.slice(2);
function arg(name, dflt) {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
}
const TTFT_MAX = Number(arg("ttft-max", "5000"));
const TIMEOUT_MS = Number(arg("timeout", String(20 * 60 * 1000)));
const LOG_PATH = arg("log", process.env.OWL_LOG || "/tmp/owl-3080-fixed.log");
const PROVIDER = arg("provider", process.env.OWL_PROBE_PROVIDER || "owl_3080_dual");
const MODEL = arg("model", process.env.OWL_PROBE_MODEL || "qwen3.8-27b");
const KEEP = args.includes("--keep");
// pi 的单请求超时（provider 层）。超过 = pi 主动断流，owl 会 "弃置 turn" ⇒ 本门判 FAIL。
const REQ_TIMEOUT_MS = Number(arg("req-timeout", process.env.OWL_PROBE_REQ_TIMEOUT_MS || "120000"));
const PI_BIN = arg("pi", process.env.PI_BIN || "pi");

const PROMPT = `写一个「鹈鹕骑自行车」的交互 demo，要求：
1. 在当前目录创建 index.html（单文件，无外部依赖，内嵌 CSS/JS）；
2. 用 <canvas> 手绘：一只鹈鹕（大嘴袋、白羽、黑眼）骑在自行车上；
3. 动画：车轮转动、脚踏曲柄旋转、鹈鹕身体轻微起伏、背景移动；
4. 自适应窗口大小，60fps 左右；
5. 写完后用 bash 校验：文件存在、包含 canvas、字节数 > 2000；
6. 最后只回复一行：DONE <文件字节数>
不要创建其他文件，不要跑本地服务器。`;

const PARSE = {
  accept: /\[chat\] 受理 session \S+ turn (\d+)\((.*?)\)/,
  metrics: /\[chat\] turn Some\((\d+)\) 指标: ttft=(\d+)ms; prefill=(\d+)ms\/(\d+)tok=([\d.]+)t\/s; decode=(\d+)ms\/(\d+)tok=([\d.]+)t\/s; cache=(\d+)\/(\d+)/,
  done: /\[chat\] turn Some\((\d+)\) 完成\(([^)]*)\)/,
  gateStart: /\[ttft\] turn (\d+) 起跑: prompt=(\d+) cached=(\d+) new=(\d+) ⇒ 首 token 预算 (\d+)ms/,
  gateTimeout: /\[ttft\] turn (\d+) 首 token 超时: 起跑后 (\d+)ms > 预算 (\d+)ms/,
  bad: /弃置 turn (\d+)|PoolExhausted|abort 回执超时|pump 错误/,
};

const turns = new Map();
function turnOf(id) {
  if (!turns.has(id)) {
    turns.set(id, { id, acceptedAt: null, prompt: null, cached: null, maxNew: null, ttft: null, prefillTok: null, prefillTps: null, decodeTps: null, completion: null, bad: [] });
  }
  return turns.get(id);
}

function feed(line, now) {
  let m = line.match(PARSE.accept);
  if (m) {
    const t = turnOf(Number(m[1]));
    t.acceptedAt = now;
    const body = m[2];
    t.prompt = Number((body.match(/prompt=(\d+)/) || [])[1] ?? NaN);
    t.cached = Number((body.match(/cached=(\d+)/) || [])[1] ?? NaN);
    t.maxNew = Number((body.match(/max_new=(\d+)/) || [])[1] ?? NaN);
    return;
  }
  m = line.match(PARSE.metrics);
  if (m) {
    const t = turnOf(Number(m[1]));
    t.ttft = Number(m[2]);
    t.prefillTok = Number(m[4]);
    t.prefillTps = Number(m[5]);
    t.decodeTps = Number(m[8]);
    t.cache = `${m[9]}/${m[10]}`;
    t.metricsAt = now;
    return;
  }
  m = line.match(PARSE.gateStart);
  if (m) {
    const t = turnOf(Number(m[1]));
    t.gateNew = Number(m[4]);
    t.budgetMs = Number(m[5]);
    return;
  }
  m = line.match(PARSE.gateTimeout);
  if (m) {
    const t = turnOf(Number(m[1]));
    t.gateTimedOut = true;
    t.bad.push(`引擎首 token 门超时: 等 ${m[2]}ms > 预算 ${m[3]}ms`);
    return;
  }
  m = line.match(PARSE.done);
  if (m) {
    const t = turnOf(Number(m[1]));
    const body = m[2];
    t.completion = Number((body.match(/completion=(\d+)/) || [])[1] ?? NaN);
    t.doneAt = now;
    return;
  }
  if (PARSE.bad.test(line)) {
    // 归到最近一个 turn（弃置行自带 id；其它错误归到未完成 turn）
    const mm = line.match(/弃置 turn (\d+)/);
    if (mm) turnOf(Number(mm[1])).bad.push(line.trim());
    else {
      const open = [...turns.values()].filter((t) => t.ttft == null || t.completion == null);
      if (open.length) open[open.length - 1].bad.push(line.trim());
    }
  }
}

/** 从 startOffset 读日志新增内容（按行回调）。 */
function makeTail(startOffset) {
  let offset = startOffset;
  return async function readNew(cb) {
    if (!existsSync(LOG_PATH)) return;
    const size = statSync(LOG_PATH).size;
    if (size <= offset) return;
    await new Promise((resolve) => {
      const s = createReadStream(LOG_PATH, { start: offset, end: size - 1, encoding: "utf8" });
      let buf = "";
      s.on("data", (d) => { buf += d; });
      s.on("end", () => {
        offset = size;
        const now = Date.now();
        for (const line of buf.split("\n")) if (line) feed(line, now);
        resolve();
      });
    });
    cb?.();
  };
}

async function main() {
  if (!existsSync(LOG_PATH)) throw new Error(`日志不存在: ${LOG_PATH}`);
  const dir = await mkdtemp(join(tmpdir(), "pelican-agent-"));
  const outPath = join(dir, "agent-output.txt");
  console.log(`[probe] pi 实战: provider=${PROVIDER} model=${MODEL}`);
  console.log(`[probe] 工作目录: ${dir}`);
  console.log(`[probe] 日志: ${LOG_PATH}  TTFT 上限: ${TTFT_MAX}ms  pi 单请求超时: ${REQ_TIMEOUT_MS}ms  总超时: ${TIMEOUT_MS}ms`);

  // 临时 agent dir：只带 models.json + 受控 settings（请求超时可判、无扩展干扰）
  const agentDir = join(dir, ".pi-agent");
  mkdirSync(agentDir, { recursive: true });
  const srcAgent = process.env.PI_CODING_AGENT_DIR || join(process.env.HOME, ".pi/agent");
  if (existsSync(join(srcAgent, "models.json"))) copyFileSync(join(srcAgent, "models.json"), join(agentDir, "models.json"));
  writeFileSync(join(agentDir, "settings.json"), JSON.stringify({
    httpIdleTimeoutMs: REQ_TIMEOUT_MS,
    retry: { provider: { timeoutMs: REQ_TIMEOUT_MS, maxRetries: 0 }, maxRetries: 0 },
    defaultThinkingLevel: "medium",
  }, null, 2));

  const readNew = makeTail(statSync(LOG_PATH).size);
  const timer = setInterval(() => { readNew().catch(() => {}); }, 400);

  const started = Date.now();
  const child = spawn(PI_BIN, ["-p", "--no-session", "--provider", PROVIDER, "--model", MODEL,
    "--tools", "read,write,edit,bash", "--", PROMPT], {
    cwd: dir,
    env: {
      ...process.env,
      PI_CODING_AGENT_DIR: agentDir,
      PI_OFFLINE: "1",
      OWL_LOG: LOG_PATH,
      OWL_PROVIDER: PROVIDER,
      OWL_MODEL: MODEL,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let stdout = "", stderr = "";
  child.stdout.on("data", (d) => { stdout += d; process.stdout.write(d); });
  child.stderr.on("data", (d) => { stderr += d; });

  let timedOut = false;
  const killer = setTimeout(() => { timedOut = true; child.kill("SIGTERM"); }, TIMEOUT_MS);
  const code = await new Promise((resolve) => child.on("exit", (c, s) => resolve(c ?? (s ? 128 : 1))));
  clearTimeout(killer);
  clearInterval(timer);
  await readNew();
  await new Promise((r) => setTimeout(r, 1500));
  await readNew();

  const wall = Date.now() - started;
  const rows = [...turns.values()].filter((t) => t.prompt != null || t.ttft != null).sort((a, b) => a.id - b.id);

  // ── 产物 ──
  const html = join(dir, "index.html");
  const htmlOk = existsSync(html);
  const htmlBytes = htmlOk ? statSync(html).size : 0;
  const htmlText = htmlOk ? readFileSync(html, "utf8") : "";
  const htmlHasCanvas = /<canvas/i.test(htmlText);

  // ── GPU 残留自旋（日志不再增长 + util 高）──
  let spin = false;
  try {
    const { execSync } = await import("node:child_process");
    const samples = [];
    for (let i = 0; i < 3; i++) {
      const out = execSync("nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader", { encoding: "utf8" });
      const utils = out.trim().split("\n").map((l) => Number(l.split(",")[1].trim()));
      samples.push(Math.max(...utils));
      await new Promise((r) => setTimeout(r, 800));
    }
    spin = samples.every((u) => u >= 90);
  } catch { /* nvidia-smi 不在也不判 */ }

  // ── 判定 ──
  const fails = [];
  if (timedOut) fails.push(`总超时 ${TIMEOUT_MS}ms`);
  if (code !== 0) fails.push(`pi 退出码 ${code}`);
  if (!rows.length) fails.push("日志里没有任何 turn 指标（请求没打到 owl？）");
  for (const t of rows) {
    if (t.ttft == null) fails.push(`turn ${t.id} 无 TTFT（未完成/被弃）`);
    else if (t.ttft > TTFT_MAX) fails.push(`turn ${t.id} TTFT ${t.ttft}ms > ${TTFT_MAX}ms`);
    if (t.gateTimedOut) fails.push(`turn ${t.id} 引擎首 token 门超时（被强停）`);
    if (t.bad.length) fails.push(`turn ${t.id} 现场错误: ${t.bad[0]}`);
  }
  if (!htmlOk) fails.push("index.html 不存在");
  else {
    if (htmlBytes < 2000) fails.push(`index.html 只有 ${htmlBytes}B（<2000）`);
    if (!htmlHasCanvas) fails.push("index.html 里没有 <canvas>");
  }
  if (spin) fails.push("运行结束后 GPU 仍 ≥90% 自旋（疑似设备死锁）");

  // ── 报表 ──
  console.log("\n[probe] 逐请求 TTFT（owl turn 指标为真源）");
  console.log("  turn  prompt  cached  ttft(ms)  prefill(tok/s)  decode(tok/s)  completion");
  for (const t of rows) {
    console.log(
      `  ${String(t.id).padStart(4)}  ${String(t.prompt ?? "-").padStart(6)}  ${t.cache ?? "-"}  ${String(t.ttft ?? "-").padStart(8)}  ${String(t.prefillTps ?? "-").padStart(13)}  ${String(t.decodeTps ?? "-").padStart(12)}  ${String(t.completion ?? "-").padStart(10)}`,
    );
  }
  const ttfts = rows.map((t) => t.ttft).filter((x) => x != null);
  if (ttfts.length) {
    const sorted = [...ttfts].sort((a, b) => a - b);
    console.log(`\n[probe] TTFT n=${ttfts.length} min=${sorted[0]} p50=${sorted[Math.floor(sorted.length / 2)]} max=${sorted[sorted.length - 1]} ms（上限 ${TTFT_MAX}）`);
  }
  console.log(`[probe] wall=${(wall / 1000).toFixed(1)}s  index.html=${htmlBytes}B  canvas=${htmlHasCanvas}  GPU自旋=${spin}`);
  writeFileSync(outPath, `STDOUT\n${stdout}\nSTDERR\n${stderr}\n`);

  if (fails.length) {
    console.log(`\n[probe] FAIL (${fails.length})`);
    for (const f of fails) console.log(`  ✗ ${f}`);
    if (!KEEP) console.log(`[probe] 现场保留: ${dir}`);
    process.exitCode = 1;
  } else {
    console.log(`\n[probe] ALL PASS  （产物: ${html}）`);
    if (!KEEP) console.log(`[probe] 产物目录: ${dir}`);
  }
}

main().catch((e) => { console.error("[probe] 崩溃:", e); process.exit(2); });
