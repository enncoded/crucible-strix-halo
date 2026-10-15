// Pi AI transport with thinking on: one reply, then a replay of that reply as history. No agent tools run.
import { pathToFileURL } from "node:url";
import path from "node:path";
const root = process.env.CRUCIBLE_PI_AI_DIR;
if (!root) throw new Error("Set CRUCIBLE_PI_AI_DIR to the installed pi-ai package.");
const { streamSimple } = await import(pathToFileURL(path.join(root, "dist/api/openai-completions.js")));
const { createInitialSystemMessage } = await import(pathToFileURL(path.join(root, "dist/utils/transcript.js")));
const model = { type: "chat", id: "crucible", name: "Crucible", provider: "crucible",
  api: "openai-completions", baseUrl: "http://127.0.0.1:8790/v1", reasoning: true, input: ["text"],
  contextWindow: 32768, maxTokens: 64, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  thinkingLevelMap: { minimal: "low", low: "low", medium: "medium", high: "xhigh", xhigh: "xhigh" },
  compat: { supportsStore: false, supportsDeveloperRole: false, supportsReasoningEffort: true,
    supportsLongCacheRetention: false, supportsStrictMode: false, supportsUsageInStreaming: true,
    supportsFinishReason: true, maxTokensField: "max_completion_tokens", thinkingFormat: "qwen" } };
const options = { apiKey: "crucible-local", maxTokens: 64, timeoutMs: 5000, maxRetries: 0, reasoning: "high" };
const user = { role: "user", content: "Add 17 and 25.", timestamp: 0 };
const first = streamSimple(model, { messages: [createInitialSystemMessage("Answer briefly.", []), user] }, options);
for await (const _ of first);
const reply = await first.result();
if (reply.stopReason === "error") throw new Error(reply.errorMessage);
const second = streamSimple(model, { messages: [createInitialSystemMessage("Answer briefly.", []), user, reply,
  { role: "user", content: "Thanks.", timestamp: 1 }] }, options);
for await (const _ of second);
const next = await second.result();
if (next.stopReason === "error") throw new Error(next.errorMessage);
console.log(JSON.stringify({ content: reply.content, usage: reply.usage }));
