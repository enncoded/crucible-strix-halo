// Exercise only Pi AI's HTTP transport. Do not load agent configuration or tools.
import { pathToFileURL } from "node:url";
import path from "node:path";
const root = process.env.CRUCIBLE_PI_AI_DIR;
if (!root) throw new Error("Set CRUCIBLE_PI_AI_DIR to the installed pi-ai package.");
const { streamSimple } = await import(pathToFileURL(path.join(root, "dist/api/openai-completions.js")));
const { createInitialSystemMessage } = await import(pathToFileURL(path.join(root, "dist/utils/transcript.js")));
const tools = process.argv[2] !== "text" ? [{ name: "add", description: "Add two integers.",
  parameters: { type: "object", properties: { a: { type: "integer" }, b: { type: "integer" } },
    required: ["a", "b"], additionalProperties: false } }] : [];
const model = { type: "chat", id: "crucible", name: "Crucible", provider: "crucible",
  api: "openai-completions", baseUrl: "http://127.0.0.1:8790/v1", reasoning: false, input: ["text"],
  contextWindow: 32768, maxTokens: 8192, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  compat: { supportsStore: false, supportsDeveloperRole: false, supportsReasoningEffort: false,
    supportsLongCacheRetention: false, supportsStrictMode: false, supportsUsageInStreaming: true,
    supportsFinishReason: true, maxTokensField: "max_completion_tokens" } };
if (process.argv[2] === "agent") {
  // Pi's agent loop: a streamed call that the server ends with finish "length" must not run, and the loop continues.
  const { agentLoop } = await import(pathToFileURL(path.join(root, "../pi-agent-core/dist/index.js")));
  const executed = [];
  const add = { ...tools[0], label: "add",
    execute: async (id, params) => { executed.push(params);
      return { content: [{ type: "text", text: String(params.a + params.b) }], details: {} }; } };
  const context = { messages: [createInitialSystemMessage("Answer briefly.", [add])], tools: [add] };
  const prompt = { role: "user", content: "Add 17 and 25.", timestamp: 0 };
  const config = { model, convertToLlm: (messages) => messages, apiKey: "crucible-local", maxTokens: 32,
    timeoutMs: 5000, maxRetries: 0 };
  const loop = agentLoop([prompt], context, config, undefined, streamSimple);
  const events = [];
  for await (const event of loop) events.push(event.type);
  const messages = await loop.result();
  console.log(JSON.stringify({ executed, events,
    turns: messages.filter((m) => m.role === "assistant").map((m) => ({ stopReason: m.stopReason,
      toolCalls: m.content.filter((c) => c.type === "toolCall").length })),
    results: messages.filter((m) => m.role === "toolResult").map((m) => ({ isError: m.isError,
      text: m.content.map((c) => c.text).join("") })) }));
  process.exit(0);
}
const stream = streamSimple(model, { messages: [createInitialSystemMessage("Answer briefly.", tools),
  { role: "user", content: "Add 17 and 25.", timestamp: 0 }] },
  { apiKey: "crucible-local", maxTokens: 32, timeoutMs: 5000, maxRetries: 0, cacheRetention: "long" });
const events = [];
for await (const event of stream) events.push(event.type);
const result = await stream.result();
if (result.stopReason === "error") throw new Error(result.errorMessage);
console.log(JSON.stringify({ events, stopReason: result.stopReason, usage: result.usage, content: result.content }));
