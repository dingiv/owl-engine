/**
 * Deterministic wire-shape demo for ImageEvictChat.
 *
 * This does NOT implement an agent loop and does NOT execute tools. It seeds a
 * small transcript so the request body shows the exact three-image window.
 * In real code, the assistant tool_calls come from completeAndRecord(), and the
 * caller executes tools and appends their results with addToolResult().
 *
 *   node demo.ts                 # print the request body (default)
 *   node demo.ts --send          # send 4-image then 3-image requests
 *   node demo.ts --base-url=http://127.0.0.1:8135 --send
 */

import { ImageEvictChat, type ChatTool } from "./image-evict-chat.ts";

const PNG = {
  red: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKklEQVR42mO4o6FBU8QwasGoBaMWjFowasGoBaMWjFowasGoBaMWDBULAIahsD0l4r5QAAAAAElFTkSuQmCC",
  green: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKklEQVR42mPQ2BJAU8QwasGoBaMWjFowasGoBaMWjFowasGoBaMWDBULABZ0sD0prtT4AAAAAElFTkSuQmCC",
  blue: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKklEQVR42mPQiLpDU8QwasGoBaMWjFowasGoBaMWjFowasGoBaMWDBULAK1aeEwg+9jvAAAAAElFTkSuQmCC",
  yellow: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAK0lEQVR42mN4dkKDpohh1IJRC0YtGLVg1IJRC0YtGLVg1IJRC0YtGCoWAAAgRVhqewqI+QAAAABJRU5ErkJggg==",
};

const TOOLS: ChatTool[] = [
  {
    type: "function",
    function: {
      name: "screenshot",
      description: "Capture the current desktop screen.",
      parameters: { type: "object", properties: {} },
    },
  },
];

const SYSTEM =
  "You are a GUI agent. Tool results contain a screenshot. " +
  "Use only the visible screen state when answering.";

function argValue(name: string, fallback: string): string {
  const prefix = `--${name}=`;
  const found = process.argv.find((arg) => arg.startsWith(prefix));
  return found ? found.slice(prefix.length) : fallback;
}

function redactDataUrls(body: unknown): unknown {
  return JSON.parse(
    JSON.stringify(body, (_key, value) => {
      if (typeof value === "string" && value.startsWith("data:image/")) {
        return `<data-url ${value.length} chars>`;
      }
      return value;
    }),
  );
}

const baseUrl = argValue("base-url", "http://127.0.0.1:8134");
const model = argValue("model", "qwen3.8-27b");
const send = process.argv.includes("--send");
const session =
  process.env.OWL_SESSION ?? `ts-image-evict-demo-${Date.now()}`;

const chat = new ImageEvictChat({
  baseUrl,
  model,
  session,
  system: SYSTEM,
  imageLimit: 3,
  maxTokens: 512,
  tools: TOOLS,
});

chat.addUser("Open the settings dialog and report the final screen colour.");

// Deterministic replay: a real client gets these assistant messages from
// completeAndRecord(). Each tool result is one screenshot + stable text bounds.
const screens = [PNG.red, PNG.green, PNG.blue, PNG.yellow];
for (let index = 0; index < screens.length; index += 1) {
  const callId = `call_demo_${index + 1}`;
  chat.appendAssistant({
    role: "assistant",
    content: null,
    tool_calls: [
      {
        id: callId,
        type: "function",
        function: { name: "screenshot", arguments: "{}" },
      },
    ],
  });
  chat.addToolResult(
    callId,
    `Step ${index + 1}: captured the current screen.`,
    screens[index],
  );
}

if (!send) {
  console.log(`history images before window = ${chat.imageCount}`);
  const removed = chat.enforceImageLimit();
  console.log(`history images after window  = ${chat.imageCount}`);
  console.log(`pruned = ${removed.length}`);

  const body = chat.buildRequestBody({ tools: TOOLS });
  console.log("\n--- POST /v1/chat/completions body ---");
  console.log(JSON.stringify(redactDataUrls(body), null, 2));
  console.log(
    "\n(dry-run; add --send to send a 4-image request first, then a 3-image request in the same session)",
  );
} else {
  // Send the pre-prune history first. This is what makes the second request a
  // real server-side deletion (same session, old prefix has four images), not
  // just a cold request that happens to contain three images.
  console.log(`\n--- first request (pre-prune, images=${chat.imageCount}) ---`);
  const first = await chat.completeAndRecord({ tools: TOOLS, prune: false });
  console.log(`finish_reason = ${first.response.choices[0]?.finish_reason}`);
  console.log(`tool_calls = ${first.message.tool_calls?.length ?? 0}`);

  // Tool execution is the caller's job. We only append the returned content;
  // no new screenshot here, so the next request can prune the oldest image.
  for (const call of first.message.tool_calls ?? []) {
    chat.addToolResult(call.id, "Synthetic tool result; no new screenshot.");
  }

  console.log(`history images before prune = ${chat.imageCount}`);
  const removed = chat.enforceImageLimit();
  console.log(`history images after prune  = ${chat.imageCount}`);
  console.log(`pruned = ${removed.length}`);
  chat.addUser(
    "Using the remaining screenshots, state the latest visible colour.",
  );

  const second = await chat.completeAndRecord({ tools: TOOLS });
  const usage = second.usage;
  const cached = usage?.prompt_tokens_details?.cached_tokens ?? 0;
  const prompt = usage?.prompt_tokens ?? 0;
  console.log("\n--- second request (post-prune) ---");
  console.log(`finish_reason = ${second.response.choices[0]?.finish_reason}`);
  console.log(`cached_tokens = ${cached}/${prompt}`);
  console.log(`tool_calls = ${second.message.tool_calls?.length ?? 0}`);
  console.log(`text = ${second.message.content ?? ""}`);
  console.log(
    "if the prefix/delete was absorbed, cached_tokens stays high and the server log shows `[img-evict] 前缀无感删除`",
  );
}
