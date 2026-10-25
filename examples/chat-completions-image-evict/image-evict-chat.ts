/**
 * Minimal Chat Completions client component for owl image eviction.
 *
 * This is deliberately NOT an agent loop. It only owns the request-side
 * contract:
 *   - build full `messages` history every turn;
 *   - mark images with `owl_evict: true`;
 *   - append assistant/tool messages with stable ids and text boundaries;
 *   - prune only the oldest image part, never the surrounding text;
 *   - post the whole history to /v1/chat/completions.
 *
 * Tool execution and the agent loop stay in the caller.
 */

export type TextPart = {
  type: "text";
  text: string;
};

/** `owl_evict` is an owl extension and must sit next to `image_url`. */
export type EvictableImagePart = {
  type: "image_url";
  image_url: { url: string };
  owl_evict: true;
};

export type ContentPart = TextPart | EvictableImagePart;

export type ToolCall = {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
};

export type ChatTool = {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters: Record<string, unknown>;
    strict?: boolean;
  };
};

export type AssistantMessage = {
  role: "assistant";
  content: string | null;
  tool_calls?: ToolCall[];
  reasoning_content?: string | null;
};

export type ChatMessage =
  | { role: "system" | "developer"; content: string }
  | { role: "user"; content: string | ContentPart[] }
  | AssistantMessage
  | {
      role: "tool";
      tool_call_id: string;
      content: string | ContentPart[];
      name?: string;
    };

export type Usage = {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_tokens_details?: { cached_tokens?: number };
  completion_tokens_details?: { reasoning_tokens?: number };
};

export type ChatCompletionResponse = {
  id: string;
  choices: Array<{
    index: number;
    message: AssistantMessage;
    finish_reason: string | null;
  }>;
  usage?: Usage;
};

export type ChatRequestBody = {
  model: string;
  messages: ChatMessage[];
  stream: false;
  session?: string;
  max_tokens?: number;
  temperature?: number;
  tools?: ChatTool[];
  tool_choice?: string;
};

export type ImageEvictChatOptions = {
  /** Origin such as http://127.0.0.1:8134 (or an exact /v1/chat/completions URL). */
  baseUrl?: string;
  /** Exact endpoint; wins over baseUrl. */
  endpoint?: string;
  model: string;
  apiKey?: string;
  /** Stable conversation key. Omit to use owl's implicit session. */
  session?: string;
  system?: string;
  /** Keep at most this many images in the full history. Default 3. */
  imageLimit?: number;
  maxTokens?: number;
  temperature?: number;
  tools?: ChatTool[];
  toolChoice?: string;
  fetchImpl?: typeof fetch;
};

export type CompleteOptions = {
  maxTokens?: number;
  temperature?: number;
  tools?: ChatTool[];
  toolChoice?: string;
  /** Default true. false is only for demos/inspection of a pre-prune body. */
  prune?: boolean;
};

export type AssistantStep = {
  response: ChatCompletionResponse;
  message: AssistantMessage;
  usage?: Usage;
};

export class OwlRequestError extends Error {
  readonly status: number;
  readonly body: string;
  readonly url: string;

  constructor(message: string, status: number, body: string, url: string) {
    super(message);
    this.name = "OwlRequestError";
    this.status = status;
    this.body = body;
    this.url = url;
  }
}

export function resolveEndpoint(baseUrl: string): string {
  const trimmed = baseUrl.replace(/\/+$/, "");
  if (trimmed.endsWith("/v1/chat/completions")) return trimmed;
  if (trimmed.endsWith("/v1")) return `${trimmed}/chat/completions`;
  return `${trimmed}/v1/chat/completions`;
}

export function evictableImage(dataUrl: string): EvictableImagePart {
  if (!dataUrl.startsWith("data:image/") || !dataUrl.includes(";base64,")) {
    throw new Error(
      "owl only accepts data:image/...;base64 URLs in this example",
    );
  }
  return {
    type: "image_url",
    image_url: { url: dataUrl },
    owl_evict: true,
  };
}

function textPart(text: string): TextPart {
  return { type: "text", text };
}

function trailingNewline(text: string): string {
  return text.endsWith("\n") ? text : `${text}\n`;
}

function isEvictableImage(part: ContentPart): part is EvictableImagePart {
  return part.type === "image_url" && part.owl_evict === true;
}

function isImagePart(part: ContentPart): part is EvictableImagePart {
  return part.type === "image_url";
}

function clone<T>(value: T): T {
  return structuredClone(value);
}

export class ImageEvictChat {
  private readonly endpoint: string;
  private readonly model: string;
  private readonly apiKey?: string;
  private readonly session?: string;
  private readonly imageLimit: number;
  private readonly maxTokens?: number;
  private readonly temperature?: number;
  private readonly tools?: ChatTool[];
  private readonly toolChoice?: string;
  private readonly fetchImpl: typeof fetch;
  private readonly messages: ChatMessage[] = [];

  /** Images removed by the most recent complete()/enforceImageLimit(). */
  lastPruned: string[] = [];
  lastUsage?: Usage;

  constructor(options: ImageEvictChatOptions) {
    if (!options.model) throw new Error("model is required");
    const limit = options.imageLimit ?? 3;
    if (
      limit !== Number.POSITIVE_INFINITY &&
      (!Number.isInteger(limit) || limit < 1)
    ) {
      throw new Error("imageLimit must be a positive integer or Infinity");
    }

    this.endpoint = resolveEndpoint(
      options.endpoint ?? options.baseUrl ?? "http://127.0.0.1:8134",
    );
    this.model = options.model;
    this.apiKey = options.apiKey;
    this.session = options.session;
    this.imageLimit = limit;
    this.maxTokens = options.maxTokens;
    this.temperature = options.temperature;
    this.tools = options.tools;
    this.toolChoice = options.toolChoice;
    this.fetchImpl = options.fetchImpl ?? fetch;

    if (options.system) this.addSystem(options.system);
  }

  /** Deep copy so callers cannot accidentally break image markers. */
  get history(): ChatMessage[] {
    return clone(this.messages);
  }

  get imageCount(): number {
    let count = 0;
    for (const message of this.messages) {
      if (!Array.isArray(message.content)) continue;
      count += message.content.filter(isImagePart).length;
    }
    return count;
  }

  addSystem(text: string): this {
    this.messages.push({ role: "system", content: text });
    return this;
  }

  addUser(content: string | ContentPart[]): this {
    this.messages.push({ role: "user", content: clone(content) });
    return this;
  }

  /**
   * User image helper. The server normalizes user text parts to `\n`, but
   * keeping a text anchor before and after every image makes the delete point
   * unambiguous and keeps the transcript readable.
   */
  addUserWithImages(text: string, images: string[]): this {
    if (!text.trim()) {
      throw new Error("addUserWithImages needs non-empty text around images");
    }
    const parts: ContentPart[] = [textPart(trailingNewline(text))];
    images.forEach((url, index) => {
      parts.push(evictableImage(url));
      parts.push(textPart(`Screenshot ${index + 1}.\n`));
    });
    this.messages.push({ role: "user", content: parts });
    return this;
  }

  /**
   * Append the assistant message returned by the model. Preserve `tool_calls`
   * and `reasoning_content` exactly: the next request must replay them.
   */
  appendAssistant(message: AssistantMessage): this {
    const next: AssistantMessage = {
      role: "assistant",
      content: message.content ?? null,
    };
    if (message.tool_calls && message.tool_calls.length > 0) {
      next.tool_calls = clone(message.tool_calls);
    }
    if (message.reasoning_content !== undefined) {
      next.reasoning_content = message.reasoning_content;
    }
    this.messages.push(next);
    return this;
  }

  /**
   * Append a tool result. With a screenshot, the wire shape is:
   *
   *   [ text ending in "\n", image(owl_evict), trailing text ]
   *
   * `role:"tool"` arrays are NOT normalized by the server, so the client owns
   * these boundaries. Never delete the whole tool message; remove only the
   * image part later.
   */
  addToolResult(callId: string, text: string, screenshot?: string): this {
    if (!callId) throw new Error("addToolResult needs a tool_call_id");

    let content: string | ContentPart[];
    if (screenshot) {
      content = [
        textPart(trailingNewline(text || "Tool result.")),
        evictableImage(screenshot),
        textPart("End of tool result."),
      ];
    } else {
      content = text;
    }

    this.messages.push({
      role: "tool",
      tool_call_id: callId,
      content,
    });
    return this;
  }

  /**
   * Remove the oldest evictable image part only.
   *
   * Text parts, tool_call_id, message order and assistant tool_calls stay
   * byte-identical. Returns the removed data URL, or undefined if no marked
   * image exists.
   */
  pruneOldestImage(): string | undefined {
    for (const message of this.messages) {
      if (!Array.isArray(message.content)) continue;
      const parts = message.content;
      const index = parts.findIndex(isEvictableImage);
      if (index < 0) continue;

      const hasText = parts.some(
        (part) => part.type === "text" && part.text.trim().length > 0,
      );
      if (!hasText) {
        throw new Error(
          "refusing to remove the only content part; keep text around images",
        );
      }

      const removed = parts.splice(index, 1)[0];
      if (!removed || removed.type !== "image_url") {
        throw new Error("internal error: removed part is not an image");
      }
      return removed.image_url.url;
    }
    return undefined;
  }

  /** Enforce the image window. Returns all removed data URLs. */
  enforceImageLimit(): string[] {
    const removed: string[] = [];
    while (this.imageCount > this.imageLimit) {
      const url = this.pruneOldestImage();
      if (!url) {
        throw new Error(
          `history has ${this.imageCount} images but none are evictable; ` +
            "every image that may be pruned must have owl_evict:true",
        );
      }
      removed.push(url);
    }
    this.lastPruned = removed;
    return removed;
  }

  /** Pure request-body builder. Call enforceImageLimit() first if needed. */
  buildRequestBody(options: CompleteOptions = {}): ChatRequestBody {
    const body: ChatRequestBody = {
      model: this.model,
      messages: clone(this.messages),
      stream: false,
    };
    if (this.session !== undefined) body.session = this.session;

    const maxTokens = options.maxTokens ?? this.maxTokens;
    if (maxTokens !== undefined) body.max_tokens = maxTokens;

    const temperature = options.temperature ?? this.temperature;
    if (temperature !== undefined) body.temperature = temperature;

    const tools = options.tools ?? this.tools;
    if (tools !== undefined) body.tools = clone(tools);

    const toolChoice = options.toolChoice ?? this.toolChoice;
    if (toolChoice !== undefined) body.tool_choice = toolChoice;

    return body;
  }

  /**
   * Send one Chat Completions request with the full history.
   *
   * Auto-prunes before building the body, so the server sees the intended
   * three-image window. Pass `prune:false` only when demonstrating the exact
   * pre-prune body; production callers should leave it alone.
   *
   * This method is non-streaming; a streaming client keeps the same history/prune
   * logic and only swaps the response reader.
   */
  async complete(
    options: CompleteOptions = {},
  ): Promise<ChatCompletionResponse> {
    if (options.prune !== false) this.enforceImageLimit();
    const body = this.buildRequestBody(options);
    const headers: Record<string, string> = {
      "content-type": "application/json",
    };
    if (this.apiKey) headers.authorization = `Bearer ${this.apiKey}`;

    const response = await this.fetchImpl(this.endpoint, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
    const raw = await response.text();

    if (!response.ok) {
      throw new OwlRequestError(
        `owl request failed: HTTP ${response.status}`,
        response.status,
        raw,
        this.endpoint,
      );
    }

    let parsed: ChatCompletionResponse;
    try {
      parsed = JSON.parse(raw) as ChatCompletionResponse;
    } catch {
      throw new OwlRequestError(
        "owl returned non-JSON response",
        response.status,
        raw,
        this.endpoint,
      );
    }

    this.lastUsage = parsed.usage;
    return parsed;
  }

  /**
   * Complete one model round and append the assistant message to history.
   * This is the only piece the caller needs around tool execution:
   *
   *   const step = await chat.completeAndRecord();
   *   for (const call of step.message.tool_calls ?? []) {
   *     chat.addToolResult(call.id, result.text, result.screenshot);
   *   }
   */
  async completeAndRecord(
    options: CompleteOptions = {},
  ): Promise<AssistantStep> {
    const response = await this.complete(options);
    const choice = response.choices[0];
    if (!choice) {
      throw new OwlRequestError(
        "owl response has no choices[0]",
        200,
        JSON.stringify(response),
        this.endpoint,
      );
    }
    this.appendAssistant(choice.message);
    return { response, message: choice.message, usage: response.usage };
  }
}
