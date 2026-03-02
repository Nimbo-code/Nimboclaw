#if os(iOS)
import Foundation
import NimboCore
import OpenClawGatewayCore

actor NimboLLMProvider: GatewayLocalLLMToolCallableProvider {
    let kind: GatewayLocalLLMProviderKind = .nimboLocal
    let model: String

    private let modelManager: NimboModelManager

    init(modelManager: NimboModelManager, modelName: String) {
        self.modelManager = modelManager
        self.model = modelName
    }

    // MARK: - GatewayLocalLLMProvider

    func complete(_ request: GatewayLocalLLMRequest) async throws -> GatewayLocalLLMResponse {
        let (inferenceManager, tokenizer) = try await self.resolveEngine()
        let messages = self.convertMessages(request.messages, systemPrompt: request.systemPrompt)
        let tokens = tokenizer.applyChatTemplate(input: messages, addGenerationPrompt: true)
        let maxTokens = 2048
        let temperature: Float = 0.7

        let (_, _, responseText) = try await inferenceManager.generateResponse(
            initialTokens: tokens,
            temperature: temperature,
            maxTokens: maxTokens,
            eosTokens: tokenizer.eosTokenIds,
            tokenizer: tokenizer)

        let trimmed = responseText.trimmingCharacters(in: .whitespacesAndNewlines)
        return GatewayLocalLLMResponse(
            text: trimmed,
            model: self.model,
            provider: .nimboLocal,
            usageInputTokens: tokens.count,
            usageOutputTokens: nil)
    }

    // MARK: - GatewayLocalLLMToolCallableProvider

    func completeWithTools(_ request: GatewayLocalLLMToolRequest) async throws -> GatewayLocalLLMToolResponse {
        let (inferenceManager, tokenizer) = try await self.resolveEngine()

        let toolSystemBlock = self.buildToolSystemPrompt(tools: request.tools)
        let messages = self.convertToolMessages(
            request.messages,
            systemPrompt: request.systemPrompt,
            toolSystemBlock: toolSystemBlock)
        let tokens = tokenizer.applyChatTemplate(input: messages, addGenerationPrompt: true)
        let maxTokens = 2048
        let temperature: Float = 0.7

        let (_, _, responseText) = try await inferenceManager.generateResponse(
            initialTokens: tokens,
            temperature: temperature,
            maxTokens: maxTokens,
            eosTokens: tokenizer.eosTokenIds,
            tokenizer: tokenizer)

        let trimmed = responseText.trimmingCharacters(in: .whitespacesAndNewlines)
        let parsed = self.parseToolCalls(from: trimmed)

        return GatewayLocalLLMToolResponse(
            text: parsed.text,
            toolCalls: parsed.toolCalls,
            model: self.model,
            provider: .nimboLocal,
            usageInputTokens: tokens.count,
            usageOutputTokens: nil)
    }

    // MARK: - Private

    private func resolveEngine() async throws -> (InferenceManager, NimboCore.Tokenizer) {
        let mgr = await MainActor.run { self.modelManager.inferenceManager }
        let tok = await MainActor.run { self.modelManager.tokenizer }
        guard let mgr, let tok else {
            throw GatewayLocalLLMProviderError.notConfigured
        }
        return (mgr, tok)
    }

    private func convertMessages(
        _ messages: [GatewayLocalLLMMessage],
        systemPrompt: String?) -> [NimboCore.Tokenizer.ChatMessage]
    {
        var result: [NimboCore.Tokenizer.ChatMessage] = []
        if let sys = systemPrompt, !sys.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            result.append(.system(sys))
        }
        for msg in messages {
            let role = msg.role.lowercased()
            switch role {
            case "system":
                result.append(.system(msg.text))
            case "user":
                result.append(.user(msg.text))
            case "assistant":
                result.append(.assistant(msg.text))
            default:
                result.append(.user(msg.text))
            }
        }
        return result
    }

    private func convertToolMessages(
        _ messages: [GatewayLocalLLMToolMessage],
        systemPrompt: String?,
        toolSystemBlock: String) -> [NimboCore.Tokenizer.ChatMessage]
    {
        var result: [NimboCore.Tokenizer.ChatMessage] = []

        // Combine user system prompt with tool definitions
        var combinedSystem = toolSystemBlock
        if let sys = systemPrompt, !sys.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            combinedSystem = sys + "\n\n" + toolSystemBlock
        }
        result.append(.system(combinedSystem))

        for msg in messages {
            switch msg.role {
            case .system:
                // Already handled above
                break
            case .user:
                result.append(.user(msg.text ?? ""))
            case .assistant:
                if !msg.toolCalls.isEmpty {
                    // Reconstruct the assistant's tool call text
                    var text = msg.text ?? ""
                    for call in msg.toolCalls {
                        text += "\n<tool_call>{\"name\":\"\(call.name)\",\"arguments\":\(call.argumentsJSON)}</tool_call>"
                    }
                    result.append(.assistant(text))
                } else {
                    result.append(.assistant(msg.text ?? ""))
                }
            case .tool:
                // Format tool result as a user message with clear labeling
                let toolName = msg.name ?? "unknown"
                let toolResult = msg.text ?? ""
                result
                    .append(
                        .user(
                            "<tool_response>\n{\"name\":\"\(toolName)\",\"result\":\(self.escapeJSON(toolResult))}\n</tool_response>"))
            }
        }
        return result
    }

    private func buildToolSystemPrompt(tools: [GatewayLocalLLMToolDefinition]) -> String {
        var toolDefs: [[String: Any]] = []
        for tool in tools {
            var def: [String: Any] = [
                "name": tool.name,
                "description": tool.description,
            ]
            if let params = self.jsonValueToAny(tool.parameters) {
                def["parameters"] = params
            }
            toolDefs.append(def)
        }

        let toolsJSON: String = if let data = try? JSONSerialization.data(
            withJSONObject: toolDefs,
            options: [.sortedKeys]),
            let str = String(data: data, encoding: .utf8)
        {
            str
        } else {
            "[]"
        }

        return """
        You have access to the following tools:
        \(toolsJSON)

        To call a tool, respond with:
        <tool_call>{"name": "tool_name", "arguments": {"key": "value"}}</tool_call>

        You may call multiple tools. After all tool calls, wait for the results before continuing.
        If you don't need to call a tool, respond normally without <tool_call> tags.
        """
    }

    private struct ParsedToolOutput {
        let text: String
        let toolCalls: [GatewayLocalLLMToolCall]
    }

    private func parseToolCalls(from text: String) -> ParsedToolOutput {
        var toolCalls: [GatewayLocalLLMToolCall] = []
        var cleanText = text

        let pattern = #"<tool_call>\s*(\{.*?\})\s*</tool_call>"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) else {
            return ParsedToolOutput(text: text, toolCalls: [])
        }

        let matches = regex.matches(in: text, range: NSRange(text.startIndex..., in: text))
        for match in matches {
            guard let jsonRange = Range(match.range(at: 1), in: text) else { continue }
            let jsonStr = String(text[jsonRange])
            guard let data = jsonStr.data(using: .utf8),
                  let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let name = parsed["name"] as? String
            else {
                continue
            }
            let arguments = parsed["arguments"] ?? [String: Any]()
            let argsJSON: String = if let argsData = try? JSONSerialization.data(withJSONObject: arguments),
                                      let argsStr = String(data: argsData, encoding: .utf8)
            {
                argsStr
            } else {
                "{}"
            }
            let callID = "call_\(UUID().uuidString.prefix(8))"
            toolCalls.append(GatewayLocalLLMToolCall(id: callID, name: name, argumentsJSON: argsJSON))
        }

        // Remove tool call tags from the text
        if !toolCalls.isEmpty {
            cleanText = regex.stringByReplacingMatches(
                in: text,
                range: NSRange(text.startIndex..., in: text),
                withTemplate: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }

        return ParsedToolOutput(text: cleanText, toolCalls: toolCalls)
    }

    private func jsonValueToAny(_ value: GatewayJSONValue) -> Any? {
        switch value {
        case .null:
            return NSNull()
        case let .bool(b):
            return b
        case let .integer(i):
            return i
        case let .double(d):
            return d
        case let .string(s):
            return s
        case let .array(arr):
            return arr.compactMap { self.jsonValueToAny($0) }
        case let .object(dict):
            var result = [String: Any]()
            for (k, v) in dict {
                result[k] = self.jsonValueToAny(v)
            }
            return result
        }
    }

    private func escapeJSON(_ string: String) -> String {
        if let data = try? JSONSerialization.data(withJSONObject: string),
           let escaped = String(data: data, encoding: .utf8)
        {
            return escaped
        }
        return "\"\(string)\""
    }
}
#endif
