# Nimboclaw

On-device AI agent for Apple platforms. Runs LLMs locally on the Apple Neural Engine — no cloud API required.

## What It Does

Nimboclaw runs a complete AI agent stack on your iPhone or iPad. The LLM runs directly on the Apple Neural Engine via CoreML, so inference stays entirely on-device. The agent can call tools, browse the web, manage your calendar, and execute multi-step tasks — all without sending a single token to an external server.

```
┌─────────────────────────────────────┐
│           Nimboclaw App             │
│  ┌───────────┐   ┌───────────────┐  │
│  │  Chat UI  │   │  Agent Tools  │  │
│  └─────┬─────┘   └───────┬───────┘  │
│        │                 │          │
│  ┌─────▼─────────────────▼───────┐  │
│  │    Local Gateway Runtime      │  │
│  │  (routing, sessions, memory)  │  │
│  └─────────────┬─────────────────┘  │
│                │                    │
│  ┌─────────────▼─────────────────┐  │
│  │        NimboCore              │  │
│  │  CoreML → Apple Neural Engine │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

## Key Features

**Local LLM Inference** — CoreML models (1–3B parameters) run on the Apple Neural Engine. Supports Llama, Qwen, and other architectures converted via [anemll](https://github.com/Anemll/anemll). No network connection needed for inference.

**Tool Calling** — The local model can invoke tools during conversation. Tool definitions are injected into the system prompt and the agent parses structured `<tool_call>` outputs to execute actions and return results in a loop.

**Agent Workspace** — A persistent file-based workspace gives the agent long-term memory and configurable behavior:
- `SOUL.md` — personality and behavioral principles
- `IDENTITY.md` — agent name, persona, avatar
- `USER.md` — user profile (name, timezone, interests)
- `MEMORY.md` — curated long-term memory across sessions
- `memory/` — daily session logs
- `TOOLS.md` — device-specific tool configuration
- `skills/` — skill definitions for specialized tasks

**Built-in Tools**:
- `web.render` / `web.extract` — fetch and extract web content
- `calendar.*` / `reminders.*` — read and create calendar events and reminders
- `contacts.search` — search device contacts
- `location.current` — get current location
- `device.status` — battery, network, storage info
- `photos.*` — access photo library
- `credentials.*` — secure keychain storage for per-skill API keys
- `ls` — list workspace files

**Multi-Provider Fallback** — While local inference is the primary mode, you can also configure cloud providers (OpenAI, Anthropic, Grok, MiniMax) as alternatives or fallbacks. Switch between local and cloud models from the chat toolbar.

**Dream Mode** — Ambient idle animations with configurable wake-word detection.

**Voice** — Talk mode with speech-to-text and text-to-speech for hands-free interaction.

**tvOS** — Run the full agent on Apple TV with remote-friendly UI.

## Getting Started

### 1. Prepare a Model

Convert a supported model to CoreML using [anemll](https://github.com/Anemll/anemll):

```bash
pip install anemll
anemll-convert --model meta-llama/Llama-3.2-1B-Instruct --output ./llama-3.2-1b
```

Transfer the output directory to your device via AirDrop or the Files app. Place it under:

```
On My iPhone > Nimboclaw > models/
```

The model directory must contain `meta.yaml` and the `.mlmodelc` bundles.

### 2. Configure the Provider

1. Open **Settings > LLM Providers**.
2. Tap **Add Provider** and select **Nimbo (On-Device ANE)**.
3. Pick your model directory from the list.
4. Tap **Save & Restart** — the model loads onto the Neural Engine.

### 3. Chat

Open the Chat tab and start talking to your local agent. Tool calls execute on-device. No data leaves your phone.

## Clone & Build

```bash
git clone https://github.com/Nimbo-code/Nimboclaw.git
cd Nimboclaw
```

Generate the Xcode project (requires [XcodeGen](https://github.com/yonaskolb/XcodeGen)):

```bash
brew install xcodegen swiftformat swiftlint
xcodegen generate
open Nimboclaw.xcodeproj
```

### Build Targets

| Scheme | Platform | Description |
|--------|----------|-------------|
| **Nimboclaw** | iOS 18+ | iPhone, iPad, Vision Pro |
| **NimboclawTV** | tvOS 18+ | Apple TV |

### Requirements

- Xcode 16.0+
- Swift 6.0
- Device with Apple Neural Engine (A14+, M1+) for local inference

### Dependencies

All Swift packages are included under `shared/`:
- **NimboCore** — CoreML inference engine (tokenizer, model loader, inference manager)
- **OpenClawKit** — Chat UI, protocol definitions, tool commands
- **OpenClawGatewayCore** — On-device gateway server, session management, LLM routing

## Architecture

The local LLM integration uses an app-layer injection pattern:

- `OpenClawGatewayCore` defines the `GatewayLocalLLMProvider` protocol and routing — it has no dependency on NimboCore.
- `NimboLLMProvider` (app layer) implements the protocol using NimboCore and is injected into the gateway at runtime.
- `NimboModelManager` handles model lifecycle — loading CoreML models onto the Neural Engine, tracking progress, and cleanup.

This keeps the gateway core portable and testable while the ANE-specific code lives in the app target.

## License

See [LICENSE](LICENSE) for details.
