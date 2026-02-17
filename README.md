# AI Debate Lab

A voice-powered debate app where you argue against an AI team. Record your position, and the AI team—Researcher, Red Teamer, and Speaker—will fact-check, critique your logic, and deliver a spoken rebuttal.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

## How It Works

1. **You speak** — Record your argument using the browser microphone.
2. **Researcher** — Searches the web for evidence that opposes your claim (powered by [ddgs](https://github.com/deedy5/ddgs)).
3. **Red Teamer** — Identifies logical fallacies, gaps, and weaknesses in your argument.
4. **Speaker** — Delivers a direct rebuttal, synthesized as spoken audio.

The AI team always argues *against* you. Stress-test your arguments on any topic.

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/MrFunnything99/AI-Debate-Team.git
cd AI-Debate-Team
pip install -r requirements-streamlit.txt
```

### 2. Configure API Key

Copy the example env file and add your [OpenRouter](https://openrouter.ai/) API key (one key for Gemini, Claude, and GPT-4o):

```bash
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY=sk-or-v1-your-key
```

### 3. Run

```bash
streamlit run app.py
```

Open http://localhost:8501 and start debating.

## Requirements

- **Python 3.10+**
- **OpenRouter API key** — Get one at [openrouter.ai](https://openrouter.ai/)
- **Microphone** — For voice input (browser-based)
- **ffmpeg** (optional) — For webm audio conversion; [download](https://ffmpeg.org/download.html)

## Tech Stack

| Component        | Technology                          |
|-----------------|-------------------------------------|
| Multi-agent orchestration | [AutoGen](https://github.com/microsoft/autogen) |
| UI               | Streamlit                           |
| Speech-to-text   | SpeechRecognition + Google (free)   |
| Text-to-speech   | edge-tts (free)                     |
| Web search       | ddgs (no API key)                   |
| LLMs             | OpenRouter (Gemini, Claude, GPT-4o)|

## Credits

This project is built on **[AutoGen](https://github.com/microsoft/autogen)** by Microsoft—a framework for building multi-agent conversations with LLMs. AutoGen powers the coordinated debate flow between the Researcher, Red Teamer, and Speaker agents.

- **AutoGen**: [github.com/microsoft/autogen](https://github.com/microsoft/autogen)
- **OpenRouter**: [openrouter.ai](https://openrouter.ai/) — unified API for multiple LLMs
- **ddgs**: [github.com/deedy5/ddgs](https://github.com/deedy5/ddgs) — web search
- **edge-tts**: Microsoft Edge TTS for speech synthesis

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute. [Contributors](CONTRIBUTORS.md) are listed separately.

## License

MIT
