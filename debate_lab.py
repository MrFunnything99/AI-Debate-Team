#!/usr/bin/env python3
"""
Human vs. AI Team Debate Application
====================================
Press F2 to toggle recording. Speak your argument, press F2 again to stop.
The AI team (Researcher, Red Teamer, Speaker) will discuss and deliver a rebuttal.
"""

import asyncio
import os
import tempfile
import threading
from pathlib import Path

# Load environment variables (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =============================================================================
# CONFIGURATION - Easy model mapping per agent
# =============================================================================

# Latest models - customize per agent (Researcher=Gemini, RedTeamer=Claude, Speaker=GPT)
MODELS = {
    "openai": "gpt-4o",
    "gemini": "gemini-1.5-pro",   # Large context, good for research
    "anthropic": "claude-3-5-sonnet-20241022",
}

# Build OAI_CONFIG_LIST for AutoGen
def build_config_list():
    config_list = []
    
    if os.getenv("OPENAI_API_KEY"):
        config_list.append({
            "model": MODELS["openai"],
            "api_key": os.getenv("OPENAI_API_KEY"),
        })
    
    if os.getenv("GOOGLE_API_KEY"):
        config_list.append({
            "model": MODELS["gemini"],
            "api_key": os.getenv("GOOGLE_API_KEY"),
            "api_type": "google",
        })
    
    if os.getenv("ANTHROPIC_API_KEY"):
        config_list.append({
            "model": MODELS["anthropic"],
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "api_type": "anthropic",
        })
    
    if not config_list:
        raise ValueError(
            "No API keys found! Set OPENAI_API_KEY, GOOGLE_API_KEY, or ANTHROPIC_API_KEY in .env"
        )
    
    return config_list


# =============================================================================
# AUDIO & RECORDING (Non-blocking)
# =============================================================================

RECORDING_STATE_IDLE = 0
RECORDING_STATE_ACTIVE = 1
recording_state = RECORDING_STATE_IDLE
recording_lock = threading.Lock()

def get_recording_state():
    with recording_lock:
        return recording_state

def set_recording_state(state):
    global recording_state
    with recording_lock:
        recording_state = state


def record_audio_to_file() -> str | None:
    """Record audio using sounddevice, save to temp WAV file. Blocks until F2 pressed again."""
    import sounddevice as sd
    import numpy as np
    import scipy.io.wavfile as wav_io
    
    sample_rate = 16000  # Whisper prefers 16kHz
    channels = 1
    
    frames = []
    
    def audio_callback(indata, frames_count, time_info, status):
        if status:
            print(f"  [Audio] {status}")
        frames.append(indata.copy())
    
    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype=np.float32,
        blocksize=int(sample_rate * 0.1),  # 100ms blocks
        callback=audio_callback,
    )
    
    stream.start()
    
    while get_recording_state() == RECORDING_STATE_ACTIVE:
        import time
        time.sleep(0.1)
    
    stream.stop()
    stream.close()
    
    if not frames:
        return None
    
    audio_data = np.concatenate(frames, axis=0)
    
    # Save to temp file
    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        wav_io.write(path, sample_rate, (audio_data * 32767).astype(np.int16))
        return path
    except Exception:
        os.close(fd)
        os.unlink(path)
        raise


def transcribe_audio(audio_path: str) -> str:
    """Transcribe using faster-whisper (preferred) or openai-whisper."""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_path, language="en", beam_size=5)
        text = " ".join(s.text for s in segments).strip()
        return text or "(no speech detected)"
    except ImportError:
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_path, language="en")
            return (result.get("text") or "").strip() or "(no speech detected)"
        except ImportError:
            raise ImportError(
                "Install faster-whisper or openai-whisper: pip install faster-whisper"
            )


def text_to_speech(text: str) -> None:
    """Convert text to speech and play using edge-tts."""
    import edge_tts
    
    async def _tts():
        communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        await communicate.save(path)
        return path
    
    path = asyncio.run(_tts())
    try:
        try:
            from playsound import playsound
            playsound(path, block=True)
        except ImportError:
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    import time
                    time.sleep(0.1)
            except ImportError:
                print("(Install playsound or pygame for audio playback)")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# =============================================================================
# GOOGLE SEARCH TOOL (for Researcher)
# =============================================================================

def web_search(query: str, num_results: int = 5) -> str:
    """Search the web for factual information. Used by the Researcher agent."""
    try:
        from googlesearch import search as gsearch
        results = list(gsearch(query, num_results=num_results))
        if not results:
            return f"No results found for: {query}"
        return "\n".join(f"- {r}" for r in results[:num_results])
    except ImportError:
        try:
            # Serper API fallback
            import requests
            api_key = os.getenv("SERPER_API_KEY")
            if not api_key:
                return f"[Search unavailable: install googlesearch-python or set SERPER_API_KEY]"
            resp = requests.get(
                "https://google.serper.dev/search",
                params={"q": query, "num": num_results},
                headers={"X-API-KEY": api_key},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            organic = data.get("organic", [])
            return "\n".join(f"- {o.get('title', '')}: {o.get('link', '')}" for o in organic[:num_results])
        except Exception as e:
            return f"[Search error: {e}]"


# =============================================================================
# F2 KEY LISTENER (Global hotkey)
# =============================================================================

def start_key_listener(on_toggle):
    """Start global F2 key listener in a daemon thread."""
    def listen():
        try:
            from pynput import keyboard
            def on_press(key):
                try:
                    if key == keyboard.Key.f2:
                        on_toggle()
                except AttributeError:
                    pass
            
            with keyboard.Listener(on_press=on_press) as listener:
                listener.join()
        except ImportError:
            try:
                import keyboard as kb
                kb.on_press_key("f2", lambda _: on_toggle())
                kb.wait()
            except ImportError:
                print("Install pynput or keyboard for F2 detection: pip install pynput")
    
    t = threading.Thread(target=listen, daemon=True)
    t.start()


# =============================================================================
# AUTOGEN DEBATE TEAM
# =============================================================================

def run_debate_round(transcribed_text: str) -> str:
    """Run the AI team huddle and return the Speaker's final rebuttal."""
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    
    try:
        from autogen.agentchat import register_function
    except ImportError:
        register_function = None
    
    config_list = build_config_list()
    
    # Filter configs by provider for each agent
    def filter_config(models):
        return [c for c in config_list if c.get("model") in models]
    
    # Prefer: Researcher=Gemini, RedTeamer=Claude, Speaker=GPT
    gemini_config = filter_config([MODELS["gemini"]]) or config_list
    claude_config = filter_config([MODELS["anthropic"]]) or config_list
    gpt_config = filter_config([MODELS["openai"]]) or config_list
    
    llm_config_base = {"cache_seed": 42, "temperature": 0.7}
    
    # --- UserProxy: Represents the human, NEVER auto-replies ---
    user_proxy = UserProxyAgent(
        name="Human",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
        description="The human debater. Presents arguments to the AI team.",
    )
    
    # --- Researcher: Fact-checker with Google Search ---
    researcher = AssistantAgent(
        name="Researcher",
        system_message="""You are a ruthless fact-checker. When the human makes a claim, verify it immediately using the web_search tool.
If they are wrong, provide the correct data to the team. If they are right, confirm with sources.
Do not speak to the human directly; speak only to the team. Be concise.""",
        llm_config={
            **llm_config_base,
            "config_list": gemini_config,
        },
        description="Fact-checks claims using web search.",
    )
    
    # --- Red Teamer: Logic filter ---
    red_teamer = AssistantAgent(
        name="Red_Teamer",
        system_message="""You are a logic filter. Before the Speaker responds to the human, analyze the planned argument.
If it contains fallacies, weak points, or unsupported claims, reject it and demand a stronger version from the team.
Only approve when the argument is logically sound and well-supported. Speak only to the team.""",
        llm_config={
            **llm_config_base,
            "config_list": claude_config,
        },
        description="Validates logic and identifies weaknesses.",
    )
    
    # --- Speaker: The face, delivers final rebuttal ---
    speaker = AssistantAgent(
        name="Speaker",
        system_message="""You are the ONLY agent who speaks to the human. Synthesize the Researcher's facts and the Red Teamer's logic 
into a sharp, persuasive, 1-paragraph rebuttal. Be charismatic but firm. Address the human directly.
Keep it concise - one compelling paragraph. Do not use bullet points in your final response.""",
        llm_config={
            **llm_config_base,
            "config_list": gpt_config,
        },
        description="Delivers the final rebuttal to the human.",
    )
    
    # Register search tool: Researcher calls it, UserProxy executes it
    if register_function:
        register_function(
            web_search,
            caller=researcher,
            executor=user_proxy,
            description="Search the web for facts and information. Use when you need to verify claims or find current data.",
        )
    
    # GroupChat: agents discuss, then Speaker responds to Human
    groupchat = GroupChat(
        agents=[user_proxy, researcher, red_teamer, speaker],
        messages=[],
        max_round=10,
        speaker_selection_method="auto",
    )
    
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config={"config_list": gpt_config, "cache_seed": 42},
    )
    
    # Initiate: Human sends transcribed argument
    chat_result = user_proxy.initiate_chat(
        manager,
        message=f"[Human's argument]: {transcribed_text}",
        summary_method="last_msg",
    )
    
    # Use summary (last message) or extract Speaker's reply from history
    final_message = (getattr(chat_result, "summary", None) or "").strip()
    if not final_message:
        history = getattr(chat_result, "chat_history", None) or getattr(groupchat, "messages", [])
        for msg in reversed(history):
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if isinstance(content, str) and content.strip() and len(content) > 30:
                name = msg.get("name", "") if isinstance(msg, dict) else getattr(msg, "name", "")
                if name == "Speaker" or "Speaker" in str(msg):
                    final_message = content
                    break
                final_message = content
                break
    
    return final_message or "No response generated."


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    import time
    from colorama import init, Fore, Style
    init(autoreset=True)
    
    print(f"""
{Fore.CYAN}{'='*60}
  HUMAN vs. AI TEAM DEBATE LAB
{'='*60}{Style.RESET_ALL}

{Fore.YELLOW}Press F2 to START recording your argument.
Press F2 again to STOP and send to the AI team.{Style.RESET_ALL}

The team will: fact-check → validate logic → deliver rebuttal.
""")
    
    recorded_audio_queue = []
    
    def _recording_worker():
        try:
            path = record_audio_to_file()
            if path:
                recorded_audio_queue.append(path)
        except Exception as e:
            recorded_audio_queue.append(("error", str(e)))
    
    def on_f2_toggle():
        state = get_recording_state()
        if state == RECORDING_STATE_IDLE:
            set_recording_state(RECORDING_STATE_ACTIVE)
            t = threading.Thread(target=_recording_worker, daemon=True)
            t.start()
            print(f"\n{Fore.RED}{Style.BRIGHT}🔴 RECORDING... (Press F2 again to stop){Style.RESET_ALL}\n")
        else:
            set_recording_state(RECORDING_STATE_IDLE)
            print(f"\n{Fore.GREEN}✓ Recording stopped. Processing...{Style.RESET_ALL}\n")
    
    start_key_listener(on_f2_toggle)
    
    while True:
        import time
        time.sleep(0.3)
        
        if not recorded_audio_queue:
            continue
        
        item = recorded_audio_queue.pop(0)
        if isinstance(item, tuple) and item[0] == "error":
            print(f"{Fore.RED}Error: {item[1]}{Style.RESET_ALL}")
            continue
        
        audio_path = item
        try:
            print(f"{Fore.CYAN}📝 Transcribing...{Style.RESET_ALL}")
            transcribed = transcribe_audio(audio_path)
            try:
                os.unlink(audio_path)
            except OSError:
                pass
            
            print(f"\n{Fore.WHITE}{Style.BRIGHT}You said:{Style.RESET_ALL} {transcribed}\n")
            
            if not transcribed or transcribed == "(no speech detected)":
                print(f"{Fore.YELLOW}No speech detected. Try again.{Style.RESET_ALL}\n")
                continue
            
            print(f"{Fore.MAGENTA}{Style.BRIGHT}🧠 TEAM THINKING...{Style.RESET_ALL}\n")
            rebuttal = run_debate_round(transcribed)
            
            print(f"\n{Fore.GREEN}{Style.BRIGHT}Speaker's Rebuttal:{Style.RESET_ALL}\n{rebuttal}\n")
            
            print(f"{Fore.CYAN}🔊 Playing rebuttal...{Style.RESET_ALL}\n")
            text_to_speech(rebuttal)
            print(f"{Fore.GREEN}Done. Press F2 to record your next argument.{Style.RESET_ALL}\n")
            
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()