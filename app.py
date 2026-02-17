#!/usr/bin/env python3
"""
AI Debate Lab - Mission Control Dashboard
Streamlit-based Human vs. AI Team Debate
"""

import asyncio
import os
import tempfile

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

# =============================================================================
# CONFIG - OpenRouter (Gemini, Claude, GPT-4o)
# =============================================================================

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODELS = {
    "researcher": "google/gemini-2.0-flash-001",
    "red_teamer": "anthropic/claude-3.5-sonnet",
    "speaker": "openai/gpt-4o",
}


# =============================================================================
# WEB SEARCH (ddgs metasearch - no API key)
# =============================================================================

def _is_bad_results(results: list, query: str) -> bool:
    """True if results look like dictionary definitions or are irrelevant."""
    if not results:
        return True
    titles = " ".join(r.get("title", "") for r in results[:3]).lower()
    bodies = " ".join(r.get("body", "") for r in results[:3]).lower()
    combined = titles + " " + bodies
    # Dictionary/definition spam
    if "definition" in combined and ("dictionary" in combined or "meaning" in combined or "merriam-webster" in combined):
        return True
    # Query words should appear in results
    words = [w for w in query.lower().split() if len(w) > 3]
    if words and not any(w in combined for w in words):
        return True
    return False


def _format_results(results: list, num_results: int) -> str:
    """Format ddgs text results as markdown."""
    out = []
    for r in results[:num_results]:
        title = r.get("title", "") or r.get("name", "")
        href = r.get("href", "") or r.get("url", "")
        body = (r.get("body", "") or r.get("snippet", "") or "")[:250]
        out.append(f"- **{title}**: {href}\n  {body}...")
    return "\n".join(out) if out else ""


def web_search(query: str, num_results: int = 5) -> str:
    """Search the web. Used by Researcher agent."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            from ddgs import DDGS
            ddgs = DDGS()
            # 1) Try text() with auto backend (tries multiple engines)
            for attempt, backend in enumerate(("auto", "bing", "mojeek", "yahoo", "duckduckgo")):
                try:
                    results = ddgs.text(query, max_results=max(num_results, 8), backend=backend)
                    results = list(results) if not isinstance(results, list) else results
                    if results and not _is_bad_results(results, query):
                        return _format_results(results, num_results)
                except Exception:
                    continue
            # 2) Fallback: news() often returns different, useful results
            try:
                news = ddgs.news(query, max_results=max(num_results, 8), backend="auto")
                news_list = list(news) if not isinstance(news, list) else news
                if news_list:
                    formatted = []
                    for n in news_list[:num_results]:
                        title = n.get("title", "")
                        url = n.get("url", "")
                        body = (n.get("body", "") or "")[:250]
                        formatted.append(f"- **{title}**: {url}\n  {body}...")
                    return "\n".join(formatted)
            except Exception:
                pass
            # 3) Simplified query (first 4–5 meaningful words)
            words = [w for w in query.split() if len(w) > 2][:5]
            if len(words) >= 2 and " ".join(words) != query:
                simple = " ".join(words)
                try:
                    results = list(ddgs.text(simple, max_results=num_results, backend="auto"))
                    if results and not _is_bad_results(results, simple):
                        return _format_results(results, num_results)
                except Exception:
                    pass
            return f"No relevant results found for: {query}"
        except ImportError:
            return "[Search unavailable: pip install ddgs]"
        except Exception as e:
            return f"[Search error: {e}]"


# =============================================================================
# SPEECH-TO-TEXT: SpeechRecognition + recognize_google (free, no API key)
# =============================================================================

def _ensure_wav(audio_bytes: bytes, original_suffix: str) -> str:
    """
    Ensure we have a WAV file. If Streamlit returns webm, convert with pydub.
    Returns path to temp WAV file.
    """
    if original_suffix.lower() in (".wav", ".wave"):
        path = tempfile.mktemp(suffix=".wav")
        with open(path, "wb") as f:
            f.write(audio_bytes)
        return path

    # webm or other format - convert with pydub
    try:
        from pydub import AudioSegment
        path_in = tempfile.mktemp(suffix=original_suffix or ".webm")
        with open(path_in, "wb") as f:
            f.write(audio_bytes)
        path_out = tempfile.mktemp(suffix=".wav")
        audio = AudioSegment.from_file(path_in)
        audio.export(path_out, format="wav")
        try:
            os.unlink(path_in)
        except OSError:
            pass
        return path_out
    except Exception:
        # Fallback: try writing as wav anyway (Streamlit often returns wav)
        path = tempfile.mktemp(suffix=".wav")
        with open(path, "wb") as f:
            f.write(audio_bytes)
        return path


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe using SpeechRecognition + recognize_google (free, no API key).
    Returns transcribed text or empty string on failure.
    """
    import speech_recognition as sr

    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)
        text = r.recognize_google(audio_data, language="en-US")
        return (text or "").strip()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""
    except Exception:
        return ""


# =============================================================================
# TEXT-TO-SPEECH: edge-tts
# =============================================================================

def text_to_speech_async(text: str, output_path: str, voice: str = "en-US-AriaNeural") -> None:
    """Generate TTS with edge-tts (high-quality neural voice)."""
    import edge_tts

    async def _tts():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)

    asyncio.run(_tts())


# =============================================================================
# AUTOGEN TEAM
# =============================================================================

def get_or_create_agents():
    """Create agents once, cache in session_state."""
    if "autogen_agents" in st.session_state:
        return st.session_state.autogen_agents

    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

    try:
        from autogen.agentchat import register_function
    except ImportError:
        register_function = None

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY in .env")

    def make_config(model_id: str) -> list:
        return [{
            "model": MODELS[model_id],
            "api_key": api_key,
            "base_url": OPENROUTER_BASE,
        }]

    llm_base = {"cache_seed": 42, "temperature": 0.7}

    user_proxy = UserProxyAgent(
        name="Human",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config=False,
        description="The human debater. Executes tool calls (e.g. web_search) for the Researcher.",
    )

    researcher = AssistantAgent(
        name="Researcher",
        system_message="""You are the opposition's fact-checker. You argue AGAINST the human—never for them.

CRITICAL: Only search for evidence that OPPOSES or UNDERMINES the human's claim. Never search for evidence that supports their position. If they support X, find evidence against X. Example: if human says "raise minimum wage", search "minimum wage job loss" or "negative effects minimum wage"—NOT "studies showing minimum wage helps".

Be concise. One brief factual summary only. Do not say "Acknowledged" or "Understood". Brief the team only. Do not speak to the human.""",
        llm_config={**llm_base, "config_list": make_config("researcher")},
        description="Finds facts opposing the human's claim.",
    )

    red_teamer = AssistantAgent(
        name="Red_Teamer",
        system_message="""You are the opposition's logic critic. Attack the human's argument. Identify fallacies, gaps, and weaknesses. Your job is to tear it down—never support their position.

Be concise. One brief analysis only. Do not say "Acknowledged" or "Understood". Brief the team only. Do not speak to the human.""",
        llm_config={**llm_base, "config_list": make_config("red_teamer")},
        description="Attacks logical weaknesses in the human's argument.",
    )

    speaker = AssistantAgent(
        name="Speaker",
        system_message="""You are the opposition debater. You are the ONLY agent who speaks to the human. Deliver a direct, persuasive rebuttal AGAINST their position. Use Researcher's facts and Red_Teamer's logic. Address the human firmly. One paragraph only. End with your rebuttal—do NOT ask "would you like to explore further" or offer to help.""",
        llm_config={**llm_base, "config_list": make_config("speaker")},
        description="Delivers the opposition's rebuttal to the human.",
    )

    if register_function:
        register_function(
            web_search,
            caller=researcher,
            executor=user_proxy,
            description="Search for evidence that OPPOSES or undermines the human's claim. Never search for evidence that supports their position.",
        )

    agents = {
        "user_proxy": user_proxy,
        "researcher": researcher,
        "red_teamer": red_teamer,
        "speaker": speaker,
        "make_config": make_config,
    }
    st.session_state.autogen_agents = agents
    return agents


def run_debate_round(transcribed_text: str) -> tuple[str, list[dict]]:
    """Run the AI team. Returns (final_rebuttal, brain_logs)."""
    from autogen import GroupChat, GroupChatManager

    agents = get_or_create_agents()
    make_config = agents["make_config"]

    user_proxy = agents["user_proxy"]
    researcher = agents["researcher"]
    red_teamer = agents["red_teamer"]
    speaker = agents["speaker"]

    groupchat = GroupChat(
        agents=[user_proxy, researcher, red_teamer, speaker],
        messages=[],
        max_round=6,  # Human→Researcher→Red_Teamer→Speaker; stop so human can rebut
        speaker_selection_method="round_robin",
        allowed_or_disallowed_speaker_transitions={
            user_proxy: [user_proxy],
            researcher: [user_proxy],
            red_teamer: [user_proxy],
            speaker: [user_proxy],
        },
        speaker_transitions_type="disallowed",
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config={"config_list": make_config("speaker"), "cache_seed": 42},
    )

    chat_result = agents["user_proxy"].initiate_chat(
        manager,
        message=f"[Human's argument]: {transcribed_text}",
        summary_method="last_msg",
    )

    # Extract brain logs (Researcher, Red_Teamer)
    brain_logs = []
    history = getattr(chat_result, "chat_history", None) or getattr(groupchat, "messages", [])
    for msg in history:
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if not content or not isinstance(content, str):
            continue
        name = msg.get("name", "") if isinstance(msg, dict) else getattr(msg, "name", "")
        role = msg.get("role", "")
        if name in ("Researcher", "Red_Teamer") or (role == "assistant" and name and name not in ("Speaker", "Human")):
            brain_logs.append({"agent": name or "System", "content": content[:2000]})

    # Extract final rebuttal (Speaker)
    final_message = (getattr(chat_result, "summary", None) or "").strip()
    if not final_message:
        for msg in reversed(history):
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            name = msg.get("name", "") if isinstance(msg, dict) else getattr(msg, "name", "")
            if name == "Speaker" and isinstance(content, str) and len(content) > 20:
                final_message = content
                break
            if isinstance(content, str) and len(content) > 50:
                final_message = content
                break

    return final_message or "No response generated.", brain_logs


# =============================================================================
# SESSION STATE
# =============================================================================

def init_session_state():
    if "public_chat" not in st.session_state:
        st.session_state.public_chat = []
    if "brain_logs" not in st.session_state:
        st.session_state.brain_logs = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "audio_input_key" not in st.session_state:
        st.session_state.audio_input_key = 0
    if "pending_tts" not in st.session_state:
        st.session_state.pending_tts = None  # (bytes, format) for playback after rerun


# =============================================================================
# STREAMLIT UI - Mission Control
# =============================================================================

def main():
    st.set_page_config(
        page_title="AI Debate Lab - Mission Control",
        page_icon="🗣️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("Set OPENROUTER_API_KEY in your .env file.")
        st.stop()

    init_session_state()

    st.title("🗣️ AI Debate Lab - Mission Control")
    st.caption("Record your argument. The AI team will fact-check, critique logic, and deliver a rebuttal.")

    # Layout: 70% Chat Transcript, 30% Brain Logs
    col_chat, col_brain = st.columns([7, 3])

    with col_chat:
        st.subheader("Chat Transcript")
        for msg in st.session_state.public_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        # Speaker TTS: play after rerun (autoplay often blocked; user can click play)
        if st.session_state.pending_tts:
            tts_bytes, fmt = st.session_state.pending_tts
            st.audio(tts_bytes, format=fmt, autoplay=True)

    with col_brain:
        st.subheader("🧠 Brain Logs")
        if st.session_state.brain_logs:
            for i, log in enumerate(reversed(st.session_state.brain_logs[-20:])):
                with st.expander(f"**{log['agent']}**", expanded=(i == 0)):
                    st.markdown(log["content"])
        else:
            st.info("Internal agent messages (Researcher, Red Teamer) will appear here.")

    # Bottom: Audio input (key changes after each run to prevent reprocessing same recording)
    st.divider()
    st.subheader("Record your rebuttal")
    audio = st.audio_input("🎤 Hold to record, release to send", key=f"audio_input_{st.session_state.audio_input_key}")

    if audio and not st.session_state.processing:
        st.session_state.processing = True
        st.session_state.pending_tts = None  # Clear previous TTS when processing new recording

        # Save audio buffer to temp file
        audio_bytes = audio.getbuffer().tobytes()
        # Streamlit audio_input typically returns WAV; handle webm if needed
        suffix = ".wav"
        if hasattr(audio, "name") and audio.name:
            ext = os.path.splitext(audio.name)[1]
            if ext.lower() in (".webm", ".ogg", ".mp3"):
                suffix = ext

        with st.spinner("Transcribing..."):
            transcribed = ""
            try:
                wav_path = _ensure_wav(audio_bytes, suffix)
                try:
                    transcribed = transcribe_audio(wav_path)
                except Exception:
                    transcribed = ""
                finally:
                    try:
                        os.unlink(wav_path)
                    except OSError:
                        pass
            except Exception as e:
                st.error(f"Audio processing failed: {e}")

        if not transcribed:
            st.warning("Could not understand audio. Please try again.")
            st.session_state.processing = False
            st.session_state.audio_input_key += 1
            st.rerun()

        # Add user message to public chat
        st.session_state.public_chat.append({"role": "user", "content": transcribed})

        rebuttal = ""
        with st.spinner("🧠 Team thinking..."):
            try:
                rebuttal, logs = run_debate_round(transcribed)
                st.session_state.brain_logs.extend(logs)
                st.session_state.public_chat.append({"role": "assistant", "content": rebuttal})
            except Exception as e:
                st.error(f"Debate round failed: {e}")
                rebuttal = f"Error: {e}"
                st.session_state.public_chat.append({"role": "assistant", "content": rebuttal})

        st.session_state.processing = False
        st.session_state.audio_input_key += 1  # Reset audio widget so same recording isn't reprocessed

        # TTS: save to session so it plays after rerun (rerun clears the page before audio loads)
        if rebuttal and not rebuttal.startswith("Error:"):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tts_path = f.name
            try:
                text_to_speech_async(rebuttal, tts_path, voice="en-US-ChristopherNeural")
                with open(tts_path, "rb") as f:
                    st.session_state.pending_tts = (f.read(), "audio/mp3")
            except Exception as tts_err:
                st.warning(f"TTS failed: {tts_err}")
            finally:
                try:
                    os.unlink(tts_path)
                except OSError:
                    pass

        st.rerun()


if __name__ == "__main__":
    main()
