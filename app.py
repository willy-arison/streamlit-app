import json
import os
import re

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI



def main():
    load_dotenv()


    def trim_repetition_loops(text: str, repeat_limit: int = 12) -> str:
        """Trim pathological repeated-word loops like 'data data data ...'."""
        pattern = re.compile(rf"\b(\w+)(?:\s+\1){{{repeat_limit},}}", flags=re.IGNORECASE)
        return pattern.sub(r"\1 [repetition trimmed]", text)

    st.set_page_config(
        page_title="Simple OpenAI Chatbot",
        page_icon="💬",
        layout="centered",
    )

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 820px;
            padding-top: 1.2rem;
            padding-bottom: 1rem;
        }
        .stApp {
            background: radial-gradient(circle at 10% 10%, #e6f5ff 0%, #f7fbff 35%, #ffffff 100%);
        }
        .chat-shell {
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid #dbe7f3;
            border-radius: 14px;
            padding: 8px 12px 12px;
            box-shadow: 0 10px 25px rgba(29, 78, 137, 0.08);
        }
        .title-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 6px;
        }
        .title-row h1 {
            margin: 0;
            font-size: 1.2rem;
            color: #12355b;
        }
        .subtitle {
            margin-top: 0;
            color: #2f4f6f;
            font-size: 0.88rem;
        }
        [data-testid="stChatMessageAvatar"] {
            display: none;
        }
        [data-testid="stChatMessageContent"] p,
        [data-testid="stChatMessageContent"] li {
            font-size: 0.93rem;
            line-height: 1.45;
        }
        [data-testid="stChatMessageContent"] {
            max-width: 700px;
        }
        [data-testid="stChatMessageContent"] > div {
            padding: 0.45rem 0.6rem;
            border-radius: 10px;
        }
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] input {
            font-size: 0.92rem !important;
        }
        [data-testid="stChatInput"] {
            margin-top: 0.35rem;
        }
        @media (max-width: 768px) {
            .block-container {
            padding-top: 0.75rem;
            padding-left: 0.65rem;
            padding-right: 0.65rem;
            }
            .chat-shell {
            border-radius: 12px;
            padding: 8px 10px 10px;
            }
            .title-row h1 {
            font-size: 1.05rem;
            }
            .subtitle {
            font-size: 0.82rem;
            margin-bottom: 0.25rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="title-row">
        <h1>OpenAI Chatbot</h1>
        </div>
        <p class="subtitle">Prompt, copy responses, and reset chat anytime.</p>
        """,
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! Ask me anything.",
            }
        ]

    with st.sidebar:
        st.header("Settings")
        model_options = [
            "gpt-5.4",
            "gpt-5.1",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5.1-chat-latest",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4o-mini",
        ]
        model = st.selectbox(
            "Model",
            model_options,
            index=1,
            help=(
                "Select which AI model to use. Some models are faster, some give better quality, "
                "and cost can be different."
            ),
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.2,
            value=0.3,
            step=0.1,
            help=(
                "Controls creativity. Low value gives safer and more focused answers. "
                "High value gives more creative answers, but can be less accurate."
            ),
        )
        system_prompt = st.text_area(
            "System prompt",
            value="You are a concise and helpful assistant.",
            height=90,
            help=(
                "Special instruction for the assistant. Use this to set tone, language, format, "
                "or rules for every reply."
            ),
        )
        with st.expander("Advanced model settings"):
            top_p = st.slider(
                "Top-p",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.05,
                help=(
                    "Another way to control randomness. Lower value means safer word choices. "
                    "Very high value (near 1.0) allows many word options, so replies can be more varied "
                    "but sometimes less clear."
                ),
            )
            max_tokens = st.number_input(
                "Max output tokens",
                min_value=64,
                max_value=4096,
                value=700,
                step=32,
                help=(
                    "Maximum length of the AI answer. Higher value allows longer answers, "
                    "but can be slower and use more tokens."
                ),
            )
            presence_penalty = st.slider(
                "Presence penalty",
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.1,
                help=(
                    "Encourages the model to bring new ideas or words. Higher value pushes more new topics. "
                    "Too high may make answers go off-topic."
                ),
            )
            frequency_penalty = st.slider(
                "Frequency penalty",
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.1,
                help=(
                    "Reduces repeating the same words or phrases. Higher value means less repetition. "
                    "Too high can make writing sound unnatural."
                ),
            )
            repetition_guard = st.checkbox(
                "Trim repeated loops",
                value=True,
                help=(
                    "If the model starts repeating one word many times (for example: data data data...), "
                    "the app will cut that part automatically."
                ),
            )
        if st.button("Start New Chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hi! Ask me anything.",
                }
            ]
            st.rerun()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY. Add it to your environment or .env file.")
        st.stop()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                safe_text = json.dumps(message["content"])
                components.html(
                    f"""
                    <div style="display:flex; justify-content:flex-end; margin: 2px 0 6px;">
                    <button
                        onclick='navigator.clipboard.writeText({safe_text}).then(() => this.innerText = "Copied ✓")'
                        style='border:none; background:#0f62a8; color:white; padding:6px 10px; border-radius:8px; cursor:pointer; font-size:12px;'>
                        Copy
                    </button>
                    </div>
                    """,
                    height=40,
                )

    if user_prompt := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                lc_messages = [
                    SystemMessage(
                        content=(
                            f"{system_prompt}\n"
                            "Do not repeat the same word or phrase many times. "
                            "Keep answers clear and concise."
                            "Write your answer using simple English but detailed. Suppose you are not in advanced in English."
                        )
                    ),
                ]
                for m in st.session_state.messages:
                    if m["role"] == "user":
                        lc_messages.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant":
                        lc_messages.append(AIMessage(content=m["content"]))

                llm = ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=int(max_tokens),
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    api_key=api_key,
                )
                response = llm.invoke(lc_messages).content
                if repetition_guard:
                    response = trim_repetition_loops(response)

            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)




def login():
    load_dotenv()
    saved_username = os.getenv("APP_USERNAME") or st.secrets.get("APP_USERNAME")
    saved_password = os.getenv("APP_PASSWORD") or st.secrets.get("APP_PASSWORD")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "login_error" not in st.session_state:
        st.session_state.login_error = ""

    if st.session_state.authenticated:
        return True

    st.title("Login")

    if st.session_state.login_error:
        st.error(st.session_state.login_error)

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if username == saved_username and password == saved_password:
            st.session_state.authenticated = True
            st.session_state.login_error = ""
            st.rerun()
        else:
            st.session_state.login_error = "Wrong username or password. Please try again."
            st.rerun()

    return False


if login():
    main()
else:
    st.stop()