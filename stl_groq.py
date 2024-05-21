import streamlit as st
from typing import Generator
from groq import Groq

st.set_page_config(page_icon="💬", layout="wide",
                   page_title="Грокаем гроком LLM")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def icon(emoji: str):
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )
#icon("🏎️")
st.subheader("Sechenov.DataMed - Quality assessor for LLM models", divider="rainbow", anchor=False)

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Define model details
models = {
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
}

# Add customization options to the sidebar
st.sidebar.title('Параметры')
#system_prompt = st.sidebar.text_input("Промт:")
model_option = st.sidebar.selectbox(
    "Модель:",
    options=list(models.keys()),
    format_func=lambda x: models[x]["name"],
    index=1  # Default to llama3-70B
)
# temper = st.sidebar.number_input(
#     "Установите температуру модели",
#     min_value=0,
#     max_value=2, value=0.5, step=0.01,
# )

# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

max_tokens_range = models[model_option]["tokens"]

# Adjust max_tokens slider dynamically based on the selected model
max_tokens = st.sidebar.slider(
    "Максимальное кол-во токенов:",
    min_value=512,  # Minimum value
    max_value=max_tokens_range,
    # Default value or max allowed if less
    value=min(32768, max_tokens_range),
    step=256,
    help=f"Настройте максимальное количество токенов для ответа модели. Для выбранной модели: {max_tokens_range}"
)


# temper = st.sidebar.number_input(
#     "Установите температуру модели",
#     min_value=0,
#     max_value=2, value=0.5, step=0.01,
# )
# st.sidebar.write("Temperature - ", temper)
# temper = st.sidebar.slider(
#     "Температура:",
#     min_value=0,  # Minimum value
#     max_value=1,
#     # Default value or max allowed if less
#     value=0.5,
#     step=0.01,
#     help=f"Настройте максимальное количество токенов для ответа модели. Для выбранной модели: {max_tokens_range}"
# )
#
#
# top_P = st.sidebar.slider(
#     label = "Тop P :",
#     min_value=0,  # Minimum value
#     max_value=1,
#     # Default value or max allowed if less
#     value=1,
#     step=0.01,
#     help="Настройте параметр Top_P (по умолчанию=1)"
# )


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

if prompt := st.chat_input("Задавайте вопрос ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)

    # Fetch response from Groq API
    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[
                {
                    "role": m["role"],
                    "content": m["content"]
                }
                for m in st.session_state.messages
            ],
            max_tokens=max_tokens,
            # temperature = 0.5,
            # top_p = 1,
            stream=True,
            stop = None
        )

        # Use the generator function with st.write_stream
        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
    except Exception as e:
        st.error(e, icon="🚨")

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
    else:
        # Handle the case where full_response is not a string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})