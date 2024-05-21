import streamlit as st
from typing import Generator
from groq import Groq

st.set_page_config(page_icon="💬", layout="wide",
                   page_title="Groq Goes Brrrrrrrr...")


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


#icon("🏎️")

st.subheader("Sechenov.DataMed - Quality assessor for LLM models", divider="rainbow", anchor=False)

client = Groq(
    #api_key=st.secrets["GROQ_API_KEY"],
    api_key="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"
)

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Define model details
models = {
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
#    "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
}

# Add customization options to the sidebar
st.sidebar.title('Настройки')
system_prompt = st.sidebar.text_input("Промт:")
model_option = st.sidebar.selectbox(
    "Модель:",
    options=list(models.keys()),
    format_func=lambda x: models[x]["name"],
    index=1  # Default to llama3-70B
)
# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

max_tokens_range = models[model_option]["tokens"]


# Adjust max_tokens slider dynamically based on the selected model
max_tokens = st.sidebar.slider(
    "Максимальное кол-во токенов:",
    min_value=512,  # Minimum value to allow some flexibility
    max_value=max_tokens_range,
    # Default value or max allowed if less
    value=min(32768, max_tokens_range),
    step=512,
    help=f"Настройте максимальное количество токенов для ответа модели. Для выбранной модели: {max_tokens_range}"
)


# Layout for model selection and max_tokens slider
# col1, col2 = st.columns(2)

# with col1:
#     model_option = st.selectbox(
#         "Choose a model:",
#         options=list(models.keys()),
#         format_func=lambda x: models[x]["name"],
#         index=2  # Default to llama3-70B
#     )



# with col2:
#     # Adjust max_tokens slider dynamically based on the selected model
#     max_tokens = st.slider(
#         "Max Tokens:",
#         min_value=512,  # Minimum value to allow some flexibility
#         max_value=max_tokens_range,
#         # Default value or max allowed if less
#         value=min(32768, max_tokens_range),
#         step=512,
#         help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
#     )

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
         #           "prompt": "Представь себе что ты больной пациент на приеме у врача. У тебя высокая температура и болит горло. Отвечай на вопросы.",
                    "content": m["content"]
                }
                for m in st.session_state.messages
            ],
            max_tokens=max_tokens,
            stream=True
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