import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.globals import set_verbose
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain_groq import ChatGroq

from prompt_service import get_patient_prompt, format_chat_history, CONDENSE_QUESTION_PROMPT, update_patient_info

set_verbose(True)

def create_conversational_chain(model, memory):
    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: format_chat_history(x["chat_history"])
        )
                            | CONDENSE_QUESTION_PROMPT
                            | model
                            | StrOutputParser(),
    )
    _context = {
        "question": lambda x: x["standalone_question"],
    }
    conversational_qa_chain = _inputs | _context | (
        lambda x: get_patient_prompt(x["question"])
    ) | model | StrOutputParser()
    return conversational_qa_chain

def clear_plot():
    hide_streamlit_style = """
                    <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    </style>
                    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def clear_git():
    # Add custom CSS to hide the GitHub icon
    hide_github_icon = """
    #GithubIcon {
      visibility: hidden;
    }
    """
    st.markdown(hide_github_icon, unsafe_allow_html=True)


def main():
    """
    This function is the main entry point of the application. It sets up the Groq client,
    the Streamlit interface, and handles the chat interaction.
    """
    st.set_page_config(layout='wide', )
    clear_plot()
    clear_git()

    load_dotenv()
    # Get Groq API key
    groq_api_key = os.getenv("GROQ_API_KEY")

    spacer, col = st.columns([5, 1])
    with col:
        st.image('img.png')

    # The title and greeting message of the Streamlit application
    st.title("Имитатор пациента")

    # Add customization options to the sidebar
    st.sidebar.title('Настройки')
    model = 'llama3-70b-8192'
    conversational_memory_length = st.sidebar.slider('Глубина памяти:', 1, 10, value=7)

    max_tokens = st.sidebar.slider(
        "Контекстное окно:",
        min_value=512,  # Minimum value
        max_value=8192,
        # Default value or max allowed if less
        value=8192,
        step=256,
        help="Настройте максимальное контекстное окно"
    )

    system_prompt = st.sidebar.text_area("Профиль пациента :",
                                         "Тебе 45 лет и тебя зовут Сидор Пеликанович."
                                         "Ты работаешь кочегаром в котельной городской больницы."
                                         "У тебя избыточный вес, ты пробовал разные диеты, "
                                         "но они не помогают его сбросить."
                                         "Ты любишь проводить время с семьей на природе и пикники."
                                         "У тебя есть две дочери 18 и 16 лет."
                                         "Ты не любишь заниматься спортом, но любишь смотреть "
                                         "футбол и пить пиво."
                                         "Сейчас ты пришел на первичный прием к врачу общей "
                                         "практики."
                                         "У тебя возникли проблемы со здоровьем."
                                         "У тебя появляется тошнота (особенно после обильного "
                                         "приема пищи), значительная потеря веса (10 кг за 6 "
                                         "недель) и хроническая усталость."
                                         "Мышечные судороги преимущественно в ногах, часто по "
                                         "ночам."
                                         "Умственная утомляемость с забывчивостью на работе."
                                         "Чувствуешь себя разбитым и утомленным в течение "
                                         "примерно 5–6 месяцев, с усилением симптомов в течение "
                                         "последних 4–8 недель."
                                         "Чувствуешь себя сильно ограниченным своим текущим "
                                         "состоянием."
                                         "Дополнительно у тебя возникают - множественные легкие "
                                         "инфекции в последнее время, эпизоды головокружения 1–2 "
                                         "раза в день,  сухость кожи, повышенная жажда (выпиваешь "
                                         "около 4–5 л воды в день) и частое мочеиспускание днем и "
                                         "ночью."
                                         "В твоей медицинской истории ранее было - гипертония,"
                                         "в настоящее время принимаешь лекарства от артериального "
                                         "давления"
                                         "(Гигротон 50 мг и рамиприл 5 мг),"
                                         "отдышка при нагрузке,  ожирение печени диагностировано "
                                         "3 года назад,"
                                         "правосторонняя паховая грыжа, пролеченная хирургически "
                                         "3 года назад,"
                                         "бываю легкие запоры,  аллергия на пенициллин с детства."
                                         "Ранее курил 4 года, когда ему было двадцать с небольшим."
                                         "Время от времени употребляет пиво (1-2 раза в неделю)."
                                         "Твой отец умер от сердечного приступа."
                                         "Мать умерла в 79 лет от диабета."
                                         "У брата диагностирован рак толстой кишки."
                                         "Отвечай на вопросы доктора, он хочет поставить тебе "
                                         "диагноз.",
                                         height=640,
                                         )



    update_patient_info(system_prompt)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length,
                                            memory_key="chat_history", return_messages=True)

    # Initialize chat history if not present
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Chat history container
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(message['human'])
            with st.chat_message("assistant"):
                st.markdown(message['AI'])

    user_question = st.text_input("Доктор, я так болен ... ")

    # Initialize Groq Langchain chat object
    groq_chat = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,
        model_name=model
    )

    # Create the conversation chain
    conversation_chain = create_conversational_chain(groq_chat, memory)

    # If the user has asked a question,
    if user_question:
        # Construct a prompt with the correct input format
        input_data = {
            "chat_history": st.session_state.chat_history,
            "human_input": user_question
        }

        # Generate response
        response = conversation_chain.invoke(input_data)
        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message)

        # Update chat history container with new messages
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_question)
            with st.chat_message("assistant"):
                st.markdown(response)


if __name__ == "__main__":
    main()
