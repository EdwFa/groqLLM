import streamlit as st
import os
from groq import Groq
import random

from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os


def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """
    # Get Groq API key
    # name = "GROQ_API_KEY"
    # os.environ.get(name)
    # groq_api_key = os.environ.get(name)
    # print(groq_api_key)
    groq_api_key = st.secrets["GROQ_API_KEY"]
    #groq_api_key = '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'
    # Display the Groq logo
    spacer, col = st.columns([5, 1])  
    with col:  
        st.image('img.png')

    # The title and greeting message of the Streamlit application
    st.title("Имитатор пациента")

    # Add customization options to the sidebar
    st.sidebar.title('Настройки')
    # model = st.sidebar.selectbox(
    #     'Модель',
    #     ['llama3-70b-8192','llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    # )
    model = 'llama3-70b-8192'
    conversational_memory_length = st.sidebar.slider('Глубина памяти:', 1, 10, value = 7)

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
"У тебя избыточный вес, ты пробовал разные диеты, но они не помогают его сбросить."
"Ты любишь проводить время с семьей на природе и пикники."
"У тебя есть две дочери 18 и 16 лет."
"Ты не любишь заниматься спортом, но любишь смотреть футбол и пить пиво." 
"Сейчас ты пришел на первичный прием к врачу общей практики."
"У тебя возникли проблемы со здоровьем." 
"У тебя появляется тошнота (особенно после обильного приема пищи), значительная потеря веса (10 кг за 6 недель) и хроническая усталость." 
"Мышечные судороги преимущественно в ногах, часто по ночам." 
"Умственная утомляемость с забывчивостью на работе." 
"Чувствуешь себя разбитым и утомленным в течение примерно 5–6 месяцев, с усилением симптомов в течение последних 4–8 недель." 
"Чувствуешь себя сильно ограниченным своим текущим состоянием."
"Дополнительно у тебя возникают - множественные легкие инфекции в последнее время, эпизоды головокружения 1–2 раза в день,  сухость кожи, повышенная жажда (выпиваешь около 4–5 л воды в день) и частое мочеиспускание днем и ночью."
"В твоей медицинской истории ранее было - гипертония,"
"в настоящее время принимаешь лекарства от артериального давления"
"(Гигротон 50 мг и рамиприл 5 мг),"
"отдышка при нагрузке,  ожирение печени диагностировано 3 года назад,"
"правосторонняя паховая грыжа, пролеченная хирургически 3 года назад,"
"бываю легкие запоры,  аллергия на пенициллин с детства."
"Ранее курил 4 года, когда ему было двадцать с небольшим." 
"Время от времени употребляет пиво (1-2 раза в неделю)."
"Твой отец умер от сердечного приступа."
"Мать умерла в 79 лет от диабета." 
"У брата диагностирован рак толстой кишки."
"Отвечай на вопросы доктора, он хочет поставить тебе диагноз.",
                                         height=720,
)


    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    user_question = st.text_input("Доктор, я так болен ... ")

    # session state variable
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]
    else:
        for message in st.session_state.chat_history:
            memory.save_context(
                {'input':message['human']},
                {'output':message['AI']}
                )


    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=model
    )


    # If the user has asked a question,
    if user_question:

        # Construct a chat prompt template using various components
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=system_prompt
                ),  # This is the persistent system prompt that is always included at the start of the chat.

                MessagesPlaceholder(
                    variable_name="chat_history"
                ),  # This placeholder will be replaced by the actual chat history during the conversation. It helps in maintaining context.

                HumanMessagePromptTemplate.from_template(
                    "{human_input}"
                ),  # This template is where the user's current input will be injected into the prompt.
            ]
        )

        # Create a conversation chain using the LangChain LLM (Language Learning Model)
        conversation = LLMChain(
            llm=groq_chat,  # The Groq LangChain chat object initialized earlier.
            prompt=prompt,  # The constructed prompt template.
            verbose=True,   # Enables verbose output, which can be useful for debugging.
            memory=memory,  # The conversational memory object that stores and manages the conversation history.
        )
        
        # The chatbot's answer is generated by sending the full prompt to the Groq API.
        response = conversation.predict(human_input=user_question)
        message = {'human':user_question,'AI':response}
        st.session_state.chat_history.append(message)
        st.write("Пациент :", response)

if __name__ == "__main__":
    main()





