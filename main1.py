import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import cohere
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import pyttsx3  # For text-to-speech
import speech_recognition as sr  # For voice recognition
import pyaudio  # Required for PyAudio

# Initialize pyttsx3 engine for TTS
engine = pyttsx3.init()

# Function to convert text to speech
def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Function to capture voice input
def get_voice_input():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        st.write("Please speak your question...")
        audio = recognizer.record(source, duration=10)  # Record for 10 seconds
        st.write("Processing...")
        try:
            question = recognizer.recognize_google(audio)
            st.success(f"You asked: {question}")
            return question
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio. Please try again.")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
            return None


# Initialize Cohere API
cohere_api_key = 'IDlfxdy11paDxht8zQKuLQZkR61dhaPRTwdNzkpF'
co = cohere.Client(cohere_api_key)

# Define the prompt template
template = """
You are a helpful assistant. Please answer the question based on the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Create a wrapper function for LangChain to use with Cohere
class CohereChain:
    def __init__(self, cohere_client, prompt_template):
        self.cohere_client = cohere_client
        self.prompt_template = prompt_template

    def run(self, context, question):
        prompt_text = self.prompt_template.format(context=context, question=question)
        return generate_text_with_cohere(prompt_text, self.cohere_client)

# Define a custom function to interact with Cohere
def generate_text_with_cohere(prompt, cohere_client):
    response = cohere_client.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=100,
        temperature=0.75,
        k=0,
        stop_sequences=[]
    )
    return response.generations[0].text

# Create the LangChain instance with the custom wrapper
chain = CohereChain(co, prompt)

# Streamlit app
st.title("Agricultural Commodity Forecasting and Chatbot Integration")

@st.cache_data
def load_data():
    df = pd.read_csv('/home/srividya/Downloads/updated_daily_price - updated_daily_price.csv')
    return df

data = load_data()

st.write("## Agricultural Commodity Prices")
st.dataframe(data)

specific_commodities = data['Commodity'].unique().tolist()

commodity = st.selectbox("Select a commodity to forecast", specific_commodities)

commodity_data = data[data['Commodity'] == commodity]

if commodity_data.empty:
    st.error(f"No valid data available for {commodity}. Please select a different commodity.")
else:
    target_column = 'Modal Price'
    commodity_data = commodity_data.dropna(subset=[target_column, 'Forecasted Price (INR/kg)'])
    
    if commodity_data.empty:
        st.error(f"No valid data available for {commodity} after cleaning. Please select a different commodity.")
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(commodity_data[target_column].values.reshape(-1, 1))

        order = st.selectbox("Select ARIMA order (p, d, q)", [(1, 1, 1), (2, 1, 2), (1, 0, 1)])
        model = ARIMA(data_scaled, order=order)
        model_fit = model.fit()

        steps = st.slider("Number of steps to forecast into the future", 1, 100, 10)
        forecast = model_fit.forecast(steps=steps)
        forecast = scaler.inverse_transform(forecast.reshape(-1, 1))  # Inverse transform to original scale

        original_prices = commodity_data[target_column].values
        forecast_index = range(len(commodity_data), len(commodity_data) + steps)
        forecast_series = pd.Series(forecast.flatten(), index=forecast_index)

        st.write(f"## {commodity} - Original Prices")
        fig, ax = plt.subplots()
        ax.plot(commodity_data.index, original_prices, label='Original Prices', color='blue')
        ax.set_xlabel("Index")
        ax.set_ylabel(f"Price (INR/kg)")
        ax.set_title(f'{commodity} - Original Prices')
        ax.legend()
        st.pyplot(fig)

        st.write(f"## {commodity} - Forecasted Prices")
        fig, ax = plt.subplots()
        ax.plot(forecast_series.index, forecast_series, label='Forecasted Prices', color='red', linestyle='--')
        ax.set_xlabel("Index")
        ax.set_ylabel(f"Price (INR/kg)")
        ax.set_title(f'{commodity} - Forecasted Prices')
        ax.legend()
        st.pyplot(fig)

        st.write(f"## Forecasted Prices for Next {steps} Steps")
        st.write(forecast)

    # Enhanced chatbot context
    context = f"""
    The dataset includes agricultural commodity prices with the following columns:
    - Commodity: The type of commodity (e.g., onion, potato)
    - Modal Price: The price of the commodity (INR/kg)
    - Forecasted Price (INR/kg): The predicted price for future periods

    Current data context:
    {data.head()}

    Forecasting details:
    For the selected commodity, the ARIMA model forecasts future prices. The parameters of the model and the forecasted results are displayed in the app.

    For any questions regarding specific commodities or forecasted prices, please ask below.
    """

    # Streamlit app
st.title("Agricultural Commodity Forecasting and Chatbot Integration")

# ... [rest of your existing code]

st.write("## Chat with the Assistant")

# Add a button to allow voice input
if st.button("Ask by Voice"):
    question = get_voice_input()  # Get voice input
    st.session_state.question = question  # Store the question in session state
else:
    question = st.session_state.get("question", "")  # Retrieve the voice question if available
    question = st.text_input("Ask the chatbot a question", question)

if st.button("Get Answer"):
    if question:
        response = chain.run(context=context, question=question)
        st.write(f"Chatbot's Answer: {response}")

        # Convert the response to speech
        text_to_speech(response)
    else:
        st.error("Please ask a question.")

# Load images where necessary
# Uncomment and provide the path to your images
# st.image("path/to/image1.png", caption="Image 1 caption")
# st.image("path/to/image2.png", caption="Image 2 caption")
