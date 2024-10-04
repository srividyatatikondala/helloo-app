import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import cohere
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

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

# Create the LangChain instance with the custom wrapper
chain = CohereChain(co, prompt)

# Streamlit app
st.title("Agricultural Commodity Forecasting and Chatbot Integration")

# File uploader for CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
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

        st.write("## Chat with the Assistant")
        question = st.text_input("Ask the chatbot a question", "What would you like to know?")

        if st.button("Get Answer"):
            if question:
                response = chain.run(context=context, question=question)
                st.write(f"Chatbot's Answer: {response}")
            else:
                st.error("Please ask a question.")
else:
    st.warning("Please upload a CSV file to get started.")
