import streamlit as st
import textwrap
import os
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain

# Load Hugging Face token
HUGGINGFACEHUB_API_TOKEN = os.getenv("HF_TOKEN")

# Load CSV for sentiment lookup
df = pd.read_csv("UserReviews.csv")

# Title
st.title("📱 App Review Analyzer: Developer Recommendations")

# Input fields
app_name = st.text_input("Enter the app name:")
review_text = st.text_area("Paste one user review of the app:")

# Show suggestion generation button only if both fields are filled
if app_name.strip() and review_text.strip():
    if st.button("Generate Developer Recommendation"):
        with st.spinner("Analyzing..."):

            # Attempt to find sentiment from CSV
            matched_row = df[(df["App"] == app_name.strip()) & (df["review"] == review_text.strip())]
            sentiment = matched_row["Sentiment"].values[0] if not matched_row.empty else "Unknown"

            # --- Step 1: Get App Category ---
            category_prompt = ChatPromptTemplate.from_template(
                "What is the most likely app category (e.g., cooking, gaming, beauty, education, sports, entertainment, health etc.) for the app called '{app_name}'?"
            )
            category_chain = LLMChain(
                prompt=category_prompt,
                llm=HuggingFaceEndpoint(
                    repo_id="HuggingFaceH4/zephyr-7b-beta",
                    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
                    task="text-generation",
                    temperature=0.3,
                    max_new_tokens=64,
                ),
            )
            category_output = category_chain.invoke({"app_name": app_name.strip()})
            app_category = category_output["text"].strip()

            # --- Step 2: Get Intent and Emotion ---
            intent_emotion_prompt = ChatPromptTemplate.from_template(
                "Analyze the following user review and identify:\n"
                "1. The main **intent** (e.g., complaint, praise, feature request).\n"
                "2. The **emotion** expressed (e.g., happiness, anger, frustration, discomfort, love, sad) in one line.\n\n"
                "Review:\n{review}\n\nRespond in the format:\nIntent: <intent>\nEmotion: <emotion>"
            )
            intent_emotion_chain = LLMChain(
                prompt=intent_emotion_prompt,
                llm=HuggingFaceEndpoint(
                    repo_id="HuggingFaceH4/zephyr-7b-beta",
                    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
                    task="text-generation",
                    temperature=0.3,
                    max_new_tokens=128,
                ),
            )
            intent_emotion_output = intent_emotion_chain.invoke({"review": review_text.strip()})["text"]

            # --- Step 3: Display extracted info ---
            st.subheader("📊 Review Metadata")
            st.markdown(f"**App Name:** {app_name}")
            st.markdown(f"**App Category:** {app_category}")
            st.markdown(f"**Sentiment:** {sentiment}")
            st.markdown(f"**Intent & Emotion:**\n{textwrap.fill(intent_emotion_output, width=100)}")

            # --- Step 4: Generate Developer Suggestions ---
            suggestion_prompt = ChatPromptTemplate.from_template(
                "You are an expert app developer. Based on the following user review for the app '{app_name}', "
                "provide actionable improvement suggestions in 3-4 lines.\n\nReview:\n{review}\n\nSuggestions:"
            )
            suggestion_chain = LLMChain(
                prompt=suggestion_prompt,
                llm=HuggingFaceEndpoint(
                    repo_id="HuggingFaceH4/zephyr-7b-beta",
                    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
                    task="text-generation",
                    temperature=0.5,
                    max_new_tokens=512,
                ),
            )
            suggestion_output = suggestion_chain.invoke({
                "app_name": app_name.strip(),
                "review": review_text.strip()
            })["text"]

            st.subheader("🔧 Suggestions for Developer")
            st.markdown(textwrap.fill(suggestion_output, width=100))
