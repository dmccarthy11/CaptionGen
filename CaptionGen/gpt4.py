from openai import OpenAI
import streamlit as st

# Constants for different styles for the prompt
FRAME = "Can you rewrite this caption to make it more"
INSTAGRAM = "Instagram"
TWITTER = "Twitter"
FACEBOOK = "Facebook"

ENGAGING = "fun and engaging"
PROFESSIONAL = "professional"
FUNNY = "funny"

# Initialize client with API key
client = OpenAI(
  api_key=st.secrets["openai_api_key"]
)

def generate_caption(prompt, temp=0.7, top_p=0.9, max_output_tokens=50, style=ENGAGING, platform=INSTAGRAM):
    # Setup the prompt to frame the output
    question = FRAME

    # Add in the style 
    if style == ENGAGING:
        question = question + ' ' + ENGAGING
    elif style == PROFESSIONAL:
        question = question + ' ' + PROFESSIONAL
    elif style == FUNNY:
        question = question + ' ' + FUNNY
    else:
        question = question + ' ' + ENGAGING

    # Add in the platform
    if platform == INSTAGRAM:
        question = question + ' for ' + INSTAGRAM + '?'
    elif platform == TWITTER:
        question = question + ' for ' + TWITTER + '?'
    elif platform == FACEBOOK:
        question = question + ' for ' + FACEBOOK + '?'
    else:
        question = question + ' for ' + INSTAGRAM + '?'

    try:
        # Make the API call to generate a caption
        response = client.responses.create(
            model="gpt-4o",  # Specify the model you want to use
            input=f"{question}\n{prompt}",
            temperature=temp,
            top_p=top_p,
            max_output_tokens=max_output_tokens,  # Specify the maximum number of tokens for the generated response
        )

        # Extract the improved caption
        improved_caption = response.output_text

        return improved_caption
    except Exception as e:
        print(f"Error generating caption: {e}")
        return None
