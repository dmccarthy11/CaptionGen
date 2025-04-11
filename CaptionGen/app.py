from PIL import Image
import streamlit as st
from gpt4 import generate_caption, INSTAGRAM, TWITTER, FACEBOOK, ENGAGING, PROFESSIONAL, FUNNY
from blip import load_blip, greedy_caption
from clip import load_clip
from gpt2 import load_gpt2, generate_gpt2_caption
from oneshot import oneshot_captions
from utils import clean_caption

st.set_page_config(page_title="CaptionGen App", layout="centered")

# Streamlit UI
with st.sidebar:
    st.title("📸 CaptionGen")

    st.subheader('Style and parameters')

    # Model
    model = st.sidebar.selectbox('Choose a Model', ['GPT-4o', 'GPT-2'], key='model')

    # Platform
    platform = st.sidebar.selectbox('Choose a Platform', ['Instagram', 'X (Twitter)', 'Facebook'], key='platform')
    if platform == 'Instagram':
        selected_platform = INSTAGRAM
    elif platform == 'X (Twitter)':
        selected_platform = TWITTER
    elif platform == 'Facebook':
        selected_platform = FACEBOOK

    # Style
    style = st.sidebar.selectbox('Choose a Style', ['Engaging', 'Professional', 'Funny'], key='style')
    if style == 'Engaging':
        selected_style = ENGAGING
    elif style == 'Professional':
        selected_style = PROFESSIONAL
    elif style == 'Funny':
        selected_style = FUNNY
    
    # Set parameter components
    with st.expander('Parameters'):
        temperature = st.slider('temperature', min_value=0.01, max_value=2.0, value=0.7, step=0.01)
        top_p = st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        max_output_tokens = st.slider('max_output_tokens', min_value=32, max_value=96, value=50, step=8)

    enable_clip = st.checkbox('Enable one-shot captioning')
    exclude_emojis = st.checkbox('Include emojis', value=True)
    exclude_hashtags = st.checkbox('Include hashtags', value=True)

blip_model, blip_processor = load_blip()
clip_model, clip_processor = load_clip()
gpt2_model, gpt2_tokenizer = load_gpt2()

# Set Caption components in main page
st.write("Upload an image, and let CaptionGen generate a caption for it!")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image:
    # Check if the image has changed
    if "last_uploaded_image" not in st.session_state or uploaded_image != st.session_state.last_uploaded_image:
        st.session_state.clear()
        st.session_state.last_uploaded_image = uploaded_image

    # Try to convert file to PIL image
    try:
        st.session_state.image = Image.open(uploaded_image)
        st.image(st.session_state.image)

        # Generate short caption if one hasn't been generated already
        if st.button("Generate Caption 🔄"):
            st.write("⏳ Generating caption...")
            if enable_clip:
                st.session_state.short_caption = oneshot_captions(blip_model, blip_processor, clip_model, clip_processor, st.session_state.image)
            else:
                st.session_state.short_caption = greedy_caption(blip_model, blip_processor, st.session_state.image)

            # Get enhanced caption
            if model == 'GPT-4o':
                st.session_state.caption = generate_caption(
                    prompt=st.session_state.short_caption,
                    temp=temperature, 
                    top_p=top_p, 
                    max_output_tokens=max_output_tokens,
                    platform=selected_platform,
                    style=selected_style
                )
            elif model == 'GPT-2':
                # st.session_state.caption == generate_gpt2_caption(
                #     model=gpt2_model, 
                #     tokenizer=gpt2_tokenizer,
                #     short_caption=st.session_state.short_caption
                # )
                caption = generate_gpt2_caption(gpt2_model, gpt2_tokenizer, st.session_state.short_caption)
                st.session_state.caption = caption

            st.session_state.cleaned_caption = clean_caption(st.session_state.caption, emojis=exclude_emojis, hashtags=exclude_hashtags)
            
            # Display result
            st.success("✅ Caption Generated!")
            st.write(f"**Caption:** {st.session_state.cleaned_caption}")
    
    # Catch wrong file type or error uploading image
    except:
        st.error("Sorry, we couldn't read that image file. Please upload a valid PNG or JPEG image.")

