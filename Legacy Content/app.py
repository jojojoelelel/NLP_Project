import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set page config for better mobile experience
st.set_page_config(
    page_title="Gen-Z Translates",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better mobile responsiveness
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stTextInput > div > div > input {
        font-size: 16px !important;  /* Larger font for mobile */
    }
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 16px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    @media (max-width: 768px) {
        .main {
            padding: 0.5rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Gen-Z Translates")
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem;'>Translate GenZ slang to Dictionary English</p>
    </div>
""", unsafe_allow_html=True)

# Initialize session state for model loading and translation
if 'model' not in st.session_state:
    with st.spinner('Loading translation model...'):
        # Load the model and tokenizer
        model_path = "google/flan-t5-base"  # You can replace this with your fine-tuned model path
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        # Store in session state
        st.session_state.tokenizer = tokenizer
        st.session_state.model = model
        st.session_state.translated_text = ""

# Create two columns for input and output
col1, col2 = st.columns(2)

with col1:
    st.subheader("GEN-Z")
    user_input = st.text_area(
        "Enter GenZ text here:",
        placeholder="Type your GenZ slang here...",
        height=150
    )
    
    # Add translate button
    translate_button = st.button("Translate", type="primary")

with col2:
    st.subheader("Dictionary English")
    if translate_button and user_input:
        with st.spinner('Translating...'):
            # Prepare the input
            prompt = f"Paraphrase this Gen-Z sentence into formal English:\nExample: {user_input}"
            
            # Tokenize and generate
            inputs = st.session_state.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
            outputs = st.session_state.model.generate(
                inputs["input_ids"],
                max_length=60,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode the output and store in session state
            st.session_state.translated_text = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Display the translation (either from session state or placeholder)
    translation_text = st.session_state.translated_text if st.session_state.translated_text else "Translation will appear here..."
    st.text_area("Translation:", value=translation_text, height=150, disabled=True)

# Add some helpful information
st.markdown("""
    <div style='text-align: center; margin-top: 2rem;'>
        <p style='font-size: 0.9rem; color: #666;'>
            Enter GenZ slang in the left box and click the Translate button to get the Dictionary English translation.
        </p>
    </div>
""", unsafe_allow_html=True) 