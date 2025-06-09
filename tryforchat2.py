import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from openai import OpenAI

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

try:
    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)
except Exception as e:
    st.error(f"Error loading class indices: {e}")
    st.stop()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0
        return img_array
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    if preprocessed_img is not None:
        predictions = model.predict(preprocessed_img)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown")
        return predicted_class_name
    return None

def get_remedies_from_llm(classification_output):
    # Extract plant name and disease/health status
    plant_name, condition = classification_output.split("___")
    
    if condition.lower() == "healthy":
        custom_instructions = (
            f"You are an expert in plant care. The user uploaded an image of a healthy {plant_name}. "
            f"Congratulate them for maintaining a healthy plant! Provide care tips including proper watering, "
            f"sunlight needs, fertilization, and pest prevention."
        )
    else:
        custom_instructions = (
            f"You are an expert in plant disease management. The user uploaded an image of a {plant_name} with {condition}. "
            f"Provide clear, effective remedies, including organic and chemical solutions, preventive measures, "
            f"and best practices to restore plant health."
        )

    try:
        response = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": custom_instructions}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=4096
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error querying the LLM: {e}")
        return "Failed to generate remedies. Please try again."

def chat_with_llm(user_query, classification_output, disease_context, chat_history):
    # Extract plant name and condition
    plant_name, condition = classification_output.split("___")
    
    if condition.lower() == "healthy":
        prompt = (
            f"You are a plant care expert. The user uploaded an image of a healthy {plant_name}. "
            f"Congratulate them on maintaining a healthy plant! Offer guidance on how to keep it in good condition, "
            f"including watering, sunlight, pest prevention, and general care. Encourage them to ask any plant care questions.\n\n"
            f"User's question: {user_query}"
        )
    else:
        prompt = (
            f"You are an expert in plant disease management. The user uploaded an image of a {plant_name} diagnosed with {condition}. "
            f"Here is the initial disease information you provided:\n\n{disease_context}\n\n"
            f"Now the user has a follow-up question. Answer based on your expertise and the context above. "
            f"Be precise, actionable, and helpful.\n\n"
            f"User's question: {user_query}"
        )

    messages = [{"role": "system", "content": "You are a helpful plant disease expert assistant."}]

    # Add chat history
    for message in chat_history:
        messages.append({"role": message["role"], "content": message["content"]})

    # Add current question
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=messages,
            temperature=0.2,
            top_p=0.7,
            max_tokens=4096
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error querying the LLM: {e}")
        return "Sorry, I couldn't process your request. Please try again."


st.set_page_config(page_title="Plant Disease Classifier", layout="wide")

# Initialize session state variables
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "llm_result" not in st.session_state:
    st.session_state.llm_result = ""
if "predicted_disease" not in st.session_state:
    st.session_state.predicted_disease = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "show_spinner" not in st.session_state:
    st.session_state.show_spinner = False

# Custom CSS for chat bubbles
# Custom CSS for chat bubbles
st.markdown("""
<style>
    .chat-container {
        margin-bottom: 20px;
    }
    
    .chat-message {
        padding: 1.5rem; 
        border-radius: 10px;
        margin-bottom: 10px;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        overflow-wrap: break-word;
    }
    .chat-message.user {
        background-color: #2C7DAC;  /* Darker blue for better contrast */
        margin-left: 20%;
        border-bottom-right-radius: 2px;
        color: white;
    }
    .chat-message.bot {
        background-color: #2E8B57;  /* Darker green for better contrast */
        margin-right: 20%;
        border-bottom-left-radius: 2px;
        color: white;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        font-size: 20px;
        background-color: white;
        color: #333;
    }
    .chat-message .content {
        display: flex;
        flex-direction: column;
        flex: 1;
    }
    .chat-message .content p {
        margin: 0;
    }
    .message-text p {
        margin-bottom: 8px;
    }
    .message-text p:last-child {
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Plant_health_logo.svg/200px-Plant_health_logo.svg.png", width=120)
    st.header("📌 Instructions")
    st.markdown("""
    1️⃣ **Upload an Image**: Choose an image of a diseased plant.  
    2️⃣ **Classify & Get Remedies**: Click the button to analyze the disease and get suggestions.  
    3️⃣ **View Results**: The detected disease and remedies will be displayed in the first tab.

    4️⃣ **Chat**: Switch to the chat tab to ask follow-up questions about the disease.
    """)
    st.divider()
    st.info("🔍 Ensure the image is clear and focuses on affected areas.")

    uploaded_image = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"], key="file_uploader")

    if st.button("🔄 Reset"):
        st.session_state.uploaded_image = None
        st.session_state.llm_result = ""
        st.session_state.predicted_disease = None
        st.session_state.chat_history = []
        st.session_state.prediction_done = False
        st.session_state.show_spinner = False
        st.experimental_rerun()

if uploaded_image and not st.session_state.uploaded_image:
    st.session_state.uploaded_image = uploaded_image
    # Reset chat when new image is uploaded
    st.session_state.chat_history = []
    st.session_state.prediction_done = False
    st.session_state.llm_result = ""

# Main content area
if st.session_state.uploaded_image:
    st.markdown(f"<p style='font-size: 14px; color: #666;'>📂 Selected File: {uploaded_image.name if uploaded_image else st.session_state.uploaded_image.name}</p>", unsafe_allow_html=True)

    # Image display section
    col1, col2 = st.columns([1, 3])
    with col1:
        resized_img = Image.open(st.session_state.uploaded_image).resize((150, 150))
        st.image(resized_img, caption="Uploaded Image", width=150)
    
    with col2:
        # Only show button if prediction hasn't been done
        if not st.session_state.prediction_done:
            if st.button("🔍 Classify and Get Remedies", key="classify_button"):
                st.session_state.show_spinner = True
                
                # Classify the image
                prediction = predict_image_class(model, st.session_state.uploaded_image, class_indices)
                if prediction:
                    st.session_state.predicted_disease = prediction
                    
                    # Get remedies with a spinner
                    with st.spinner("Fetching remedies and suggestions..."):
                        st.session_state.llm_result = get_remedies_from_llm(prediction)
                    
                    # Initialize chat history with the system's first response
                    st.session_state.chat_history = [
                        {"role": "assistant", "content": st.session_state.llm_result}
                    ]
                    
                    # Mark prediction as done
                    st.session_state.prediction_done = True
                    st.session_state.show_spinner = False
                    st.experimental_rerun()
                else:
                    st.error("Failed to make a prediction.")
                    st.session_state.show_spinner = False
    
    # Display spinner if waiting for results
    if st.session_state.show_spinner:
        with st.spinner("Processing..."):
            st.empty()
    
    # Display tabs after prediction is done
    if st.session_state.prediction_done:
        tab1, tab2 = st.tabs(["Disease Information", "Chat with Expert"])
        
        with tab1:
            # Show disease information and remedies
            st.success(f'**Prediction:** {st.session_state.predicted_disease}')
            st.markdown("### 🌿 Remedies & Suggestions:")
            st.markdown(st.session_state.llm_result, unsafe_allow_html=True)
        
        with tab2:
            # Chat interface with improved styling
            st.subheader(f"💬 Chat about {st.session_state.predicted_disease}")
            
            # Display chat history with enhanced UI
            chat_container = st.container()
            with chat_container:
                for i, message in enumerate(st.session_state.chat_history):
                    # Skip the first assistant message which contains the initial remedies info
                    if i == 0 and message["role"] == "assistant":
                        continue
                        
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-message user">
                            <div class="avatar">👤</div>
                            <div class="content">
                                <div class="message-text">{message['content']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message bot">
                            <div class="avatar">🤖</div>
                            <div class="content">
                                <div class="message-text">{message['content']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                                
            # User input for chat
            with st.form(key="chat_form", clear_on_submit=True):
                user_query = st.text_input("Ask a question about the disease or treatment:", key="user_query")
                submit_button = st.form_submit_button("Send")
                
                if submit_button and user_query:
                    # Add user query to chat history
                    st.session_state.chat_history.append({"role": "user", "content": user_query})
                    
                    # Get response from LLM
                    with st.spinner("Getting expert response..."):
                        response = chat_with_llm(
                            user_query, 
                            st.session_state.predicted_disease, 
                            st.session_state.llm_result,
                            st.session_state.chat_history[:-1]  # Exclude the just-added user message
                        )
                        
                        # Add response to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    st.experimental_rerun()
else:
    # Welcome screen when no image is uploaded
    st.markdown("""
    # 🌱 Plant Disease Classifier & Advisor
    
    Welcome to the Plant Disease Classifier and Advisor! This application helps you:
    
    * Identify plant diseases from images
    * Get expert remedies and treatment advice
    * Chat with an AI plant disease expert for follow-up questions
    
    To get started, upload an image of your diseased plant using the sidebar on the left.
    """)