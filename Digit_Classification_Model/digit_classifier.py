import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import pickle
import time
import matplotlib.pyplot as plt
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Load the pre-trained model
@st.cache_resource
def load_model(model_file):
    try:
        model = pickle.load(model_file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess the uploaded image
def preprocess_image(image):
    try:
        # Convert to grayscale
        img = image.convert('L')
        # Invert image colors to match MNIST style
        img = ImageOps.invert(img)
        # Resize to 28x28
        img = img.resize((28, 28), Image.LANCZOS)
        img = np.array(img)
        # Normalize pixel values
        img = img / 255.0
        # Reshape for model input
        img = img.reshape(1, 28, 28, 1)
        return img
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Visualization function
def plot_probabilities(probs):
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(10), probs, color='skyblue')
    ax.set_xticks(range(10))
    ax.set_ylim(0, 1)
    ax.set_title('Prediction Probabilities', pad=20)
    ax.set_xlabel('Digits')
    ax.set_ylabel('Probability')
    
    # Highlight the predicted digit
    predicted_digit = np.argmax(probs)
    bars[predicted_digit].set_color('orange')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}', ha='center', va='bottom')
    
    return fig

# Main app function
def main():
    # Custom CSS for better visuals
    st.markdown("""
    <style>
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
        .st-b7 {
            color: white;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .result-box {
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            background-color: #f0f2f6;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .digit-display {
            font-size: 5rem;
            text-align: center;
            margin: 1rem 0;
            color: #4CAF50;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("✍️ Handwritten Digit Classifier")
    st.markdown("Upload an image of a handwritten digit (0-9) and click **Predict** to see the results")

    # Session state for tracking
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False

    # Model upload section
    with st.expander("⚙️ Model Settings", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            model_file = st.file_uploader(
                "Upload your model file (.pkl)", 
                type=["pkl"],
                help="Upload a trained MNIST model in pickle format"
            )
        with col2:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            use_default = st.checkbox("Use default model", value=True)

    # Load model
    if use_default:
        default_model_path = "Web Dev/mnist_model.pkl"
        try:
            with open(default_model_path, "rb") as f:
                model = pickle.load(f)
            st.success("Default model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading default model: {e}")
            model = None
    elif model_file:
        with st.spinner('Loading model...'):
            model = load_model(model_file)
            if model:
                st.success("Custom model loaded successfully!")
    else:
        model = None
        st.info("Please upload a model or use the default one")

    # Image upload and prediction section
    st.header("📷 Image Upload")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', width=300)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔮 Predict", disabled=not model):
                    st.session_state.prediction_made = True
            with col2:
                if st.button("🔄 Clear Prediction"):
                    st.session_state.prediction_made = False
                    st.experimental_rerun()

            if st.session_state.prediction_made and model:
                with st.spinner('Analyzing the digit...'):
                    # Progress bar animation
                    progress_bar = st.progress(0)
                    for percent_complete in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(percent_complete + 1)
                    
                    processed_image = preprocess_image(image)
                    if processed_image is not None:
                        prediction = model.predict(processed_image)
                        predicted_class = np.argmax(prediction)
                        confidence = np.max(prediction)
                        
                        # Display results with visual effects
                        st.markdown("---")
                        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                        st.markdown("### 📊 Prediction Results")
                        
                        # Animated digit display
                        placeholder = st.empty()
                        for i in range(1, 6):
                            placeholder.markdown(f"""
                            <div class='digit-display' style='font-size: {3+i}rem;'>
                                {predicted_class}
                            </div>
                            """, unsafe_allow_html=True)
                            time.sleep(0.05)
                        
                        # Confidence meter
                        st.metric(label="Confidence Level", value=f"{confidence:.2%}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Probability visualization
                        st.markdown("### 📈 Probability Distribution")
                        fig = plot_probabilities(prediction[0])
                        st.pyplot(fig)
                        
                        # Detailed probabilities
                        with st.expander("🔍 View Detailed Probabilities"):
                            for i, prob in enumerate(prediction[0]):
                                st.progress(float(prob), text=f"Digit {i}: {prob:.2%}")

        except Exception as e:
            st.error(f"Error processing the uploaded image: {e}")

if __name__ == "__main__":
    main()