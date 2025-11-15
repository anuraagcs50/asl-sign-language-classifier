import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
import numpy as np
from PIL import Image
import pickle

# Page config
st.set_page_config(
    page_title="ASL Classifier",
    page_icon="🤟",
    layout="centered"
)

# Title and description
st.title("🤟 American Sign Language Classifier")
st.markdown("Upload an image of an ASL sign (A-Z or 0-9) and the model will predict it!")

# Class labels - 0-9 then a-z (matching your training data)
CLASS_LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z'
]

@st.cache_resource
def build_model():
    """Rebuild the model architecture (MobileNetV2 + custom layers)"""
    # Create the base model
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Build the same architecture as your trained model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),  # Adjust if your dropout was different
        Dense(36, activation='softmax')
    ])
    
    return model

@st.cache_resource
def load_model_with_weights(weights_file):
    """Load the model and apply saved weights"""
    try:
        # Build the model architecture
        with st.spinner("Building model architecture..."):
            model = build_model()
        
        st.success("✅ Model architecture built!")
        
        # Load the weights
        with st.spinner("Loading trained weights..."):
            weights = pickle.load(weights_file)
            model.set_weights(weights)
        
        st.success("✅ Weights loaded successfully!")
        
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess image to match training format"""
    # Resize to 224x224
    img = image.resize((224, 224))
    
    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to array and normalize
    img_array = np.array(img)
    img_array = img_array / 255.0  # Rescale to [0,1]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict(model, image):
    """Make prediction on preprocessed image"""
    preprocessed = preprocess_image(image)
    predictions = model.predict(preprocessed, verbose=0)
    
    # Get top prediction
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [(CLASS_LABELS[idx], predictions[0][idx] * 100) for idx in top_3_idx]
    
    return CLASS_LABELS[predicted_class_idx], confidence, top_3_predictions

# Sidebar for weights upload
st.sidebar.header("⚙️ Model Settings")
st.sidebar.markdown("Upload your `model_weights.pkl` file")

weights_file = st.sidebar.file_uploader(
    "Choose weights file",
    type=['pkl'],
    help="Upload the model_weights.pkl file from Colab"
)

# Load model
model = None
if weights_file is not None:
    model = load_model_with_weights(weights_file)
    
    if model is not None:
        st.sidebar.success("🎉 Model ready!")
        st.sidebar.info(f"Input shape: (224, 224, 3)")
        st.sidebar.info(f"Output classes: 36")
else:
    st.sidebar.warning("⚠️ Please upload the weights file")

# Main area - Image upload and prediction
if model is not None:
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an ASL sign image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of an ASL hand sign"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("🔮 Prediction")
            
            # Make prediction
            with st.spinner("Analyzing..."):
                predicted_label, confidence, top_3 = predict(model, image)
            
            # Display main prediction
            st.markdown(f"### **{predicted_label.upper()}**")
            st.markdown(f"**Confidence:** {confidence:.2f}%")
            
            # Progress bar for confidence (convert to float for Streamlit)
            st.progress(float(confidence / 100))
            
            # Show top 3 predictions
            st.markdown("#### Top 3 Predictions:")
            for label, conf in top_3:
                st.text(f"{label.upper()}: {conf:.2f}%")
        
        st.markdown("---")
        
        # Additional info
        with st.expander("ℹ️ Tips for better predictions"):
            st.markdown("""
            - Use clear, well-lit images
            - Center the hand sign in the frame
            - Use a plain background if possible
            - Make sure the hand gesture is clearly visible
            - Avoid shadows or reflections
            """)
            
else:
    st.info("👆 Upload your `model_weights.pkl` file in the sidebar to get started")
    
    # Show example of what the app does
    st.markdown("---")
    st.subheader("📖 How it works")
    st.markdown("""
    1. **Upload weights**: Upload your `model_weights.pkl` file in the sidebar
    2. **Model rebuilds**: The app reconstructs your MobileNetV2 model architecture
    3. **Weights load**: Your trained weights are applied to the model
    4. **Upload image**: Upload an ASL sign image (letters a-z or numbers 0-9)
    5. **Get predictions**: See instant predictions with confidence scores
    """)
    
    st.markdown("---")
    st.info("💡 **Note**: This app rebuilds your exact model architecture (MobileNetV2 + custom layers) and loads your trained weights.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Built with Streamlit & TensorFlow | MobileNetV2 Architecture</div>",
    unsafe_allow_html=True
)