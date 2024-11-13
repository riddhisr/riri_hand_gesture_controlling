import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# Load the trained model
model = tf.keras.models.load_model('gesturefinal.h5')

# List of gestures (ensure they match the order in your training data)
gestures = ['scroll_up', 'scroll_down', 'back', 'forward', 'screenshot', 'close_window', 'openapp', 'none']

# Define the confidence threshold and action cooldown settings
threshold = 0.7
action_cooldown = 1  # seconds between actions
last_action_time = 0

# Streamlit UI setup
st.set_page_config(page_title="Gesture Recognition Control", page_icon="🖐️", layout="wide")
st.title("🤖 Real-time Gesture Recognition Control")

# Sidebar setup for customization options
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Confidence Threshold", 0.5, 1.0, 0.7, 0.05)
action_cooldown = st.sidebar.slider("Action Cooldown (seconds)", 0.5, 5.0, 1.0, 0.5)

# Placeholder for displaying the video
st.markdown(
    """
    <style>
        .main { background-color: #f0f2f6; }
        h1 { color: #4A90E2; }
        .stButton>button { background-color: #4CAF50; color: white; font-size: 16px; }
    </style>
    """, unsafe_allow_html=True
)
st.write("### Gesture-based System Controls")

frame_placeholder = st.empty()
status_placeholder = st.sidebar.empty()

# Function to perform action based on recognized gesture
def perform_action(gesture):
    global last_action_time
    current_time = time.time()
    if current_time - last_action_time > action_cooldown:
        action_message = ""
        
        if gesture == 'scroll_up':
            # Implement scroll functionality (Replace pyautogui with another approach if needed)
            action_message = "Scrolled up"
            
        elif gesture == 'scroll_down':
            # Implement scroll functionality
            action_message = "Scrolled down"
            
        elif gesture == 'back':
            # Implement 'back' action functionality
            action_message = "Back action triggered"
            
        elif gesture == 'forward':
            # Implement 'forward' action functionality
            action_message = "Forward action triggered"
            
        elif gesture == 'screenshot':
            time.sleep(2)
            screenshot = Image.new('RGB', (300, 200), color = (73, 109, 137))  # Placeholder screenshot
            screenshot.save("screenshot.png")
            action_message = "Screenshot taken"
            st.image(screenshot, caption="Screenshot")

        elif gesture == 'close_window':
            # Implement close window functionality
            action_message = "Window closed"
            
        elif gesture == 'openapp':
            # Implement open application functionality
            action_message = "Opened Notepad"
        
        # Display action feedback
        status_placeholder.success(action_message)
        last_action_time = current_time

# Using Streamlit's camera input
camera_image = st.camera_input("Capture Your Gesture")

if camera_image is not None:
    image = Image.open(camera_image)
    image = np.array(image)  # Convert to numpy array

    # Preprocess the image for gesture recognition
    image_resized = cv2.resize(image, (64, 64))
    image_normalized = image_resized / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)

    # Predict gesture
    prediction = model.predict(image_input)
    confidence = np.max(prediction)
    predicted_class = np.argmax(prediction, axis=1)

    # Determine the gesture and perform action if confidence is high enough
    if confidence > threshold:
        predicted_gesture = gestures[predicted_class[0]]
        perform_action(predicted_gesture)
    else:
        predicted_gesture = 'none'
    
    # Display prediction and confidence
    st.write(f"Predicted: {predicted_gesture} ({confidence * 100:.2f}%)")

