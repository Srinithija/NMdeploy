import streamlit as st
import requests
import pyttsx3
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import random
import io
import time
import threading
import matplotlib.pyplot as plt

# Set page config as the very first Streamlit command
st.set_page_config(page_title="Multi-Digit Drawing Game", page_icon="🎯", layout="centered")

# --- Speech ---
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("voice", engine.getProperty("voices")[1].id)
    engine.say(text)
    engine.runAndWait()

def speak_async(text):
    threading.Thread(target=speak, args=(text,), daemon=True).start()

# --- State Initialization ---
if "page" not in st.session_state:
    st.session_state.page = "home"
if "difficulty" not in st.session_state:
    st.session_state.difficulty = "Easy"
if "target_digits" not in st.session_state:
    st.session_state.target_digits = []
if "score" not in st.session_state:
    st.session_state.score = 0
if "attempts" not in st.session_state:
    st.session_state.attempts = 0
if "high_score" not in st.session_state:
    st.session_state.high_score = 0
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = []

# --- Styles ---
def set_theme():
    st.markdown(
        """
        <style>
            .stApp {
                background-color: #ffe6f0;
                color: #000;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

set_theme()

# --- Home Page ---
if st.session_state.page == "home":
    st.title("🎯 Multi-Digit Drawing Game")
    st.markdown("Choose your difficulty and try to draw the number shown!")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🟢 Easy (1 digit)"):
            st.session_state.difficulty = "Easy"
            st.session_state.target_digits = [random.randint(0, 9)]
            st.session_state.page = "game"
            st.session_state.start_time = time.time()
            st.rerun()
    with col2:
        if st.button("🟡 Medium (3 digits)"):
            st.session_state.difficulty = "Medium"
            st.session_state.target_digits = [random.randint(0, 9) for _ in range(3)]
            st.session_state.page = "game"
            st.session_state.start_time = time.time()
            st.rerun()
    with col3:
        if st.button("🔴 Hard (5 digits)"):
            st.session_state.difficulty = "Hard"
            st.session_state.target_digits = [random.randint(0, 9) for _ in range(5)]
            st.session_state.page = "game"
            st.session_state.start_time = time.time()
            st.rerun()

    if st.button("ℹ️ Help / Instructions"):
        with st.expander("Instructions"):
            st.markdown("""
            - Select a difficulty level.
            - Draw the number shown using your mouse or touchscreen.
            - Your score and accuracy are shown.
            - Aim to improve your high score and get on the leaderboard!
            """)

# --- Game Page ---
elif st.session_state.page == "game":
    st.header("🧠 Draw the Number")

    target_number = "".join(str(d) for d in st.session_state.target_digits)
    st.subheader(f"Target: **{target_number}**")

    # Timer
    time_limit = 30
    elapsed_time = int(time.time() - st.session_state.start_time)
    remaining_time = max(0, time_limit - elapsed_time)
    st.markdown(f"⏳ Time left: **{remaining_time} seconds**")

    if remaining_time == 0:
        st.error("⏰ Time's up!")
        speak_async("Time's up!")
        if st.button("🔁 Try Again"):
            st.session_state.page = "home"
            st.rerun()
        st.stop()

    # Drawing canvas
    st.markdown("✍️ Draw below:")
    canvas = st_canvas(
        fill_color="#000000",
        stroke_width=10,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas_fixed"
    )

    # Prediction Button
    if st.button("✨ Check Drawing"):
        st.session_state.attempts += 1

        if canvas.image_data is not None:
            full_img = Image.fromarray((255 - canvas.image_data[:, :, 0]).astype(np.uint8)).resize((280, 280))
            img_bytes_io = io.BytesIO()
            full_img.save(img_bytes_io, format="PNG")
            img_bytes = img_bytes_io.getvalue()
        else:
            st.warning("Draw something first.")
            st.stop()

       # Update API URL (add /predict)
API_URL = "https://digit-recognition-api-8r20.onrender.com/predict"  # 👈 Fixed

# In your prediction button code:
try:
    # Convert canvas to PNG properly
    img = Image.fromarray(255 - canvas.image_data[:, :, 0].astype(np.uint8))
    img_bytes_io = io.BytesIO()
    img.save(img_bytes_io, format="PNG")
    img_bytes = img_bytes_io.getvalue()
    
    # Add timeout and verify response
    response = requests.post(
        API_URL,
        files={"file": ("drawing.png", img_bytes, "image/png")},  # 👈 Proper file upload
        timeout=10
    )
    response.raise_for_status()  # Will raise error for bad status
    result = response.json()
    
    if "error" in result:
        st.error(f"API Error: {result['error']}")
    else:
        # Process successful response
        predicted_digits = result["predicted"]
        
except requests.exceptions.RequestException as e:
    st.error(f"Connection failed: {str(e)}")
except ValueError:
    st.error("Invalid response from server")

            predicted_number = "".join(str(d) for d in predicted_digits)
            st.markdown(f"✅ You drew: **{predicted_number}**")

            # Score and feedback
            if predicted_digits == st.session_state.target_digits:
                st.success("🎉 Correct!")
                st.balloons()
                st.session_state.score += 1
                st.session_state.high_score = max(st.session_state.score, st.session_state.high_score)
                st.session_state.leaderboard.append({
                    "score": st.session_state.score,
                    "accuracy": round(100 * st.session_state.score / st.session_state.attempts, 2),
                    "time": time.strftime("%H:%M:%S")
                })
                speak_async("Great job! That's correct.")
            else:
                st.error("❌ Try again!")
                speak_async(f"You wrote {predicted_number}. But the answer is {target_number}.")

            st.markdown(f"**Score:** {st.session_state.score}")
            st.markdown(f"**High Score:** {st.session_state.high_score}")
            st.markdown(f"**Accuracy:** {100 * st.session_state.score / st.session_state.attempts:.2f}%")

            # Confidence chart
            st.markdown("### 📊 Confidence Chart")
            fig, ax = plt.subplots()
            ax.bar(range(len(predicted_digits)), confidences, tick_label=[str(d) for d in predicted_digits], color="#ff69b4")
            ax.set_xlabel("Digits")
            ax.set_ylabel("Confidence (%)")
            ax.set_title("Prediction Confidence")
            st.pyplot(fig)

    # Buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔁 New Set"):
            digits_count = {"Easy": 1, "Medium": 3, "Hard": 5}[st.session_state.difficulty]
            st.session_state.target_digits = [random.randint(0, 9) for _ in range(digits_count)]
            st.session_state.start_time = time.time()
            st.rerun()
    with col2:
        if st.button("🔄 Reset Score"):
            st.session_state.score = 0
            st.session_state.attempts = 0
            st.session_state.leaderboard = []
            st.rerun()
    with col3:
        if st.button("🏠 Home"):
            st.session_state.page = "home"
            st.rerun()

    # Leaderboard display
    if st.session_state.leaderboard:
        st.markdown("## 🏆 Leaderboard")
        st.dataframe(st.session_state.leaderboard)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>🚀 Built with ❤️ by 2^ </p>", unsafe_allow_html=True)
