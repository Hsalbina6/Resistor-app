import os
os.system("pip uninstall -y opencv-python")

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.set_page_config(
    page_title="Smart Resistor Detection System",
    page_icon="⚡",
    layout="wide"
)

SPECIALIST_VALUES = ["10", "220", "330", "1000", "4700", "6800", "8200", "9200", "10000", "20000"]
MODEL_PATH = "my_SP_1_Model.pt"


def format_resistance(value):
    value = int(value)
    if value >= 1000:
        return f"{value/1000:g} kΩ"
    return f"{value} Ω"


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

st.title("⚡ Smart Resistor Detection System")
st.write(
    "This app demonstrates a resistor detection system using a trained YOLO Specialist model. "
    "The future goal is to combine a Specialist model, a Generalist model, and Smart Logic to improve reliability."
)

left_col, main_col, right_col = st.columns([1.1, 2.2, 1.2])

with left_col:
    st.subheader("Specialist Values")
    st.write("Current resistor values supported by the Specialist model:")

    for value in SPECIALIST_VALUES:
        st.markdown(f"- **{format_resistance(value)}**")

    st.info(
        "Specialist Note:\n\n"
        "The Specialist model is limited to the resistor values listed above, "
        "but it can perform well in tougher image conditions because it directly "
        "learns the full visual pattern of each known resistor value."
    )

with right_col:
    st.subheader("Model Notes")

    st.markdown("### Specialist Model")
    st.write(
        "Detects common resistor values directly. It is accurate for known values, "
        "but cannot predict values outside its trained list."
    )

    st.markdown("### Generalist Model")
    st.write(
        "Coming soon. This model will detect individual resistor color bands and compute "
        "the resistance value. It is not limited to a fixed value list, but needs clearer image conditions."
    )

    st.markdown("### Smart Logic")
    st.write(
        "Coming soon. This logic will combine the Specialist and Generalist outputs to handle agreement, "
        "conflict, and confidence levels."
    )

with main_col:
    st.subheader("Choose Detection Mode")

    model_choice = st.selectbox(
        "Select model mode:",
        [
            "Specialist Model",
            "Generalist Model (Coming Soon)",
            "Smart Logic System (Coming Soon)"
        ]
    )

    if model_choice != "Specialist Model":
        st.warning("This mode is planned but not active yet. Please use the Specialist Model for now.")
        st.stop()

    st.success("Specialist Model selected.")

    st.subheader("Smart Logic Workflow Preview")

    st.markdown(
        """
        ```text
        Models Output Computed Values
                    |
                    v
        Do both models agree?
            | Yes --> Display Value (High Confidence)
            |
            No
            v
        Does the predicted value exist in the Specialist database?
            | Yes --> Display Specialist Value (Medium Confidence)
            |
            No
            v
        Display Generalist Value (Low Confidence)
        ```
        """
    )

    st.caption(
        "Smart Logic idea: the system gives high confidence when both models agree, "
        "downgrades confidence when there is conflict, and uses model limitations to guide the final output."
    )

    st.divider()

    st.subheader("Prediction Settings")
    confidence = st.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)

    option = st.radio(
        "Choose input method:",
        ["Upload Image", "Use Camera"]
    )

    image_file = None

    if option == "Upload Image":
        image_file = st.file_uploader(
            "Upload resistor image",
            type=["jpg", "jpeg", "png"]
        )
    else:
        image_file = st.camera_input("Take a resistor picture")

    if image_file is not None:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Input Image", use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name

        results = model.predict(
            source=temp_path,
            conf=confidence,
            save=False
        )

        result = results[0]
        plotted_image = result.plot()[..., ::-1] # Converting BGR to RGB for Streamlit

        st.subheader("Prediction Result")
        st.image(plotted_image, caption="Detected Resistor Value", use_container_width=True)

        st.subheader("Detected Classes")

        if len(result.boxes) == 0:
            st.warning("No resistor value detected. Try a clearer image or lower the confidence threshold.")
        else:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                conf_score = float(box.conf[0])

                st.success(f"Predicted Value: **{format_resistance(class_name)}**")
                st.write(f"Raw Model Class: {class_name} Ω")
                st.write(f"Confidence: {conf_score:.2f}")

        os.remove(temp_path)
    else:
        st.info("Upload an image or use the camera to start detection.")
