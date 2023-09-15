import streamlit as st
from fastai.vision.all import *
import pathlib
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

current_dir = os.getcwd()
model = 'resnet_model_14_Sep_1.pkl'
model_path = os.path.join(current_dir, model)

model = load_learner(model_path)


st.title("Defect Classification")
st.write("Upload an image to check whether if it's normal or defective (stain)")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform Predictions
    if st.button("Predict"):
        img = PILImage.create(uploaded_file)
        # Make predictions
        predictions, _ = model.get_preds(dl=model.dls.test_dl([img]))
        # Get the predicted class index
        predicted_class_idx = predictions.argmax(dim=1).item()
        # Get the confidence score for the predicted class
        confidence_score = predictions[0][predicted_class_idx].item()
        # Map the class index to class name
        class_names = model.dls.vocab
        predicted_class = class_names[predicted_class_idx]

        # Display the prediction and confidence score
        st.write(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence_score:.4f}")

