
# MNIST Digit Classifier Web App

This project is a web application that classifies handwritten digits using a Random Forest classifier trained on the MNIST dataset. The model is served through a FastAPI backend, and the frontend is a simple HTML page that allows users to upload images of handwritten digits and receive predictions.

## Overview

The application consists of two parts:

1. **Model Training (Backend)**: A machine learning model is trained on the MNIST dataset using a Random Forest Classifier. The trained model is then serialized using `pickle` and saved as a `.pkl` file.
2. **Prediction API (FastAPI)**: A FastAPI server serves the trained model and handles predictions. The user can upload an image of a digit, and the server will predict the digit and return the result.
3. **Frontend (HTML & JavaScript)**: The frontend consists of a simple HTML page where users can upload an image, which will then be sent to the backend for prediction. The result is displayed on the page.

## Requirements

To run this project, you need the following Python packages:

- `scikit-learn`
- `pandas`
- `pickle`
- `fastapi`
- `uvicorn`
- `Pillow`
- `numpy`

You can install the required packages by running:

```bash
pip install scikit-learn pandas fastapi uvicorn pillow numpy
```

## Model Training

The following code trains a Random Forest model on the MNIST dataset and saves the trained model to a file (`mnist_model.pkl`).

```python
import pickle
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the MNIST dataset
X, y = fetch_openml(name='mnist_784', version=1, return_X_y=True)

# Split the data into training and test sets
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the Random Forest model
clf = RandomForestClassifier(n_jobs=-1)

# Train the model
clf.fit(X_train, y_train)

# Save the trained model using pickle
with open('mnist_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
```

## FastAPI Backend

The FastAPI backend handles the prediction requests. The model is loaded from the saved `.pkl` file and used to predict the digit from an uploaded image.

```python
import io
import pickle
import numpy as np
import PIL.Image
import PIL.ImageOps
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Load the trained model
with open('mnist_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a FastAPI instance
app = FastAPI()

# Enable CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Define the prediction route
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    pil_image = PIL.Image.open(io.BytesIO(contents)).convert('L')
    pil_image = PIL.ImageOps.invert(pil_image)
    pil_image = pil_image.resize((28, 28), resample=PIL.Image.Resampling.LANCZOS)
    img_array = np.array(pil_image).reshape(1, -1)
    prediction = model.predict(img_array)
    return {"prediction": int(prediction[0])}
```

### Running the FastAPI Server

To run the FastAPI server, use the following command:

```bash
uvicorn main:app --reload
```

This will start the server on `http://127.0.0.1:8000`.

## Frontend (HTML)

The frontend allows users to upload an image of a handwritten digit and view the prediction result. The following is the HTML and JavaScript used for the frontend:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Digit Classifier</title>
  <style>
    /* Styling for the page */
  </style>
</head>
<body>
  <div class="container">
    <h1>🔢 MNIST Digit Classifier</h1>
    <label class="custom-file-upload">
      Upload a digit image
      <input type="file" id="file-input" accept="image/*" required />
    </label>
    <img id="preview" />
    <button id="classify-btn">Classify Digit</button>
    <div id="result"></div>
  </div>

  <script>
    // JavaScript code to handle file input and prediction
  </script>
</body>
</html>
```

## How to Use

1. Run the FastAPI backend server using the command:
   ```bash
   uvicorn main:app --reload
   ```
2. Open the `index.html` file in your browser.
3. Upload an image of a handwritten digit (28x28 pixels).
4. Click "Classify Digit" to get the predicted digit.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
