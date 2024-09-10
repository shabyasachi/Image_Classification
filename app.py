
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np



# Define a Flask app
app = Flask(__name__)

# Path to your saved model
MODEL_PATH = 'vgg19.h5'

# Load your trained model
model = load_model(MODEL_PATH)

# If needed, compile the model manually (depending on how the model was saved)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make a prediction
        preds = model_predict(file_path, model)

        # Process your result for human understanding
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result

    return None

if __name__ == '__main__':
    app.run(debug=True)

