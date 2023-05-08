from flask import Flask, render_template, request
from pydub import AudioSegment
from pydub.silence import split_on_silence
from python_speech_features import mfcc
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os 
import pickle

# Load model from JSON file
# Load trained model
with open("mlp.pkl", "rb") as f:
    model = pickle.load(f)

# Load model weights from file
#model.load_weights('model_weights.h5')


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get uploaded file
        audio_file = request.files['file']
        
        # Save audio file
        audio_path = 'temp.wav'
        audio_file.save(audio_path)

        n_mfcc = 20
        
        # Load audio file
        audio = AudioSegment.from_file(audio_path)
        audio_data = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate

        # Split audio into chunks of silence
        audio_chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-50)

        # Extract MFCCs from each chunk of audio
        mfccs_list = []
        nfft = 2048 # Set larger FFT size
        for chunk in audio_chunks:
            chunk_data = np.array(chunk.get_array_of_samples())
            mfccs = mfcc(chunk_data, samplerate=sample_rate, numcep=n_mfcc, nfft=nfft) # Set larger nfft value
            mfccs = np.mean(mfccs, axis=0)
            mfccs_list.append(mfccs)

        # Convert MFCCs to a numpy array
        mfccs_array = np.array(mfccs_list)

        # Reshape MFCCs to have 1 channel
        #mfccs_array = mfccs_array.reshape(mfccs_array.shape[0], mfccs_array.shape[1], 1)

        # Load trained model
        #model = load_model('model.h5')

        # Get predicted probabilities for each class
        probs = model.predict(mfccs_array)

        # Convert probabilities to predicted class labels
        predicted_classes = np.argmax(probs, axis=1)

        # Convert predicted class labels to class names
        class_names = ['Angry', 'Happy', 'Neutral', 'Sad']
        predicted_class_names = [class_names[i] for i in predicted_classes]

        # Delete audio file
        os.remove(audio_path)


        # Return predicted class
        return render_template('result.html', class_name=predicted_class_names[0])


if __name__ == '__main__':
    app.run(debug=True)
