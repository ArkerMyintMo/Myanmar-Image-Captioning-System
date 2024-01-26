import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from wtforms import SubmitField
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu

app = Flask(__name__)
app.config['SECRET_KEY'] = 'my_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'

MODEL_PATH = 'model/best_model_(1210).h5'
FEATURES_PATH = 'working/features_1210.pkl'
TOKENIZER_PATH = 'working/tokenization_pyi_1210.pkl'

vgg_model = VGG16()
vgg_model = tf.keras.Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

with open(FEATURES_PATH, 'rb') as f:
    features = pickle.load(f)

with open(TOKENIZER_PATH, 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

max_length = 20
model = load_model(MODEL_PATH)

class ImageForm(FlaskForm):
    image = FileField('Browse Image')
    submit = SubmitField('Generate Caption')

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')[0] 
        yhat = model.predict([image, np.array(sequence).reshape(1, -1)], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        if word == 'startseq':
            word = '<'
        in_text += " " + word
        if word == 'endseq':
            break
    in_text = in_text.replace('startseq', '').replace('endseq', '')
    return in_text.strip()

def idx_to_word(index, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word
    return None

def generate_caption(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    image_features = vgg_model.predict(img)
    caption = predict_caption(model, image_features, tokenizer, max_length)
    return caption

#def calculate_bleu(image_path, actual_captions):
    #img = Image.open(image_path)
    #img = img.resize((224, 224))
    #img = np.array(img)
    #img = preprocess_input(img)
    #img = np.expand_dims(img, axis=0)
    #image_features = vgg_model.predict(img)

    #actual, predicted = list(), list()

    #for _ in range(5): 
        #y_pred = predict_caption(model, image_features, tokenizer, max_length)
        
        # Split into words
        #ctual_captions_split = [caption.split() for caption in actual_captions]
        #y_pred = y_pred.split()
        
        # Append to the list
        #actual.append(actual_captions_split)
        #predicted.append(y_pred)

    # Calculate BLEU score
    #bleu_1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    #bleu_2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))

    #return bleu_1, bleu_2 

@app.context_processor
def inject_os():
    return dict(os=os)
@app.route('/', methods=['GET', 'POST'])
def index():
    form = ImageForm()
    image_path = None
    caption = None
    bleu_scores = None

    if form.validate_on_submit():
        image = form.image.data
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)

        caption = generate_caption(image_path)

    return render_template('index.html', form=form, image_path=image_path, caption=caption, bleu_scores=bleu_scores)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

#@app.route('/calculate_bleu', methods=['POST'])
#def calculate_bleu_endpoint():
    #input_image_path = request.form['image_path']

    # Read the actual captions from the text file for the entered image path
    #actual_captions = []
    #with open('C:/Users/arker/myanmar text/captions.txt', 'r', encoding='utf-8') as file:
        #for line in file:
            #parts = line.strip().split(',')
            #if len(parts) == 2 and parts[0] == os.path.basename(input_image_path):
                #actual_captions.append(parts[1])

    #bleu_scores = calculate_bleu(input_image_path, actual_captions)
    #return jsonify({'bleu_1': bleu_scores[0], 'bleu_2': bleu_scores[1]})


if __name__ == '__main__':
    app.run(debug=True)