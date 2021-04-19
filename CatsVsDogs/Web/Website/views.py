from flask import Blueprint, flash, request, url_for
from flask.helpers import url_for
from flask.templating import render_template
from werkzeug.utils import redirect
from . import process_image, predict_image
from . import model
from .models import Prediction
from . import db
import numpy as np
import PIL

views = Blueprint('views', __name__)

@views.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            answer = Prediction.query.filter_by(id = 1).first()
            db.session.delete(answer)
            db.session.commit()
        except:
            pass
        try:
            image = request.files['img']
            image = PIL.Image.open(image)
            image = image.convert('RGB')
            image = np.asarray(image).astype(np.float32)
            flash('Image recieved', category="success")
            print(np.shape(image))
        except Exception as e:
            print(e)
        try:
            image = process_image(image)
            print(np.shape(image))
        except Exception as e:
            print(e)
        image = image[np.newaxis, :]
        try:
            prediction = predict_image(image, model)
            pred_cat = round(prediction[0][0]*100, 2)
            pred_dog = round(prediction[0][1]*100, 2)
        except Exception as e:
            print(e)
        pred_object = Prediction(id = 1, cat = pred_cat, dog = pred_dog)
        print(f"probability of dog: {pred_dog}, probability of cat: {pred_cat}")
        db.session.add(pred_object)
        db.session.commit()
        return redirect(url_for('views.answer'))
    return render_template('home.html')

@views.route('/answer', methods=['GET', 'POST'])
def answer():
    if request.method == 'POST':
        answer = Prediction.query.filter_by(id=1).first()
        db.session.delete(answer)
        db.session.commit()
        return redirect(url_for('views.home'))
    answer = Prediction.query.filter_by(id=1).first()
    return render_template('answer.html', user = answer)
