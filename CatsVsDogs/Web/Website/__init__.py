from flask import Flask
from tensorflow.keras import models
from tensorflow import image as im2
from tensorflow import float32
from flask_sqlalchemy import SQLAlchemy
from os import path

db = SQLAlchemy()
DB_NAME = 'image.db'


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'Very_secret'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)
    from .views import views
    app.register_blueprint(views, url_prefix='/')

    create_database(app)

    return app


model = models.load_model('cat_vs_dog_model')

def process_image(image):
    image = im2.resize(image, [224, 224]) / 255.0
    return image

def predict_image(image, model):
    prediction = model.predict(image).tolist()
    return prediction

def create_database(app):
    if not path.exists('website/' + DB_NAME):
        db.create_all(app = app)
        print("Database created!")

