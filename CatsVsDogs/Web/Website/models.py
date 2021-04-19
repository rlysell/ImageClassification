from . import db

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    cat = db.Column(db.Float)
    dog = db.Column(db.Float)
