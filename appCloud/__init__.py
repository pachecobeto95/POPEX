from flask import Flask, render_template, session, g
from appCloud.api.controllers import api


app = Flask(__name__, static_folder="static")


app.config.from_object("config")
app.config['JSON_AS_ASCII'] = False
app.secret_key = 'xyz'
app.register_blueprint(api)