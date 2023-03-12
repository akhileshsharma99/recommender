from flask import Flask

# flask --app api run
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World!"