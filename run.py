from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/api/", methods=['GET'])
def hello():
    return jsonify({"title": "Hello world"})

if __name__ == "__main__":
    app.run(debug=True)