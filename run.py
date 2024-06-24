from flask import Flask, request, jsonify
from gnoseIA_free import reponse_gnoseia, reponse_gnoseia_darby, reponse_legislative

app = Flask(__name__)


@app.route('/api/gnoseia', methods=['POST'])
def take_question_gnoseia():
    # file = request.form.get('file')
    data = request.json  # Récupérer le JSON envoyé dans la requête
    if not data or 'question' not in data:
        return jsonify({'error': 'Question not provided'}), 400
    
    question = data['question']  # Récupérer la question depuis le JSON
    file = f"https://gnoseia-corpus-storage.s3.eu-west-3.amazonaws.com/{data['file']}"

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    
    response = reponse_gnoseia(question, file)

    return jsonify({ 'reponse': response }), 200

@app.route('/api/darby', methods=['POST'])
def upload_file():
    # file = request.form.get('file')
    data = request.json  # Récupérer le JSON envoyé dans la requête
    if not data or 'question' not in data:
        return jsonify({'error': 'Question not provided'}), 400
    
    question = data['question']  # Récupérer la question depuis le JSON

    response = reponse_gnoseia_darby(question)

    print(f"question: {question} \n reponse: {response} \n references: {references}")
    return jsonify({ 'reponse': response }), 200

@app.route('/api/legislative', methods=['POST'])
def upload_file():
    # file = request.form.get('file')
    data = request.json  # Récupérer le JSON envoyé dans la requête
    if not data or 'question' not in data:
        return jsonify({'error': 'Question not provided'}), 400
    
    question = data['question']  # Récupérer la question depuis le JSON

    response = reponse_legislative(question)

    return jsonify({ 'reponse': response }), 200


if __name__ == '__main__':
    app.run(debug=True)
