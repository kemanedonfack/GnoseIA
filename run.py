from flask import Flask, request, jsonify
import os
# from gnoseia import process_questions_and_files

app = Flask(__name__)


@app.route('/api/question', methods=['POST'])
def upload_file():
    # file = request.form.get('file')
    data = request.json  # Récupérer le JSON envoyé dans la requête
    if not data or 'question' not in data:
        return jsonify({'error': 'Question not provided'}), 400
    
    question = data['question']  # Récupérer la question depuis le JSON


    # if file.filename == '':
    #     return jsonify({'error': 'No selected file'}), 400

    # if file:
        #Eto enao no mampiditra ireo fonctionao izao mampanao test ny IA rehetra
        #eto no manoratra ny traitement rehetra

    response = process_questions_and_files(question, file)
    # response = "Bienvenu a ici chez moi"
    # references = ['Welcome', 'Darbie', "Phenomene"]

    # print(f"question: {question} \n reponse: {response} \n references: {references}")
    # return jsonify({ 'reponse': response, 'references': references}), 200

    
    print(f"question: {question} \n reponse: {response['reponse']} \n references: {response['references']}")
    return jsonify({ 'reponse': {response['reponse']}, 'references': {response['references']}}), 200

if __name__ == '__main__':
    app.run(debug=True)
