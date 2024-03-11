from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///uploads.db'
db = SQLAlchemy(app)

class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    question = db.Column(db.String(255), nullable=True)

    def __repr__(self):
        return f"<Upload {self.filename}>"

@app.route('/api/question', methods=['POST'])
def upload_file():
    file = request.form.get('file')
    question = request.form.get('question')  # Récupération de la question depuis le formulaire

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        upload = Upload(filename=filename, question=question)  # Création de l'instance Upload

        #Eto enao no mampiditra ireo fonctionao izao mampanao test ny IA rehetra
        #eto no manoratra ny traitement rehetra


        db.session.add(upload)  # Ajout de l'instance à la session
        db.session.commit()  # Commit des changements

        #atao anaty io jsonify io ny resultat rehetra izay hitanao
        response = 'Ireo reponse ny teste rehetra atao ato'
        return jsonify({'message': 'File uploaded successfully', 'reponse': response}), 200

if __name__ == '__main__':
    app.run(debug=True)
