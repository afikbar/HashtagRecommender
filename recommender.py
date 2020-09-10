import pickle
import os
from flask import Flask, flash, request, redirect, url_for, g
from flask.templating import render_template
from werkzeug.utils import secure_filename
from HashtagRecommenderModel import HashtagRecommender, join_file

MYDIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join('static', 'uploads', '')
hr_model = None
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMPLATES_AUTO_RELOAD'] = True

ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def upload_form():
    image_name = request.args.get('image_name', default=False)
    return render_template('upload.html', image_name=image_name)


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        try:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)
                global hr_model
                hashtags = hr_model.predict_hashtags(save_path, num_neighbors=20, num_predict=10).values[0]
                flash(f"Recommended Hashtags: {', '.join(map(lambda x: '#'+x, hashtags))}", 'success')
                return redirect(url_for('upload_form', image_name=filename))
            else:
                flash(f"Allowed Files are: {ALLOWED_EXTENSIONS}", 'error')
                return redirect(request.url)
        except Exception as e:
            flash('Something went wrong, please try again', 'error')
            print(f"Error: {e}")
        return redirect(request.url)



@app.route('/read_hashtag', methods=['POST'])
def read_hashtag():
    if request.method == 'POST':
        input_hashtags = request.form['hashtag'].split(',')
        filename = request.args.get('image_name')
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        global hr_model
        hashtags = hr_model.predict_hashtags(save_path, num_neighbors=20, num_predict=10, selected_hashtags=input_hashtags).values[0]
        flash(f"Recommended Hashtags: {', '.join(map(lambda x: '#'+x, hashtags))}", 'success')
        return redirect(request.environ['HTTP_REFERER'])


@app.before_first_request
def before_first_request_func():
    join_file(source_dir='Model_Files', dest_file="model.pkl")
    global hr_model
    hr_model = pickle.load(open('model.pkl', 'rb'))

if __name__ == '__main__':
    app.run()
