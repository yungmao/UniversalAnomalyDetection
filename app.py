# We'll render HTML templates and access data sent by POST
# using the request object from flask. Redirect and url_for
# will be used to redirect the user once the upload is done
# and send_from_directory will help us to send/show on the
# browser the file that the user just uploaded
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
from scipy.io import loadmat
import numpy as np
import os

from metaod.models.utility import prepare_trained_model
from metaod.models.predict_metaod import select_model

from scipy.io import loadmat

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pyod.utils.utility import standardizer
from pyod.models.abod import ABOD
from pyod.models.cof import COF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM

from joblib import dump, load


# Initialize the Flask application
app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['mat','csv','json'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def index():
    return render_template('index.html')

# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = secure_filename(file.filename)
    if filename != '':
        if file and allowed_file(filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            extension = filename.rsplit('.', 1)[1]
            data = []
            if extension == 'mat':
                data = access_data_mat(filename)
            elif extension == 'csv':
                data = access_data_csv(filename)
            elif extension == 'json':
                data = access_data_json(filename)
            make_model(data)
            report(data)
    return redirect(url_for('index'))


def access_data_mat(filename):
    data = loadmat(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return data

def access_data_csv(filename):
    data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    data = data.to_numpy()
    data = np.delete(data,0,1)
    return data

def access_data_json(filename):
    data = pd.read_json(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    data = data.to_numpy()
    return data

def make_model(data):
    X_train, X_test = train_test_split(data, train_size=0.4, shuffle=True)
    X_train_norm, X_test_norm = standardizer(X_train, X_test)
    X_train = StandardScaler(X_train).copy
    X_test = StandardScaler(X_test).copy
    prepare_trained_model()
    selected_models = select_model(X_train, n_selection=1)
    for foo, model in enumerate(selected_models):
        model = model.item(0)
        name = model
        best_clf = name.split(" ")
        clf = best_clf[0]
        param = best_clf[1:]
        if clf == "ABOD":
            n_neighbour = param[0]
            n_neighbour = int(n_neighbour)
            model = ABOD(n_neighbors=n_neighbour, method='fast')
        if clf == "COF":
            n_neighbours = param[0]
            model = COF(n_neighbours=int(n_neighbours))
        if clf == "HBOS":
            n_histograms, tolerance = get_param(param)
            model = HBOS(n_bins=int(n_histograms), tol=float(tolerance))
        if clf == "Iforest":
            n_estimators, max_features = get_param(param)
            model = IForest(n_estimators=int(n_estimators), max_features=float(max_features))
        if clf == "kNN":
            n_neighbours, method = get_param(param)
            model = KNN(n_neighbors=int(n_neighbours), method=method)
        if clf == "LODA":
            n_bins, n_random_cuts = get_param(param)
            model = LODA(n_bins=int(n_bins), n_random_cuts=int(n_random_cuts))
        if clf == "LOF":
            n_neighbours, method = get_param(param)
            model = LOF(n_neighbours=int(n_neighbours),metric=method)
        if clf == "OCSVM":
            nu, kernel = get_param(param)
            model = OCSVM(kernel=kernel, nu=float(nu))
        dump(model, "static/model/clf.joblib")

def report(data):
    clf = load('static/model/clf.joblib')
    data_standarize = StandardScaler(data).copy
    clf.fit(data_standarize)
    labels, decision_score = clf.labels_, clf.decision_scores_
    labels_T = labels.reshape((-1, 1))
    decision_score_T = decision_score.reshape((-1, 1))
    data_with_labels = np.append(labels_T,data, axis=1)
    dataframe = pd.DataFrame(data_with_labels)
    

def get_param(params):
    p1 = params[0]
    p2= params[1]
    p1 = p1.split("(")[1]
    p1 = p1.split(",")[0]
    p2 = p2.split(")")[0]
    return p1,p2

if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int("80"),
        debug=True
    )



