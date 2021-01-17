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
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

from pyod.utils.utility import standardizer
from pyod.models.abod import ABOD
from pyod.models.cof import COF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM

from fpdf import FPDF
from datetime import datetime, timedelta

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
@app.route('/report', methods=['POST'])
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
            clf_used, parameters = make_model(data)
            df, prediction = report(data=data, algorithm=clf_used)
            EDA(df,prediction= prediction)
            clean_data(df, filename, extension)
            df.rename(columns={0: "Wynik"}, inplace=True)
            for col_number in range(1, len(df.columns)):
                df.rename(columns={col_number: "Cecha " + str(col_number)}, inplace=True)
            for row_number in range(0, len(df)):
                df.rename(index={row_number: "Obserwacja " + str(row_number)}, inplace=True)

    return render_template('report.html',clasificator = clf_used ,parameters= parameters ,file_name=filename, tables=[df.to_html(classes='data')], titles=df.columns.values)

def access_data_mat(filename):
    data = loadmat(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return data

def access_data_csv(filename):
    data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    check_NAN = data.isnull().values.any()
    if check_NAN == True:
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer = imputer.fit(data)
        data = imputer.transform(data)
    data = data.to_numpy()
    data = np.delete(data,0,1)
    return data

def access_data_json(filename):
    data = pd.read_json(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    check_NAN = data.isnull().values.any()
    if check_NAN == True:
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer = imputer.fit(data)
        data = imputer.transform(data)
    data = data.to_numpy()
    return data

def make_model(data):
    transformer = RobustScaler().fit(data)
    Data_RobustScaled = transformer.transform(data)
    X_train, X_test = train_test_split(Data_RobustScaled, train_size=0.4, shuffle=True)
    prepare_trained_model()
    selected_models = select_model(X_train, n_selection=1)
    for foo, model in enumerate(selected_models):
        model = model.item(0)
        name = model
        best_clf = name.split(" ")
        clf = best_clf[0]
        param = best_clf[1:]
        parameters = {0:0}
        if clf == "ABOD":
            n_neighbour = param[0]
            n_neighbour = int(n_neighbour)
            model = ABOD(n_neighbors=n_neighbour, method='fast')
            parameters = {'Liczba sąsiadów': n_neighbour}
        if clf == "COF":
            n_neighbours = param[0]
            model = COF(n_neighbours=int(n_neighbours))
            parameters = {'Liczba sąsiadów': n_neighbours}
        if clf == "HBOS":
            n_histograms, tolerance = get_param(param)
            model = HBOS(n_bins=int(n_histograms), tol=float(tolerance))
            parameters = {'Liczba słupków histogramu': n_histograms,
                          'Tolerancja': tolerance}
        if clf == "Iforest":
            n_estimators, max_features = get_param(param)
            model = IForest(n_estimators=int(n_estimators), max_features=float(max_features))
            parameters = {'Liczba estymatorów': n_estimators,
                          'Limit cech': max_features}
        if clf == "kNN":
            n_neighbours, method = get_param(param)
            method = method[1:-1]
            model = KNN(n_neighbors=int(n_neighbours), method=method)
            parameters = {'Liczba sąsiadów': n_neighbours,
                          'Metoda': method}
        if clf == "LODA":
            n_bins, n_random_cuts = get_param(param)
            model = LODA(n_bins=int(n_bins), n_random_cuts=int(n_random_cuts))
            parameters = {'Liczba słupków histogramu': n_bins,
                          'Liczba losowych cięć': n_random_cuts}
        if clf == "LOF":
            n_neighbours, method = get_param(param)
            method = method[1:-1]
            model = LOF(n_neighbours=int(n_neighbours),metric=method)
            parameters = {'Liczba sąsiadów': n_neighbours,
                          'Metoda': method}
        if clf == "OCSVM":
            nu, kernel = get_param(param)
            kernel = kernel[1:-1]
            model = OCSVM(kernel=str(kernel), nu=float(nu))
            parameters = {'nu': nu,
                          'Jądro': kernel}
        dump(model, "static/model/clf.joblib")
        return clf, parameters

def report(data, algorithm):
    clf = load('static/model/clf.joblib')
    transformer = RobustScaler().fit(data)
    data_standarize = transformer.transform(data)
    clf.fit(data_standarize)
    pred = clf.predict_proba(data_standarize)
    labels, decision_score = clf.labels_, clf.decision_scores_
    labels_T = labels.reshape((-1, 1))
    labels_T = np.where(labels_T == 1, "Anomalia", "Prawidłowość")
    data_with_labels = np.append(labels_T,data, axis=1)
    dataframe = pd.DataFrame(data_with_labels)

    return dataframe, pred

def get_param(params):
    p1 = params[0]
    p2= params[1]
    p1 = p1.split("(")[1]
    p1 = p1.split(",")[0]
    p2 = p2.split(")")[0]
    return p1,p2

def clean_data(dataframe, filename, extension):
    df = dataframe.copy()
    indexNames = dataframe[dataframe[0] == "Anomalia"].index
    df.drop(indexNames, inplace=True)
    df.drop(df.columns[0], inplace=True, axis=1)
    if extension == "json":
        df.to_json('static/clean_data/clean_'+str(filename))
    elif extension == "csv":
        df.to_csv('static/clean_data/clean_'+str(filename))

def make_pdf_report():
    return 0

def EDA(df, prediction):
    dataframe_rows = len(df)
    dataframe_col = df.columns
    foo = df[0].value_counts()
    number_anomalies = foo["Anomalia"]
    percentage_anomaly = (number_anomalies/dataframe_rows)*100
    index_non_anomalies = df[df[0] != "Anomalia"].index
    index_anomalies = df[df[0] == "Anomalia"].index
    prob = np.delete(prediction,index_non_anomalies,axis=0)
    index_proba = []
    for i in range(len(index_anomalies)):
        index_proba.append([index_anomalies[i],prob[i][1]])
    index_proba = np.array(index_proba)
    return 0
if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int("80"),
        debug=True
    )



