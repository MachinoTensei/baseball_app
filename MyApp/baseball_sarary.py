from flask import Flask, request, render_template
import keras
import tensorflow as tf

app = Flask(__name__)

#---------------モデルを読み込む---------------
#import pickle
#f = open("baseball_model", "rb")
#model = pickle.load(f)

model = keras.models.load_model("baseball_model.h5")
model._make_predict_function()
graph = tf.get_default_graph()
#---------------------------------------------------------------------------


#---------------モデルを実行するための関数---------------
import numpy as np
def model_run(data):

	mean_train = np.array([0.25924, 0.32521668, 46.94, 93.69334, 16.78, 2.4466667, 9, 44.1, 35.193333, 57.013332, 8.576667, 6.76])
	std_train = np.array([3.7369091e-02, 4.3865968e-02, 2.8560400e+01, 5.1223877e+01, 1.0147821e+01, 2.6179805e+00, 9.0631123e+00, 2.9168533e+01, 2.4886461e+01, 3.3527500e+01, 1.1886300e+01, 5.9678917e+00])
	std_label = 1248.6285
	mean_label = 1274.0566
	
	for i in range(12):
		data[i] = (data[i] - mean_train[i]) / std_train[i]
	data = data.reshape(1, 12)
	
	global graph
	with graph.as_default():
		ans = model.predict(data)
	ans = model.predict(data)
	ans = (ans * std_label) + mean_label
	ans = int(ans[0][0]*1000)
	
	return ans
#---------------------------------------------------------------------------

@app.route("/")
def home():
	message = "起動したいアプリを選んでください。"
	
	return render_template("home.html", message = message)

@app.route("/baseball")
def baseball():
	message = "下の項目を入力すると、年俸が算出されます。(1992年)"
	
	return render_template("baseball.html", message = message)


@app.route("/baseball/result", methods = ["POST"])
def baseball_result():
	message = "あなたの年俸が予測されました！"
	
	batting_avg = float(request.form["batting_avg"])
	obp = float(request.form["obp"])
	run = float(request.form["run"])
	hit = float(request.form["hit"])
	double = float(request.form["double"])
	triple = float(request.form["triple"])
	homerun = float(request.form["homerun"])
	rbi = float(request.form["rbi"])
	fourball = float(request.form["fourball"])
	strikeout = float(request.form["strikeout"])
	stolen = float(request.form["stolen"])
	error = float(request.form["error"])
	
	data_list = [batting_avg, obp, run, hit, double, triple, homerun, rbi, fourball, strikeout, stolen, error]
	data_np = np.array(data_list)
	sarary = model_run(data_np)
	sarary_yen = sarary * 128
	return render_template("baseball_result.html", message = message, sarary = sarary, sarary_yen = sarary_yen)


