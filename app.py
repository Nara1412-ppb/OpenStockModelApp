from flask import Flask,render_template,request,jsonify
import numpy as np
import ml_models
import ml_predict

app = Flask(__name__)
stocks = []
stock_list = list(ml_models.stocks())
@app.route('/', methods = ['GET','POST']) 
def index():
    if request.method =='POST':
        data = request.form.get('stocks')
        print(data)
        if data:
            stocks.append(data)
            stock_list.remove(data)
            return render_template('predict.html', stock_name = stock_list, stocks = stocks)
        return render_template('predict.html', stock_name = stock_list, stocks = stocks)    
    return render_template('predict.html', stock_name = stock_list)

@app.route('/predict', methods = ['GET'])
def predict():
    results = []      
    for stock in stocks:
        prediction = ml_predict.predict_stock(stock)
        result = '{} : {}'.format(str(stock),str(prediction[0]))
        results.append(result)
    return render_template('predict.html', names = results)

  

@app.route('/prediction/<name>', methods = ["GET"])
def prediction(name):
    prediction = ml_predict.predict_stock(name)
    result = prediction[0]
    return jsonify({'prediction': result})

if __name__=='__main__':
    app.run(debug=True)