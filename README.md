This is a project of Alogorithmic trading to predict Openstock price build by NARASIMHA RAO NANNDIKONDA

#Installation and setting up application
1. install python3.8
2. create a virtual environment in the projectdirectory using "python -m venv env"
3. activate the virtual environment by following these in terminal
        1. cd env/Scripts
        2. activate
4. install requirements from requirements.txt file by using "pip install -r requirements.txt"
5. run the application using "flask run" or "python app.py".

#application interface
6. application will be open with local host @ "localhost:5000" 
7. select the stocks from the drop down and press select button one by one and you can see the selected stocks in list
8. click the submit button to get the estimated stock prices of the selected stocks.

9. This flask application contains an api you test it in testing enviromment. 
   by using "localhost:5000/prediction/< stock symbol name >"

   here stock symbol name is the official name of the company from yaahoo finance.

10. request should be "GET" for api testing.
	try out with below url "http://127.0.0.1:5000/prediction/AXISBANK" and "http://127.0.0.1:5000/prediction/TATAMOTORS"
