from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline  import CustomData,PredictPipeline


application = Flask(__name__)  #This will help to tell start point to elastic beanstalk 
app = application

@app.route('/')
def hello_world():
    # return 'Hello, World!'
    return render_template('index.html')  


@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    else:
        customize_data_config=CustomData(
                   carat=float(request.form.get('carat')),
                   depth=float(request.form.get('depth')),
                   table=float(request.form.get('table')),
                   x=float(request.form.get('x')),
                   y=float(request.form.get('y')),
                   z=float(request.form.get('z')),
                   cut=(request.form.get('cut')),
                   color=(request.form.get('color')),
                   clarity=(request.form.get('clarity'))
                   )

        final_data=customize_data_config.get_data_as_dataframe()
        pipeline=PredictPipeline()
        predicted_data=pipeline.pipeline_config(final_data)
        
        results=round(predicted_data[0],2)
        print(results)
        return render_template('form.html',final_result=results)
        





if __name__=="__main__":
    app.run(host="0.0.0.0" )
