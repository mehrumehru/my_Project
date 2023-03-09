import numpy as np
import pandas as pd
import streamlit as st 
# import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import joblib

scale = joblib.load('scale')
# Loading forecasting model
model = load_model('Forecasting.h5')

def main():
    
    st.title("Forecasting")
    st.sidebar.title("Forecasting")

    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Forecasting </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

    uploadedFile = st.sidebar.file_uploader("Choose a file" ,type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")
    if uploadedFile is not None :
        try:

            data=pd.read_csv(uploadedFile,  index_col=0)
        except:
                try:
                    data = pd.read_excel(uploadedFile,  index_col=0)
                except:      
                    data = pd.DataFrame(uploadedFile)
                
    else:
        st.sidebar.warning("you need to upload a csv or excel file.")
    
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")
    
    
    if st.button("Predict"):
        engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
        scaled_data = scale.transform(data)
        
        ###############################################
        st.subheader(":red[Forecast for Test data]", anchor=None)
        test_predictions = []
        n_input = 12
        n_features = 1
        first_eval_batch = scaled_data[-12:]
        current_batch = first_eval_batch.reshape((1, n_input, n_features))
        
        for i in range(len(data)):

            # get the prediction value for the first batch
            current_pred = model.predict(current_batch)[0]

            # append the prediction into the array
            test_predictions.append(current_pred) 

            # use the prediction to update the batch and remove the first value
            current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1) 
        
        true_predictions = scale.inverse_transform(test_predictions)

        data['Predictions'] = true_predictions
        
        data.to_sql('forecast_results', con = engine, if_exists = 'replace', index = False, chunksize = 1000)
        
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap=True)
        st.table(data.style.background_gradient(cmap=cm).set_precision(2))
        
        ###############################################
        st.text("")
        st.subheader(":red[plot forecasts against actual outcomes]", anchor=None)
        #plot forecasts against actual outcomes
        fig, ax = plt.subplots()
        ax.plot(data)
        #ax.plot(data.Predictions, color = 'red')
        st.pyplot(fig)
        
        ###############################################
        #st.text("")
        #st.subheader(":red[Forecast for the nest 12 months]", anchor=None)
        
        #forecast = pd.DataFrame(model.predict(start=data.index[-1] + 1, end=data.index[-1] + 12))
        #st.table(forecast.style.background_gradient(cmap=cm).set_precision(2))
        
        # data.to_sql('forecast_pred', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
        # #st.dataframe(result) or
        # #st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))
                           
        # import seaborn as sns
        # cm = sns.light_palette("blue", as_cmap=True)
        # st.table(result.style.background_gradient(cmap=cm).set_precision(2))

                           
if __name__=='__main__':
    main()


