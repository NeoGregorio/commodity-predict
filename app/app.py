from flask import Flask, render_template, request
import joblib
import pandas as pd

model = joblib.load('commodity-predict.pkl')
commodity_categories = ['commodity_Anchovies', 'commodity_Bananas (lakatan)', 'commodity_Bananas (latundan)', 'commodity_Bananas (saba)', 'commodity_Beans (green, fresh)', 'commodity_Beans (mung)', 'commodity_Beans (string)', 'commodity_Bitter melon', 'commodity_Bottle gourd', 'commodity_Cabbage', 'commodity_Cabbage (chinese)', 'commodity_Calamansi', 'commodity_Carrots', 'commodity_Chicken', 'commodity_Choko', 'commodity_Coconut', 'commodity_Crab', 'commodity_Eggplants', 'commodity_Eggs', 'commodity_Eggs (duck)', 'commodity_Fish (fresh)', 'commodity_Fish (frigate tuna)', 'commodity_Fish (mackerel, fresh)', 'commodity_Fish (milkfish)', 'commodity_Fish (redbelly yellowtail fusilier)', 'commodity_Fish (roundscad)', 'commodity_Fish (slipmouth)', 'commodity_Fish (threadfin bream)', 'commodity_Fish (tilapia)', 'commodity_Garlic', 'commodity_Garlic (large)', 'commodity_Garlic (small)', 'commodity_Ginger', 'commodity_Groundnuts (shelled)', 'commodity_Groundnuts (unshelled)', 'commodity_Maize (white)', 'commodity_Maize (yellow)', 'commodity_Maize flour (white)', 'commodity_Maize flour (yellow)', 'commodity_Mandarins', 'commodity_Mangoes (carabao)', 'commodity_Mangoes (piko)', 'commodity_Meat (beef)', 'commodity_Meat (beef, chops with bones)', 'commodity_Meat (chicken, whole)', 'commodity_Meat (pork)', 'commodity_Meat (pork, hock)', 'commodity_Meat (pork, with bones)', 'commodity_Meat (pork, with fat)', 'commodity_Oil (cooking)', 'commodity_Onions (red)', 'commodity_Onions (white)', 'commodity_Papaya', 'commodity_Pineapples', 'commodity_Potatoes (Irish)', 'commodity_Rice (milled, superior)', 'commodity_Rice (paddy)', 'commodity_Rice (premium)', 'commodity_Rice (regular, milled)', 'commodity_Rice (special)', 'commodity_Rice (well milled)', 'commodity_Semolina (white)', 'commodity_Semolina (yellow)', 'commodity_Shrimp (endeavor)', 'commodity_Shrimp (tiger)', 'commodity_Squashes', 'commodity_Sugar (brown)', 'commodity_Sugar (white)', 'commodity_Sweet Potato leaves', 'commodity_Sweet potatoes', 'commodity_Taro', 'commodity_Tomatoes', 'commodity_Water spinach']
commodity_categories = [s.replace('commodity_', '') for s in commodity_categories]
priceflag_categories = ['actual', 'aggregate']
pricetype_categories = ['Farm Gate', 'Retail', 'Wholesale']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    df_encoded = pd.DataFrame(columns=['region', 'year', 'month'] + 
                          [f'priceflag_{c}' for c in priceflag_categories] +
                          [f'pricetype_{c}' for c in pricetype_categories] +
                          [f'commodity_{c}' for c in commodity_categories]
                          )
    print(df_encoded.columns)
    print("Number of columns:", len(df_encoded.columns))
    # Get data from form
    data = {
        'commodity': [request.form.get('commodity')],
        'year': [int(request.form.get('year'))],
        'month': [int(request.form.get('month'))],
        'priceflag': [request.form.get('priceflag')],
        'pricetype': [request.form.get('pricetype')],
        'region': [int(request.form.get('region'))]
    }
    df = pd.DataFrame(data)
        
    # One-hot encode the categorical columns
    df_one_hot = pd.get_dummies(df, columns=['priceflag', 'pricetype', 'commodity'])
    print(df_one_hot)
    print("Number of columns:", len(df_one_hot.columns))
    # Add the one-hot encoded data to df_encoded
    for column in df_one_hot.columns:
        df_encoded[column] = df_one_hot[column]
    # Fill any remaining missing values with 0
    df_encoded = df_encoded.fillna(0)
    print(df_encoded.values)
    # Use the trained model to make predictions
    prediction = model.predict(df_encoded.values)

    # Print prediction
    print('Prediction:', prediction)

    # Return prediction
    return {'prediction': prediction.tolist()}