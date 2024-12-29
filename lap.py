import pickle
import pandas as pd
import streamlit as st
df = pd.read_csv('laptop_project2.csv')
with open("laptop_model.pkl", "rb") as f:
    model, label_encoders = pickle.load(f)
st.image('download.jpg')
st.title('Laptop price pridiction')
data = {}
for i in model.feature_names_in_:
     value= st.selectbox(f'Pick {i}' ,df[i].unique() )
     data[i] = [value]
di = pd.DataFrame(data)
a = di.copy()
st.dataframe(a)
for col in a.columns:
    if col in label_encoders:
        a[col] = label_encoders[col].transform(a[col])


di['price'] = model.predict(a)
col =['price', 'Brand' ,'Processor','RAM', 'ROM', 'Graphic Card']
input_features_df = di[col]
def recommend_laptops(df, input_features_df, max_rows=5, price_tolerance=5000):
    filtered_df = df.copy()
    for feature, value in input_features_df.iloc[0].items():
        if feature == 'price':  # Handle price range
            filtered_df = filtered_df[(filtered_df['price'] >= value - price_tolerance) &
                                       (filtered_df['price'] <= value + price_tolerance)]
        else:
            filtered_df = filtered_df[filtered_df[feature] == value]
        if len(filtered_df) <= max_rows:
            break
    return filtered_df 

recommended_laptops = recommend_laptops(df, input_features_df)


if st.button('predict'):
    st.write(model.predict(a))
    st.write('Recomended laptops')
    st.dataframe(recommended_laptops)

