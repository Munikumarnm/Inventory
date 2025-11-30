# import the packages
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import xlrd

# set the web page configuration
st.set_page_config(page_title="Inventory Engine",
                   initial_sidebar_state="auto",
                   page_icon="😎")
# Define the title of the application & the markdown
st.title('Inventory Optimization Engine! ⚙️')
st.write('Generates accurate Forecast, Safety stock, Reorder level & Economic order quantity. Play with the app by changing inputs, have fun🙂')

# Define the tabs of the application
tabs = ["Application", "About"]
page = st.sidebar.radio("Pages, feel free to close me for wider view 🧾", tabs)

# write the content for the page Application
if page == "Application":
    st.header('Upload Input Data 🔧 ')
    st.write('By default sample data is loaded. For your calculations please input new data.')
    st.subheader("1. Input demand history for minimum 12 periods.")
# write the collapse information
    with st.expander("Input format"):
        st.write("Input should be a dataframe/table with two columns: Period and Demand. "
                 "Period can be in Day|Week|Month|Year "
                 "ideally in the format YYYY-MM-DD. "
                 "Demand column must be numeric.")
        demand_format = pd.DataFrame({'Period': ['2021-01-01','2021-02-01','2021-03-01','2021-04-01','2021-05-01','2021-06-01','2021-07-01','2021-08-01'], 'Demand': [10, 12, 13, 11, 12, 9,10,13]})
        st.write(demand_format)

    uploaded_file = st.file_uploader('')
    if uploaded_file is None:
        st.write("upload csv file")
    try:
        demand= pd.read_csv(uploaded_file)
    except:
        demand=demand_format
    demand= pd.DataFrame(demand)
    demand.columns= ['Period','Demand']
    st.subheader('2. Enter Supplier Lead Time 🔧 ')
    LeadTime = st.number_input('Caution:Demand & Lead time should be in same time frame. Example,If you have chosen weekly demand then the lead time should be in weeks.',1)
    st.subheader('3. Select Desired Service Level 🔧 ')
    ServiceLevel = st.slider('It is the probability of product is available to your customer',min_value=90.0, max_value=99.9,value=99.0,step=0.1, format="%g percent")
    LT= pd.DataFrame({'LeadTime':[LeadTime]})
    SL= ServiceLevel
    st.subheader('4. Enter The Item Cost 🔧 ')
    ItemCost = st.number_input('Item cost is amount paid per product during the purchase',min_value=1, max_value=100000,value=100)
    st.subheader('5. Ordering Cost 🔧 ')
    with st.expander("How to calculate ordering cost"):
        st.write("Ordering cost per item = ((Avg hours spent per order creation + follow up) * Man Hour Rate)/Number of products in the order.")
    OrderingCost = st.number_input('Enter Ordering cost',min_value=1, max_value=100000,value=100)
    st.subheader('6. Inventory Percentage 🔧 ')
    InventoryPercentage= st.slider("It is the x% of item cost considered for inventory holding cost. Usually supply chain/operations managers defines this value", min_value=10, max_value=30,value=15, format="%g percent")
    st.subheader('7. Annual Demand  🔧 ')
    AnnualDemand = st.number_input('Annual Demand of the item',min_value=1, max_value=100000,value=500)
# Subheader
    #st.header('Recap Input Data')
    #col1,col2,col3,col4,col5,col6= st.beta_columns(6)
    #col1.write("Demand Input")
    #col1.write(demand)
    #col2.write("Supplier Lead Time")
    #col2.write(LT)
    #col3.write("Service Level")
    #col3.write(SL)
    #col4.write("Item Cost")
    #col4.write(ItemCost)
    #col5.write("Ordering Cost")
    #col5.write(OrderingCost)
    #col6.write("Inventory Percentage")
    #col6.write(InventoryPercentage)

#Subheader
    demand['Period']= pd.to_datetime(demand['Period'])
    forecast_horizon=1
    # Build a minimal test dataframe for prediction (Auto_TS expects a DataFrame)
    try:
        last_period = demand['Period'].max()
        # infer a period delta; fall back to 1 day if inference fails
        diffs = demand['Period'].diff().dropna()
        if len(diffs) > 0:
            delta = diffs.median()
        else:
            delta = pd.Timedelta(days=1)
        next_period = last_period + delta
        testdata = pd.DataFrame({'Period': [next_period], 'Demand': [np.nan]})
        # Simple linear regression for forecasting
        X = np.arange(len(demand)).reshape(-1, 1)
        y = demand['Demand'].values
        model = LinearRegression()
        model.fit(X, y)
        forecast_demand = float(model.predict([[len(demand)]])[0])
    except Exception:
        # fallback: use the recent average demand if forecasting fails
        forecast_demand = float(demand['Demand'].tail(3).mean())
    Lead_Time_Demand = forecast_demand*LeadTime
    Standard_Deviation = demand['Demand'].std()
    SL1=SL/100 # divide SL% by 100 for the calculation
    Service_Factor = norm.ppf(SL1)
    Lead_Time_Factor =np.sqrt(LeadTime)
    Safety_Stock = Standard_Deviation*Service_Factor*Lead_Time_Factor
    Reorder_Point = Safety_Stock+Lead_Time_Demand
    EOQ= np.sqrt((2*AnnualDemand*OrderingCost)/(ItemCost*(InventoryPercentage/100)))
    st.header('Forecast Generated Using AutoML')
    st.write("Forecasting model used", best_model[0:1])
    st.write("Forecast for the next period", round(forecast_demand,2))
    st.header('We Are Done, Check The Result!')
    st.subheader("Safety Stock :")
    st.write(round(Safety_Stock,0))
   #st.write('Safety Stock of,round(Safety_Stock,0) to avoid stock out' )
    st.subheader("Reorder Level:")
    st.write(round(Reorder_Point, 0))
    #st.write('New purchase order for this product to be raised when stock reaches')
    st.subheader("Economic Order Quantity:")
    st.write(round(EOQ, 0))
    #st.write('The optimum order quantity needs to be while placing new order')

if page == "About":
    #st.image("Inv1.jpg")
    st.header("About")
    st.markdown("Official documentation of **[Streamlit](https://docs.streamlit.io/en/stable/getting_started.html)**")
    st.markdown("Logic for the calculation of Safety stock & reorder level **[Link](https://www.lokad.com/calculate-safety-stocks-with-sales-forecasting)**")
    st.markdown("Logic for the calculation of Economic order quantity **[Link](https://www.accountingformanagement.org/economic-order-quantity/)**")
    st.markdown("Forecast package used **[Auto_TS](https://github.com/AutoViML/Auto_TS/blob/master/README.md/)**")
    st.write("Author:")
    st.markdown(""" **[Munikumar N.M](https://www.linkedin.com/in/munikumarnm/)**""")
    st.markdown("""**[Source code](https://github.com/Munikumarnm/streamlit)**""")
    st.write("Created on 20/03/2021")
    st.write("Last updated: **13/06/2021**")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
