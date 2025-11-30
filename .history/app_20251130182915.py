# import the packages
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import xlrd
# (imports moved to top)

from contextlib import nullcontext
 

# Helper: return column-like objects in a Streamlit-version-safe way
def get_columns(arg):
    """Return columns for a given arg (list of widths or integer count).
    Uses `st.columns` when available, falls back to `st.beta_columns`,
    and finally to a list of containers as a graceful fallback.
    """
    # integer case
    try:
        if isinstance(arg, int):
            if hasattr(st, 'columns'):
                return st.columns(arg)
            if hasattr(st, 'beta_columns'):
                return st.beta_columns(arg)
            return [st.container() for _ in range(arg)]
        # list/iterable widths case
        if hasattr(st, 'columns'):
            return st.columns(arg)
        if hasattr(st, 'beta_columns'):
            return st.beta_columns(arg)
        length = len(arg)
        return [st.container() for _ in range(length)]
    except Exception:
        # final fallback
        if isinstance(arg, int):
            return [st.container() for _ in range(arg)]
        return [st.container() for _ in range(len(arg))]

    # Helper: display a metric in a column; fall back when `metric` isn't available
def display_metric(col, label, value, delta=None):
    try:
        if hasattr(col, 'metric'):
            # some Streamlit versions support an optional delta value
            if delta is None:
                col.metric(label, value)
            else:
                col.metric(label, value, delta)
        else:
            # fallback: show label and value plainly
            try:
                col.markdown(f"**{label}**")
            except Exception:
                col.write(label)
            col.write(value)
    except Exception:
        # Last-resort fallback
        col.write(f"{label}: {value}")

# set the web page configuration
st.set_page_config(page_title="Inventory Engine",
                   initial_sidebar_state="auto",
                   page_icon="😎",
                   layout="wide")
# Define the title of the application & the markdown
st.title('Inventory Optimization Engine! ⚙️')
st.write('Generates accurate Forecast, Safety stock, Reorder level & Economic order quantity. Play with the app by changing inputs, have fun🙂')

# Define the tabs of the application
tabs = ["Application", "About"]
page = st.sidebar.radio("Pages, feel free to close me for wider view 🧾", tabs)

# Sample data used when no file is uploaded
demand_format = pd.DataFrame({'Period': ['2021-01-01','2021-02-01','2021-03-01','2021-04-01','2021-05-01','2021-06-01','2021-07-01','2021-08-01'], 'Demand': [10, 12, 13, 11, 12, 9,10,13]})

# Sidebar inputs (cleaner UI) placed inside a form so calculations only run on submit
st.sidebar.header("Inputs / Parameters")
import io, base64
csv_bytes = io.BytesIO()
pd.DataFrame(demand_format).to_csv(csv_bytes, index=False)
csv_bytes.seek(0)

# Render download link/button outside the form (always available)
try:
    st.sidebar.download_button('Download sample data (CSV)', data=csv_bytes, file_name='sample_demand.csv', mime='text/csv')
except Exception:
    b64 = base64.b64encode(csv_bytes.getvalue()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="sample_demand.csv">Download sample data (CSV)</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)

# Parameters form: inputs are collected but calculations run only after submit
try:
    # Prefer the modern form API when available
    with st.sidebar.form("params_form"):
        uploaded_file = st.file_uploader('Upload CSV (Period,Demand)', type=['csv'])
        use_sample = st.checkbox('Use sample data', value=(uploaded_file is None))
        LeadTime = st.number_input('Lead time (periods)', 1, value=1)
        ServiceLevel = st.slider('Service level (%)', min_value=90.0, max_value=99.9, value=99.0, step=0.1, format="%g percent")
        ItemCost = st.number_input('Item cost', min_value=1, max_value=100000, value=100)
        OrderingCost = st.number_input('Ordering cost', min_value=1, max_value=100000, value=100)
        InventoryPercentage = st.slider('Inventory % of item cost', min_value=10, max_value=30, value=15, format="%g percent")
        AnnualDemand = st.number_input('Annual demand', min_value=1, max_value=100000, value=500)
        submit = st.form_submit_button('Submit')
except Exception:
    # Fallback for older Streamlit versions without form(): render inputs directly
    uploaded_file = st.sidebar.file_uploader('Upload CSV (Period,Demand)', type=['csv'])
    use_sample = st.sidebar.checkbox('Use sample data', value=(uploaded_file is None))
    LeadTime = st.sidebar.number_input('Lead time (periods)', 1, value=1)
    ServiceLevel = st.sidebar.slider('Service level (%)', min_value=90.0, max_value=99.9, value=99.0, step=0.1, format="%g percent")
    ItemCost = st.sidebar.number_input('Item cost', min_value=1, max_value=100000, value=100)
    OrderingCost = st.sidebar.number_input('Ordering cost', min_value=1, max_value=100000, value=100)
    InventoryPercentage = st.sidebar.slider('Inventory % of item cost', min_value=10, max_value=30, value=15, format="%g percent")
    AnnualDemand = st.sidebar.number_input('Annual demand', min_value=1, max_value=100000, value=500)
    submit = st.sidebar.button('Submit')

# write the content for the page Application
if page == "Application":
    st.header('Upload Input Data 🔧 ')
    st.write('By default sample data is loaded. For your calculations please input new data.')
    st.subheader("1. Input demand history for minimum 12 periods.")
    # write the collapse information
    # Compatibility: some Streamlit versions use `expander`, older ones may use `beta_expander`.
    if hasattr(st, 'expander'):
        expander = st.expander
    elif hasattr(st, 'beta_expander'):
        expander = st.beta_expander
    else:
        def _dummy_expander(title=""):
            st.markdown(f"**{title}**")
            return nullcontext()
        expander = _dummy_expander

    with expander("Input format"):
        st.write("Input should be a dataframe/table with two columns: Period and Demand. "
                 "Period can be in Day|Week|Month|Year "
                 "ideally in the format YYYY-MM-DD. "
                 "Demand column must be numeric.")
        st.write(demand_format)

    # Wait for user to submit the form before performing calculations
    if not ('submit' in globals() or 'submit' in locals()):
        # In some Streamlit versions the form variables may not be set yet
        st.info('Set inputs in the sidebar and click Submit to run calculations.')
        st.stop()

    if not submit:
        st.info('Set inputs in the sidebar and click Submit to run calculations.')
        st.stop()

    # pick data source: uploaded file or sample (after submit)
    if use_sample or uploaded_file is None:
        demand = demand_format.copy()
        st.info('Using sample data (you can upload your CSV from the sidebar).')
    else:
        try:
            demand = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error('Failed to read uploaded CSV; using sample data.')
            demand = demand_format.copy()
    demand = pd.DataFrame(demand)
    demand.columns = ['Period', 'Demand']
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
    Lead_Time_Demand = forecast_demand * LeadTime
    Standard_Deviation = demand['Demand'].std()
    SL1 = ServiceLevel / 100.0
    Service_Factor = norm.ppf(SL1)
    Lead_Time_Factor = np.sqrt(LeadTime)
    Safety_Stock = Standard_Deviation * Service_Factor * Lead_Time_Factor
    Reorder_Point = Safety_Stock + Lead_Time_Demand
    EOQ = np.sqrt((2 * AnnualDemand * OrderingCost) / (ItemCost * (InventoryPercentage / 100)))

    # Present results in a compact, visual way
    st.subheader('Forecast & Key Metrics')
    st.write('Forecast for next period:')
    cols = get_columns([2, 1, 1, 1])
    with cols[0]:
        # show chart of history + forecast
        df_plot = demand.copy()
        try:
            last_period = pd.to_datetime(demand['Period']).max()
            diffs = pd.to_datetime(demand['Period']).diff().dropna()
            delta = diffs.median() if len(diffs) > 0 else pd.Timedelta(days=1)
            next_period = last_period + delta
        except Exception:
            next_period = pd.Timestamp.now()
        df_plot = df_plot.set_index(pd.to_datetime(df_plot['Period']))
        df_plot = df_plot['Demand']
        df_forecast = pd.Series([forecast_demand], index=[pd.to_datetime(next_period)])
        chart_df = pd.concat([df_plot, df_forecast.rename('Forecast')], axis=0)
        st.line_chart(chart_df)
    # metrics
    cols_metrics = get_columns(3)
    display_metric(cols_metrics[0], 'Safety Stock', f"{int(round(Safety_Stock))}")
    display_metric(cols_metrics[1], 'Reorder Point', f"{int(round(Reorder_Point))}")
    display_metric(cols_metrics[2], 'Economic Order Qty', f"{int(round(EOQ))}")

    st.markdown('---')
    st.header('Details')
    st.write('Forecast (next period):', round(forecast_demand, 2))
    st.write('Lead time demand:', round(Lead_Time_Demand, 2))
    st.write('Std deviation of demand:', round(Standard_Deviation, 2))

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
