import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent
FEATURES = ['Instance Memory', 'vCPUs']
TARGET = 'On Demand'
POINT_COLOR = '#2a6fdb'
MODEL_COLOR = '#d9480f'
HOURS_PER_MONTH = 730

st.set_page_config(page_title="EC2 Cost Predictor", layout="wide")


@st.cache_data
def load_data():
    data = pd.read_csv(BASE_DIR / 'ec2dataset.csv')

    # Same cleaning steps as MLmodel.ipynb
    cost_columns = ['On Demand', 'Linux Reserved cost', 'Linux Spot Minimum cost',
                    'Windows On Demand cost', 'Windows Reserved cost']
    for column in cost_columns:
        data[column] = pd.to_numeric(
            data[column].astype(str).str.replace(r'[$,]|hourly', '', regex=True).str.strip(),
            errors='coerce')
    data['Instance Memory'] = pd.to_numeric(data['Instance Memory'].astype(str).str.replace(' GiB', ''), errors='coerce')
    data['vCPUs'] = pd.to_numeric(data['vCPUs'].astype(str).str.extract(r'(\d+)', expand=False), errors='coerce')

    data_cleaned = data.dropna(subset=[TARGET] + FEATURES)

    # IQR outlier removal
    mask = pd.Series(True, index=data_cleaned.index)
    for column in [TARGET] + FEATURES:
        q1 = data_cleaned[column].quantile(0.25)
        q3 = data_cleaned[column].quantile(0.75)
        iqr = q3 - q1
        mask &= data_cleaned[column].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    return data, data_cleaned[mask].copy()


@st.cache_resource
def load_model():
    with open(BASE_DIR / 'modelEC2.pkl', 'rb') as f:
        return pickle.load(f)


def predict(model, memory, vcpus):
    return float(model.predict(pd.DataFrame([[memory, vcpus]], columns=FEATURES))[0])


def style(fig):
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation='h', y=1.08))
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)', zeroline=False)
    return fig


raw_data, data = load_data()
model = load_model()

# Same split as the notebook, so metrics match
X = data[FEATURES]
y = data[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred_test = model.predict(X_test)

# ---------- Sidebar: model inputs ----------
st.sidebar.header("Run the model")
memory = st.sidebar.number_input("Instance Memory (GiB)", min_value=0.5, max_value=1024.0, value=4.0, step=0.5)
vcpus = st.sidebar.number_input("vCPUs", min_value=1, max_value=192, value=2, step=1)

prediction = predict(model, memory, vcpus)
st.sidebar.metric("Predicted On-Demand cost", f"${prediction:.4f} / hr")
st.sidebar.caption(f"≈ ${prediction * HOURS_PER_MONTH:,.2f} / month ({HOURS_PER_MONTH} hrs)")

mem_min, mem_max = data['Instance Memory'].min(), data['Instance Memory'].max()
cpu_min, cpu_max = data['vCPUs'].min(), data['vCPUs'].max()
if not (mem_min <= memory <= mem_max and cpu_min <= vcpus <= cpu_max):
    st.sidebar.warning(
        f"Input is outside the training range ({mem_min:g}–{mem_max:g} GiB, "
        f"{cpu_min:g}–{cpu_max:g} vCPUs). The prediction is an extrapolation.")

# ---------- Header + KPIs ----------
st.title("EC2 On-Demand Cost Predictor")
st.caption("Log-log linear regression: log(cost) ~ log(memory) + log(vCPUs). Trained on ec2dataset.csv with IQR outliers removed.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Instances (after cleaning)", f"{len(data):,}", f"{len(data) - len(raw_data):,} removed", delta_color="off")
k2.metric("Median cost", f"${data[TARGET].median():.4f} / hr")
k3.metric("R² (test set)", f"{r2_score(y_test, y_pred_test):.3f}")
k4.metric("MAE (test set)", f"${mean_absolute_error(y_test, y_pred_test):.4f}",
          f"RMSE ${np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}", delta_color="off")

# ---------- Similar real instances ----------
st.subheader(f"Real instances closest to {memory:g} GiB / {vcpus} vCPUs")
distance = np.hypot(np.log(data['Instance Memory'] / memory), np.log(data['vCPUs'] / vcpus))
similar = data.assign(Predicted=model.predict(data[FEATURES])).loc[distance.nsmallest(5).index]
st.dataframe(
    similar[['Name', 'API Name', 'Instance Memory', 'vCPUs', TARGET, 'Predicted']],
    hide_index=True, width='stretch',
    column_config={
        'Instance Memory': st.column_config.NumberColumn(format="%g GiB"),
        TARGET: st.column_config.NumberColumn("Actual $/hr", format="$%.4f"),
        'Predicted': st.column_config.NumberColumn("Predicted $/hr", format="$%.4f"),
    })

# ---------- Charts ----------
log_axes = st.toggle("Log scale axes", value=True)
axis_type = 'log' if log_axes else 'linear'
hover = ['Name', 'API Name', 'Instance Memory', 'vCPUs', TARGET]


def feature_chart(feature, other, other_value, unit):
    fig = px.scatter(data, x=feature, y=TARGET, hover_data=hover, opacity=0.5,
                     color_discrete_sequence=[POINT_COLOR])
    fig.data[0].name = 'Instances'
    fig.data[0].showlegend = True
    grid = np.geomspace(data[feature].min(), data[feature].max(), 100)
    curve = model.predict(pd.DataFrame({feature: grid, other: other_value})[FEATURES])
    fig.add_trace(go.Scatter(x=grid, y=curve, mode='lines', line=dict(color=MODEL_COLOR, width=2),
                             name=f'Model ({other} = {other_value:g})'))
    fig.add_trace(go.Scatter(x=[memory if feature == 'Instance Memory' else vcpus], y=[prediction],
                             mode='markers', name='Your input',
                             marker=dict(color=MODEL_COLOR, size=12, symbol='diamond',
                                         line=dict(color='white', width=2))))
    fig.update_xaxes(type=axis_type, title=f"{feature} ({unit})")
    fig.update_yaxes(type=axis_type, title="On-Demand cost ($/hr)")
    return style(fig)


tab_mem, tab_cpu, tab_fit, tab_dist, tab_data = st.tabs(
    ["Cost vs Memory", "Cost vs vCPUs", "Actual vs Predicted", "Cost distribution", "Data"])

with tab_mem:
    st.plotly_chart(feature_chart('Instance Memory', 'vCPUs', vcpus, 'GiB'), width='stretch')
    st.caption("Model line holds vCPUs at your sidebar input.")

with tab_cpu:
    st.plotly_chart(feature_chart('vCPUs', 'Instance Memory', memory, 'count'), width='stretch')
    st.caption("Model line holds Instance Memory at your sidebar input.")

with tab_fit:
    fit = X_test.assign(Actual=y_test, Predicted=y_pred_test).join(data[['Name', 'API Name']])
    fig = px.scatter(fit, x='Actual', y='Predicted', hover_data=['Name', 'API Name'] + FEATURES,
                     opacity=0.6, color_discrete_sequence=[POINT_COLOR])
    fig.data[0].name = 'Test instances'
    fig.data[0].showlegend = True
    lo, hi = fit[['Actual', 'Predicted']].min().min(), fit[['Actual', 'Predicted']].max().max()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', name='Perfect prediction',
                             line=dict(color='gray', dash='dash', width=2)))
    fig.update_xaxes(type=axis_type, title="Actual cost ($/hr)")
    fig.update_yaxes(type=axis_type, title="Predicted cost ($/hr)")
    st.plotly_chart(style(fig), width='stretch')
    st.caption(f"{len(fit)} held-out test instances (20% split, random_state=42).")

with tab_dist:
    fig = px.histogram(data, x=TARGET, nbins=40, color_discrete_sequence=[POINT_COLOR])
    fig.add_vline(x=prediction, line=dict(color=MODEL_COLOR, width=2),
                  annotation_text=f"Your prediction ${prediction:.4f}")
    fig.update_traces(marker_line=dict(color='white', width=1))
    fig.update_xaxes(title="On-Demand cost ($/hr)")
    fig.update_yaxes(title="Instances")
    st.plotly_chart(style(fig), width='stretch')

with tab_data:
    st.dataframe(data[['Name', 'API Name', 'Instance Memory', 'vCPUs', TARGET,
                       'Linux Reserved cost', 'Linux Spot Minimum cost',
                       'Windows On Demand cost', 'Windows Reserved cost']],
                 hide_index=True, width='stretch')
    st.download_button("Download cleaned data (CSV)", data.to_csv(index=False), "ec2_cleaned.csv", "text/csv")
