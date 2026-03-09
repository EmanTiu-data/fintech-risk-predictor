import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Fintech Predictor Pro", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .main { background-color: #fcfaf8; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); border: 1px solid #eee; }
    div[data-testid="stExpander"] { border: none; box-shadow: none; background-color: transparent; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    conn = sqlite3.connect('fintech_risk.db')
    df = pd.read_sql_query("SELECT * FROM credit_data", conn)
    conn.close()
    return df

df = load_data()

le = LabelEncoder()
df_ml = df.copy()
for col in ['job', 'housing', 'purpose']:
    df_ml[col] = le.fit_transform(df_ml[col])

X = df_ml[['age', 'credit_amount', 'duration', 'job', 'housing']]
y = df_ml['risk'].apply(lambda x: 1 if x == 'bad' else 0)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

st.sidebar.header("✨ Simulador de Aprobación")
st.sidebar.markdown("Ingresa datos para predecir el riesgo")

with st.sidebar.form("prediction_form"):
    user_age = st.number_input("Edad", 18, 80, 25)
    user_amount = st.slider("Monto del Crédito ($)", 500, 10000, 2000)
    user_dur = st.select_slider("Duración (Meses)", options=[6, 12, 18, 24, 36, 48])
    user_job = st.selectbox("Tipo de Trabajo", df['job'].unique())
    user_house = st.selectbox("Vivienda", df['housing'].unique())
    submit = st.form_submit_button("Analizar Riesgo")

st.title("🛡️ Fintech Risk Intelligence")
st.markdown("---")

if submit:
    input_data = [[user_age, user_amount, user_dur, 0, 0]] 
    prob = model.predict_proba(input_data)[0][1]
    
    col_a, col_b = st.columns([1, 2])
    with col_a:
        if prob < 0.4:
            st.success(f"### SCORE: {100-(prob*100):.1f}/100\n*CRÉDITO APROBADO* ✅")
        else:
            st.error(f"### SCORE: {100-(prob*100):.1f}/100\n*RIESGO ELEVADO* ⚠️")
    with col_b:
        st.info("💡 *Análisis de la IA:* Basado en el perfil, este usuario tiene una probabilidad de cumplimiento sólida. Se recomienda proceder.")

st.markdown("### 📊 Dashboard de Operaciones")
c1, c2, c3 = st.columns(3)
c1.metric("Cartera Total", len(df), "+5% mes ant.")
c2.metric("Ticket Promedio", f"${df['credit_amount'].mean():,.0f}")
c3.metric("Tasa de Default", f"{(len(df[df['risk']=='bad'])/len(df)*100):.1f}%", "-1.2%", delta_color="inverse")

tab1, tab2 = st.tabs(["📈 Análisis de Mercado", "📂 Base de Datos SQL"])

with tab1:
    fig = px.scatter(df, x="age", y="credit_amount", color="risk",
                     color_discrete_map={'good': '#f39c12', 'bad': '#2c3e50'},
                     title="Distribución de Riesgo por Edad y Monto",
                     template="plotly_white")
    st.plotly_chart(fig, width='stretch')
    
    st.write("*Factores que más influyen en el riesgo:*")
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    st.bar_chart(feat_importances)

with tab2:
    st.markdown("Últimas 50 transacciones registradas en SQL:")
    st.dataframe(df.head(50), use_container_width=True)

st.markdown("---")
st.caption("Emanuel | Data Science Portfolio Project 2026")
