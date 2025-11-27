# single_step_app.py
# Single-step Streamlit interface: upload new water data -> get WQI & Safe/Unsafe predictions immediately.
# Saves/loads RandomForest models in ./models and trains from groundwater.csv if models not found.

# ------------------- Silence noisy warnings -------------------
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)

# ------------------- Imports -------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# ------------------- Config -------------------
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)
REG_FILE = MODEL_DIR / "rf_wqi_regressor.pkl"
CLF_FILE = MODEL_DIR / "rf_wqi_classifier.pkl"
TRAIN_CSV = "groundwater.csv"   # used to train if models missing

# Candidate features to use (edit to match your CSV if needed)
CANDIDATE_FEATURES = ['pH','E.C','TDS','Cl','HCO3','CO3','Na','Ca','Mg','SO4','K','NO3','Fe','Mn']

st.set_page_config(page_title="Single-step WQI Predictor", layout="wide")
st.title("Single-step Groundwater WQI & Safety Predictor")

st.markdown(
    """
    **How to use (single step):**
    1. Upload a CSV of new wells (columns like `pH`, `E.C`, `TDS`, `Cl`, `HCO3`, ...).  
    2. Click **Predict** — the app will load or train models automatically and return WQI & Safe/Unsafe results immediately.  
    OR use **Single sample** to enter one well and Predict instantly.
    """
)

# ------------------- Utility functions -------------------
def compute_approx_wqi(df):
    S = {'pH':8.5, 'TDS':500, 'E.C':750, 'Cl':250, 'HCO3':300}
    def q(series, key):
        return (series.fillna(0) / S[key]) * 100 if key in S else pd.Series(0, index=series.index)
    Q_ph  = q(df['pH'], 'pH')  if 'pH' in df.columns else 0
    Q_tds = q(df['TDS'], 'TDS') if 'TDS' in df.columns else 0
    Q_ec  = q(df['E.C'], 'E.C') if 'E.C' in df.columns else 0
    Q_cl  = q(df['Cl'], 'Cl')  if 'Cl' in df.columns else 0
    Q_hco3= q(df['HCO3'],'HCO3') if 'HCO3' in df.columns else 0
    w_ph, w_tds, w_ec, w_cl, w_hco3 = 4,3,3,2,2
    denom = w_ph + w_tds + w_ec + w_cl + w_hco3
    df['WQI'] = (Q_ph*w_ph + Q_tds*w_tds + Q_ec*w_ec + Q_cl*w_cl + Q_hco3*w_hco3) / denom
    return df

def train_models(train_csv=TRAIN_CSV, features=None):
    if not Path(train_csv).exists():
        raise FileNotFoundError(f"Training CSV '{train_csv}' not found. Upload it to app folder or provide models in ./models.")
    df_train = pd.read_csv(train_csv)
    # determine features
    if features is None:
        features = [c for c in CANDIDATE_FEATURES if c in df_train.columns]
    if len(features) < 3:
        raise ValueError("Not enough features found in training CSV for training. Check column names.")
    # numeric conversion
    for f in features:
        df_train[f] = pd.to_numeric(df_train[f], errors='coerce')
    if 'WQI' not in df_train.columns:
        df_train = compute_approx_wqi(df_train)
    train_df = df_train[features + ['WQI']].copy().dropna(subset=['WQI'])
    for f in features:
        train_df[f] = train_df[f].fillna(train_df[f].median())
    X = train_df[features].values
    y = train_df['WQI'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    rf_reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_train)
    # save reg
    joblib.dump(rf_reg, REG_FILE)
    # classifier
    train_df['WQI_Class'] = np.where(train_df['WQI'] < 100, 1, 0)
    Xc = train_df[features].values
    yc = train_df['WQI_Class'].values
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
    rf_clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_clf.fit(Xc_train, yc_train)
    joblib.dump(rf_clf, CLF_FILE)
    # metrics
    mse = mean_squared_error(y_test, rf_reg.predict(X_test))
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, rf_reg.predict(X_test))
    acc = accuracy_score(yc_test, rf_clf.predict(Xc_test))
    return {'features':features, 'medians':{f:train_df[f].median() for f in features}, 'rmse':rmse, 'r2':r2, 'acc':acc}

def load_or_train_models():
    if REG_FILE.exists() and CLF_FILE.exists():
        reg = joblib.load(REG_FILE)
        clf = joblib.load(CLF_FILE)
        # try infer features from training CSV if available
        if Path(TRAIN_CSV).exists():
            tmp = pd.read_csv(TRAIN_CSV)
            feats = [c for c in CANDIDATE_FEATURES if c in tmp.columns]
            med = {f: pd.to_numeric(tmp[f], errors='coerce').median() for f in feats}
        else:
            feats = [c for c in CANDIDATE_FEATURES if c in df_cols]  # fallback
            med = {f:0.0 for f in feats}
        return reg, clf, feats, med, None
    else:
        # train and save
        info = train_models()
        reg = joblib.load(REG_FILE)
        clf = joblib.load(CLF_FILE)
        return reg, clf, info['features'], info['medians'], info

def predict_df(df_new, features, medians, reg, clf):
    df = df_new.copy()
    for f in features:
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors='coerce')
        else:
            df[f] = np.nan
    for f in features:
        df[f] = df[f].fillna(medians.get(f, 0.0))
    X = df[features].values
    df['WQI_Pred'] = reg.predict(X)
    df['Safe_Pred'] = clf.predict(X)
    df['Safe_Label'] = df['Safe_Pred'].map({1:'Safe', 0:'Unsafe'})
    return df

# ------------------- App logic -------------------
# load dataframe columns early to use as fallback
df_cols = []
if Path(TRAIN_CSV).exists():
    try:
        tmp = pd.read_csv(TRAIN_CSV, nrows=5)
        df_cols = list(tmp.columns)
    except:
        df_cols = []

with st.spinner("Loading or training models (if needed)..."):
    try:
        reg, clf, features, medians, train_info = load_or_train_models()
        st.success("Models ready.")
    except Exception as e:
        st.error(f"Model load/train error: {e}")
        st.stop()

st.write(f"Features used by the model: **{features}**")

# Input section: Upload CSV OR single sample
st.header("Upload new dataset (CSV) — single step prediction")
uploaded = st.file_uploader("Upload CSV with samples (columns like pH, E.C, TDS, Cl, HCO3, ...)", type=['csv'])

col1, col2 = st.columns(2)
with col1:
    st.write("Or enter single sample values below (leave blank to use medians):")

# single-sample inputs
sample = {}
for i, f in enumerate(features):
    if i % 2 == 0:
        sample[f] = st.text_input(f"{f}", value="")
    else:
        sample[f] = st.text_input(f"{f}", value="")

# Predict button (single step)
if st.button("Predict"):
    try:
        if uploaded is not None:
            new_df = pd.read_csv(uploaded)
            res = predict_df(new_df, features, medians, reg, clf)
            st.success(f"Predicted {len(res)} samples.")
            st.dataframe(res[[*features, 'WQI_Pred', 'Safe_Label']].head(20))
            csv_out = res.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions (CSV)", csv_out, file_name="predictions.csv")
            # show counts
            st.write("Counts (Safe/Unsafe):")
            st.write(res['Safe_Label'].value_counts().to_frame(name='count'))
            # map if lat/long present
            if 'lat_gis' in res.columns and 'long_gis' in res.columns:
                st.map(res.rename(columns={'lat_gis':'lat','long_gis':'lon'})[['lat','lon']].dropna())
        else:
            # build single sample df
            single = {}
            for f in features:
                val = sample.get(f)
                try:
                    single[f] = float(val) if val not in (None, "") else medians.get(f, 0.0)
                except:
                    single[f] = medians.get(f, 0.0)
            single_df = pd.DataFrame([single])
            res = predict_df(single_df, features, medians, reg, clf)
            wqi = float(res.loc[0,'WQI_Pred'])
            label = str(res.loc[0,'Safe_Label'])
            st.success(f"Predicted WQI = {wqi:.2f} → {label}")
            st.json(res[['WQI_Pred','Safe_Label']].iloc[0].to_dict())
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Note: If models were trained automatically, training used 'groundwater.csv' present in the app folder. Edit candidate features in the code if your CSV uses different column names.")
