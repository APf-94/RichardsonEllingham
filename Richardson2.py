import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

# --- Seite konfigurieren ---
st.set_page_config(page_title="Richardson-Ellingham Diagram", layout="wide")

# --- Physikalische Konstanten & Phasenübergänge ---
PHASE_DATA = {
    "Na": {"Tm": 371, "Tb": 1156, "Hm": 2.6, "Hb": 97},
    "Mg": {"Tm": 923, "Tb": 1363, "Hm": 8.5, "Hb": 128},
    "Zn": {"Tm": 692, "Tb": 1180, "Hm": 7.3, "Hb": 115},
    "Ca": {"Tm": 1115, "Tb": 1757, "Hm": 8.5, "Hb": 154},
    "Al": {"Tm": 933, "Tb": 2792, "Hm": 10.7, "Hb": 294},
    "Li": {"Tm": 453, "Tb": 1603, "Hm": 3.0, "Hb": 147},
    "K":  {"Tm": 336, "Tb": 1032, "Hm": 2.3, "Hb": 77},
    "Fe": {"Tm": 1811, "Tb": 3134, "Hm": 13.8, "Hb": 340},
    "Ni": {"Tm": 1728, "Tb": 3186, "Hm": 17.5, "Hb": 370},
    "Mn": {"Tm": 1519, "Tb": 2334, "Hm": 12.9, "Hb": 221},
    "Ag": {"Tm": 1234, "Tb": 2435, "Hm": 11.3, "Hb": 250},
    "Si": {"Tm": 1687, "Tb": 3538, "Hm": 50.2, "Hb": 359},
    "Cr": {"Tm": 2180, "Tb": 2944, "Hm": 21.0, "Hb": 340},
    "Ti": {"Tm": 1941, "Tb": 3560, "Hm": 14.1, "Hb": 425},
    "V":  {"Tm": 2183, "Tb": 3680, "Hm": 21.5, "Hb": 460},
}

@st.cache_data
def load_data(file):
    """Lädt die Datei und bereinigt Spalten sowie Enthalpie-Werte."""
    df = pd.read_excel(file, skiprows=1)
    df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
    
    # Identifiziere alle relevanten Spalten
    h_cols = [c for c in df.columns if 'ΔfH' in c] # Findet ΔfH0° und ΔfH°
    s_col = [c for c in df.columns if 'S°' in c][0]
    
    for col in h_cols + [s_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def get_thermo_values(df, formula, state='cr'):
    """Sucht H und S mit Priorität auf 298K Werten und Fallbacks."""
    # Erweiterte Fallbacks für fehlende Tabellenwerte
    fallbacks = {
        "CaO": (-635.1, 39.75), "MgO": (-601.6, 26.9),
        "Fe3O4": (-1118.4, 146.1), "FeO": (-272.0, 57.6),
        "K2O": (-361.5, 94.1), "V2O3": (-1218.8, 98.3)
    }
    
    res = df[(df['Formula'] == formula) & (df['State'] == state)]
    if res.empty:
        res = df[df['Formula'] == formula].head(1)
    
    if not res.empty:
        # Prüfe beide H-Spalten (ΔfH° ist meist bei 298K befüllt)
        h_val = res['ΔfH°'].values[0] if 'ΔfH°' in res.columns else np.nan
        if pd.isna(h_val) and 'ΔfH0° kJ mol-1' in res.columns:
            h_val = res['ΔfH0° kJ mol-1'].values[0]
            
        s_col = [c for c in df.columns if 'S°' in c][0]
        s_val = res[s_col].values[0]
        
        if not pd.isna(h_val) and not pd.isna(s_val):
            return h_val, s_val / 1000.0
            
    if formula in fallbacks:
        h_f, s_f = fallbacks[formula]
        return h_f, s_f / 1000.0
    return None, None

def calculate_dg_with_phases(T_array, dH_298, dS_298, metal_key, n_m):
    dg_values = []
    for T in T_array:
        curr_H, curr_S = dH_298, dS_298
        if metal_key in PHASE_DATA:
            p = PHASE_DATA[metal_key]
            if T > p["Tm"]:
                curr_H -= n_m * p["Hm"] 
                curr_S -= n_m * (p["Hm"] / p["Tm"])
            if T > p["Tb"]:
                curr_H -= n_m * p["Hb"]
                curr_S -= n_m * (p["Hb"] / p["Tb"])
        dg_values.append(curr_H - T * curr_S)
    return np.array(dg_values)

# --- Hauptprogramm ---
st.title("Richardson-Ellingham Diagram Generator")
st.markdown("Data based on NBS-Tables. | by Dr. Andreas Pfeiffer")
st.markdown("Disclaimer: All information is provided without guarantee. No responsibility is taken for the accuracy, completeness, or timeliness of the content.")

# GitHub Link Korrektur
GITHUB_URL = "https://github.com/APf-94/RichardsonEllingham/raw/refs/heads/main/NBS_Tables_Library.xlsx"

st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload own NBS table (optional)", type=["xlsx"])

# Zentrales Laden
if uploaded_file:
    df = load_data(uploaded_file)
    st.sidebar.success("Using uploaded file")
else:
    try:
        df = load_data(GITHUB_URL)
        st.sidebar.success("Loaded default library from GitHub")
    except Exception as e:
        st.error(f"Error: Could not find the file on GitHub. {e}")
        st.stop()

# Reaktionen (Normiert auf 1 Mol O2)
rxn_dict = {
    "2Mg + O2 -> 2MgO": ("Mg", "MgO", 2, 2),
    "2Ca + O2 -> 2CaO": ("Ca", "CaO", 2, 2),
    "4/3Al + O2 -> 2/3Al2O3": ("Al", "Al2O3", 4/3, 2/3),
    "Ti + O2 -> TiO2": ("Ti", "TiO2", 1, 1),
    "Si + O2 -> SiO2": ("Si", "SiO2", 1, 1),
    "2Mn + O2 -> 2MnO": ("Mn", "MnO", 2, 2),
    "2Zn + O2 -> 2ZnO": ("Zn", "ZnO", 2, 2),
    "4/3Cr + O2 -> 2/3Cr2O3": ("Cr", "Cr2O3", 4/3, 2/3),
    "2Fe + O2 -> 2FeO": ("Fe", "FeO", 2, 2),
    "6FeO + O2 -> 2Fe3O4": ("FeO", "Fe3O4", 6, 2),
    "4Fe3O4 + O2 -> 6Fe2O3": ("Fe3O4", "Fe2O3", 4, 6),
    "2Ni + O2 -> 2NiO": ("Ni", "NiO", 2, 2),
    "2C + O2 -> 2CO": ("C", "CO", 2, 2),
    "C + O2 -> CO2": ("C", "CO2", 1, 1),
    "2CO + O2 -> 2CO2": ("CO", "CO2", 2, 2),
    "4Ag + O2 -> 2Ag2O": ("Ag", "Ag2O", 4, 2),
    "4Li + O2 -> 2Li2O": ("Li", "Li2O", 4, 2),
    "4Na + O2 -> 2Na2O": ("Na", "Na2O", 4, 2),
    "4K + O2 -> 2K2O": ("K", "K2O", 4, 2),
    "4/3V + O2 -> 2/3V2O3": ("V", "V2O3", 4/3, 2/3),
}

selected = st.sidebar.multiselect("Oxides:", list(rxn_dict.keys()), 
                                  default=["2Fe + O2 -> 2FeO", "2C + O2 -> 2CO", "4Ag + O2 -> 2Ag2O"])

st.sidebar.markdown("---")
log_h2 = st.sidebar.slider("H2 / H2O ratio (10^x)", -10.0, 10.0, 0.0, step=0.1)
log_co = st.sidebar.slider("CO / CO2 ratio (10^x)", -10.0, 10.0, 0.0, step=0.1)
h2_ratio, co_ratio = 10**log_h2, 10**log_co

T_celsius = np.linspace(0, 2000, 300)
T_kelvin = T_celsius + 273.15
R = 0.008314 

fig = go.Figure()

# O2 Referenz (S=205.1 J/molK)
_, s_o2 = get_thermo_values(df, "O2", "g")
if s_o2 is None: s_o2 = 0.2051

for name in selected:
    m_form, ox_form, n_m, n_ox = rxn_dict[name]
    h_m, s_m = get_thermo_values(df, m_form)
    h_ox, s_ox = get_thermo_values(df, ox_form)
    
    if h_m is not None and h_ox is not None:
        dH_reakt = (n_ox * h_ox) - (n_m * h_m)
        dS_reakt = (n_ox * s_ox) - (n_m * s_m + s_o2)
        dG_series = calculate_dg_with_phases(T_kelvin, dH_reakt, dS_reakt, m_form, n_m)
        fig.add_trace(go.Scatter(x=T_celsius, y=dG_series, name=name))

# Gaslinien
h_h2, s_h2 = get_thermo_values(df, "H2", "g")
h_h2o, s_h2o = get_thermo_values(df, "H2O", "g")
if h_h2o:
    dg_h2_eff = (2*h_h2o - 2*0) - T_kelvin*(2*s_h2o - 2*s_h2 - s_o2) + R * T_kelvin * np.log((1/h2_ratio)**2)
    fig.add_trace(go.Scatter(x=T_celsius, y=dg_h2_eff, name="H2/H2O", line=dict(dash='dash', color='blue')))

h_co, s_co = get_thermo_values(df, "CO", "g")
h_co2, s_co2 = get_thermo_values(df, "CO2", "g")For='red')))

fig.update_layout(
    title="Richardson-Ellingham Diagram (normalized to 1 Mol O₂)",
    xaxis_title="Temperature / °C",
    yaxis_title="ΔG° / kJ / mol O₂",
    height=800,
    template="plotly_white",
    yaxis=dict(range=[-1200, 100]) # Erhöht auf +100 für Silber
)

st.plotly_chart(fig, use_container_width=True)

