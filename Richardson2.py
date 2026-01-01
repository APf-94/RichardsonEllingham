import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

# --- Seite konfigurieren ---
st.set_page_config(page_title="Richardson-Ellingham Diagramm Profi", layout="wide")

# --- Physikalische Konstanten & Phasenübergänge ---
# Tm = Schmelzpunkt [K], Tb = Siedepunkt [K]
# Hm = Schmelzenthalpie [kJ/mol], Hb = Verdampfungsenthalpie [kJ/mol]
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
    """Lädt die Excel-Datei und bereinigt die Spaltennamen."""
    df = pd.read_excel(file, skiprows=1)
    df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
    # Spaltennamen in der NBS-Tabelle identifizieren
    h_col = [c for c in df.columns if 'ΔfH0°' in c][0]
    s_col = [c for c in df.columns if 'S°' in c][0]
    
    # In Zahlen umwandeln
    df[h_col] = pd.to_numeric(df[h_col], errors='coerce')
    df[s_col] = pd.to_numeric(df[s_col], errors='coerce')
    return df

def get_thermo_values(df, formula, state='cr'):
    """Extrahiert H und S aus dem Dataframe mit Fallbacks für Lücken."""
    # Spezielle Fallbacks für Calciumoxid und andere oft fehlende Werte in den NBS-Listen
    fallbacks = {
        "CaO": (-635.1, 39.75), 
        "MgO": (-601.6, 26.9),
        "Fe3O4": (-1118.4, 146.1),
        "FeO": (-272.0, 57.6)
    }
    
    res = df[(df['Formula'] == formula) & (df['State'] == state)]
    if res.empty:
        res = df[df['Formula'] == formula].head(1)
    
    if not res.empty:
        h_col = [c for c in df.columns if 'ΔfH0°' in c][0]
        s_col = [c for c in df.columns if 'S°' in c][0]
        h = res[h_col].values[0]
        s = res[s_col].values[0]
        
        if not pd.isna(h) and not pd.isna(s):
            return h, s / 1000.0  # S in kJ/K umrechnen
            
    # Falls nichts gefunden wurde, Fallback nutzen
    if formula in fallbacks:
        h_f, s_f = fallbacks[formula]
        return h_f, s_f / 1000.0
        
    return None, None

def calculate_dg_with_phases(T_array, dH_298, dS_298, metal_key, n_m):
    dg_values = []
    for T in T_array:
        # Startwerte bei 298K
        current_dH = dH_298
        current_dS = dS_298
        
        if metal_key in PHASE_DATA:
            p = PHASE_DATA[metal_key]
            # Schmelzpunkt-Korrektur
            if T > p["Tm"]:
                # dH und dS ändern sich so, dass dG am Schmelzpunkt stetig bleibt!
                # dG_neu = (H + Hm) - T * (S + Hm/Tm)
                current_dH -= n_m * p["Hm"] 
                current_dS -= n_m * (p["Hm"] / p["Tm"])
            
            # Siedepunkt-Korrektur
            if T > p["Tb"]:
                current_dH -= n_m * p["Hb"]
                current_dS -= n_m * (p["Hb"] / p["Tb"])
        
        dg_values.append(current_dH - T * current_dS)
    return np.array(dg_values)

# --- Hauptprogramm ---
st.title("Richardson-Ellingham Diagramm Generator")
st.markdown("Interaktive Thermodynamik basierend auf NBS-Tabellen.")

uploaded_file = st.sidebar.file_uploader("NBS Excel Datei hochladen", type=["xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)
    
    # Alle angefragten Reaktionen (Normiert auf 1 Mol O2)
    # Format: "Label": (Metall_Formel, Oxid_Formel, n_Metall, n_Oxid)
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

    # Sidebar Auswahl
    selected = st.sidebar.multiselect("Oxide einblenden:", list(rxn_dict.keys()), 
                                      default=["2Fe + O2 -> 2FeO", "2C + O2 -> 2CO", "2CO + O2 -> 2CO2"])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Gleichgewichts-Gase (logarithmisch)")

    # Wir wählen den Exponenten von -10 bis +10
    log_h2 = st.sidebar.slider("H2 / H2O Verhältnis (10^x)", -10.0, 10.0, 0.0, step=0.1)
    h2_ratio = 10**log_h2

    log_co = st.sidebar.slider("CO / CO2 Verhältnis (10^x)", -10.0, 10.0, 0.0, step=0.1)
    co_ratio = 10**log_co

    # Anzeige der aktuellen Werte für den User
    st.sidebar.write(f"H2/H2O: **{h2_ratio:.2e}**")
    st.sidebar.write(f"CO/CO2: **{co_ratio:.2e}**")

    # Berechnungsbereich
    T_celsius = np.linspace(0, 2000, 200)
    T_kelvin = T_celsius + 273.15
    R = 0.008314 # kJ/mol*K
    
    fig = go.Figure()

    # O2 Referenz
    h_o2, s_o2 = get_thermo_values(df, "O2", "g")
    if h_o2 is None: h_o2, s_o2 = 0.0, 0.2051

    # Zeichnen der Metall-Oxid Linien
    for name in selected:
        m_form, ox_form, n_m, n_ox = rxn_dict[name]
        h_m, s_m = get_thermo_values(df, m_form)
        h_ox, s_ox = get_thermo_values(df, ox_form)
        
        if h_m is not None and h_ox is not None:
            dH_reakt = (n_ox * h_ox) - (n_m * h_m + 0)
            dS_reakt = (n_ox * s_ox) - (n_m * s_m + s_o2)
            
            dG_series = calculate_dg_with_phases(T_kelvin, dH_reakt, dS_reakt, m_form, n_m)
            fig.add_trace(go.Scatter(x=T_celsius, y=dG_series, name=name, mode='lines'))

    # Gaslinie: 2H2 + O2 -> 2H2O
    h_h2, s_h2 = get_thermo_values(df, "H2", "g")
    h_h2o, s_h2o = get_thermo_values(df, "H2O", "g")
    if h_h2o:
        dg_h2_std = (2*h_h2o - 2*0 - 0) - T_kelvin*(2*s_h2o - 2*s_h2 - s_o2)
        dg_h2_eff = dg_h2_std + R * T_kelvin * np.log((1/h2_ratio)**2)
        fig.add_trace(go.Scatter(x=T_celsius, y=dg_h2_eff, name=f"H2/H2O (r={h2_ratio:.1e})", 
                                 line=dict(dash='dash', color='blue')))

    # Gaslinie: 2CO + O2 -> 2CO2
    h_co, s_co = get_thermo_values(df, "CO", "g")
    h_co2, s_co2 = get_thermo_values(df, "CO2", "g")
    if h_co2:
        dg_co_std = (2*h_co2 - 2*h_co - 0) - T_kelvin*(2*s_co2 - 2*s_co - s_o2)
        dg_co_eff = dg_co_std + R * T_kelvin * np.log((1/co_ratio)**2)
        fig.add_trace(go.Scatter(x=T_celsius, y=dg_co_eff, name=f"CO/CO2 (r={co_ratio:.1e})", 
                                 line=dict(dash='dot', color='red')))

    # Layout Verschönerung
    fig.update_layout(
        title="Richardson-Ellingham Diagramm (auf 1 Mol O₂ normiert)",
        xaxis_title="Temperatur [°C]",
        yaxis_title="ΔG° [kJ / mol O₂]",
        hovermode="x unified",
        height=800,
        template="plotly_white"
    )
    fig.update_yaxes(range=[-1200, 0]) # Typischer Ellingham Bereich

    st.plotly_chart(fig, use_container_width=True)

    # Export Sektion
    img_bytes = fig.to_image(format="png", width=1200, height=800)
    st.sidebar.download_button("Diagramm als PNG exportieren", data=img_bytes, 
                               file_name="ellingham_diagramm.png", mime="image/png")

else:
    st.info("Bitte lade die Datei 'NBS_Tables Library.xlsx' hoch, um die Berechnungen zu starten.")