import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Metallurgy AI Lab",
    page_icon="🔬",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
}

.subtitle {
    font-size: 20px;
    text-align: center;
    margin-bottom: 30px;
}

.metric-card {
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #ddd;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# GENERATE METALLURGY DATASET
# ============================================================

@st.cache_data
def generate_dataset(n=2500):

    np.random.seed(42)

    # --------------------------------------------------------
    # Chemical composition
    # --------------------------------------------------------

    carbon = np.random.uniform(0.05, 1.2, n)
    manganese = np.random.uniform(0.2, 2.0, n)
    silicon = np.random.uniform(0.1, 1.2, n)
    chromium = np.random.uniform(0, 3.0, n)
    nickel = np.random.uniform(0, 2.5, n)
    molybdenum = np.random.uniform(0, 1.0, n)

    # --------------------------------------------------------
    # Heat treatment
    # --------------------------------------------------------

    heat_treatment = np.random.choice(
        [
            "Annealing",
            "Normalizing",
            "Quenching",
            "Quenching + Tempering"
        ],
        n
    )

    cooling_rate = np.random.uniform(1, 100, n)

    tempering_temperature = np.random.uniform(200, 700, n)

    # --------------------------------------------------------
    # Metallurgical relationships
    # --------------------------------------------------------

    treatment_strength = {
        "Annealing": 0,
        "Normalizing": 70,
        "Quenching": 250,
        "Quenching + Tempering": 180
    }

    treatment_hardness = {
        "Annealing": 0,
        "Normalizing": 5,
        "Quenching": 20,
        "Quenching + Tempering": 15
    }

    # --------------------------------------------------------
    # Tensile strength
    # --------------------------------------------------------

    strength = (
        280
        + 420 * carbon
        + 45 * manganese
        + 25 * silicon
        + 30 * chromium
        + 15 * nickel
        + 35 * molybdenum
        + np.array(
            [treatment_strength[x] for x in heat_treatment]
        )
        + 1.2 * cooling_rate
        - 0.08 * (tempering_temperature - 450) ** 2 / 100
        + np.random.normal(0, 35, n)
    )

    # --------------------------------------------------------
    # Hardness
    # --------------------------------------------------------

    hardness = (
        80
        + 45 * carbon
        + 8 * manganese
        + 7 * chromium
        + 10 * molybdenum
        + np.array(
            [treatment_hardness[x] for x in heat_treatment]
        )
        + 0.15 * cooling_rate
        - 0.025 * (tempering_temperature - 450)
        + np.random.normal(0, 5, n)
    )

    # --------------------------------------------------------
    # Yield strength
    # --------------------------------------------------------

    yield_strength = (
        strength * 0.78
        + np.random.normal(0, 20, n)
    )

    # --------------------------------------------------------
    # Elongation
    # --------------------------------------------------------

    elongation = (
        32
        - 12 * carbon
        - 2 * manganese
        - 1.5 * chromium
        - 0.04 * cooling_rate
        + np.where(
            heat_treatment == "Annealing",
            7,
            0
        )
        + np.random.normal(0, 2, n)
    )

    # --------------------------------------------------------
    # Avoid unrealistic negative values
    # --------------------------------------------------------

    strength = np.maximum(strength, 200)
    yield_strength = np.maximum(yield_strength, 150)
    hardness = np.maximum(hardness, 80)
    elongation = np.maximum(elongation, 2)

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame({

        "Carbon (%)": carbon,

        "Manganese (%)": manganese,

        "Silicon (%)": silicon,

        "Chromium (%)": chromium,

        "Nickel (%)": nickel,

        "Molybdenum (%)": molybdenum,

        "Heat Treatment": heat_treatment,

        "Cooling Rate (°C/s)": cooling_rate,

        "Tempering Temperature (°C)": tempering_temperature,

        "Yield Strength (MPa)": yield_strength,

        "Tensile Strength (MPa)": strength,

        "Hardness (HV)": hardness,

        "Elongation (%)": elongation
    })

    return df


df = generate_dataset()


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def train_models(data):

    # --------------------------------------------------------
    # Input features
    # --------------------------------------------------------

    X = data[
        [
            "Carbon (%)",
            "Manganese (%)",
            "Silicon (%)",
            "Chromium (%)",
            "Nickel (%)",
            "Molybdenum (%)",
            "Cooling Rate (°C/s)",
            "Tempering Temperature (°C)"
        ]
    ].copy()

    # --------------------------------------------------------
    # One-hot encode heat treatment
    # --------------------------------------------------------

    treatment_encoded = pd.get_dummies(
        data["Heat Treatment"],
        prefix="Treatment"
    )

    X = pd.concat(
        [
            X,
            treatment_encoded
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    y_strength = data["Tensile Strength (MPa)"]

    y_hardness = data["Hardness (HV)"]

    y_elongation = data["Elongation (%)"]

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, ys_train, ys_test = train_test_split(
        X,
        y_strength,
        test_size=0.2,
        random_state=42
    )

    _, _, yh_train, yh_test = train_test_split(
        X,
        y_hardness,
        test_size=0.2,
        random_state=42
    )

    _, _, ye_train, ye_test = train_test_split(
        X,
        y_elongation,
        test_size=0.2,
        random_state=42
    )

    # --------------------------------------------------------
    # Random Forest models
    # --------------------------------------------------------

    strength_model = RandomForestRegressor(
        n_estimators=250,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    hardness_model = RandomForestRegressor(
        n_estimators=250,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    elongation_model = RandomForestRegressor(
        n_estimators=250,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    # --------------------------------------------------------
    # Train models
    # --------------------------------------------------------

    strength_model.fit(
        X_train,
        ys_train
    )

    hardness_model.fit(
        X_train,
        yh_train
    )

    elongation_model.fit(
        X_train,
        ye_train
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    strength_pred = strength_model.predict(X_test)

    hardness_pred = hardness_model.predict(X_test)

    elongation_pred = elongation_model.predict(X_test)

    # --------------------------------------------------------
    # Model metrics
    # --------------------------------------------------------

    metrics = {

        "Strength R2":
            r2_score(
                ys_test,
                strength_pred
            ),

        "Strength MAE":
            mean_absolute_error(
                ys_test,
                strength_pred
            ),

        "Hardness R2":
            r2_score(
                yh_test,
                hardness_pred
            ),

        "Hardness MAE":
            mean_absolute_error(
                yh_test,
                hardness_pred
            ),

        "Elongation R2":
            r2_score(
                ye_test,
                elongation_pred
            ),

        "Elongation MAE":
            mean_absolute_error(
                ye_test,
                elongation_pred
            )
    }

    return (
        strength_model,
        hardness_model,
        elongation_model,
        X.columns,
        metrics
    )


(
    strength_model,
    hardness_model,
    elongation_model,
    feature_columns,
    metrics
) = train_models(df)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔬 Metallurgy AI Lab")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📊 Dataset Explorer",
        "🤖 Property Predictor",
        "🔥 Heat Treatment",
        "🧪 Metallurgical Calculator",
        "📈 Model Performance"
    ]
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">'
        '🔬 Metallurgy AI Lab'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-Based Steel Property Prediction & Heat Treatment Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Dashboard metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Dataset Samples",
        f"{len(df):,}"
    )

    col2.metric(
        "Features",
        "8+"
    )

    col3.metric(
        "ML Models",
        "3"
    )

    col4.metric(
        "Application",
        "Metallurgy"
    )

    st.divider()

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    st.subheader("🎯 Project Objective")

    st.write(
        """
        This application combines metallurgical engineering concepts
        with machine learning to analyze steel composition,
        heat treatment and cooling conditions.
        """
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    st.subheader("⚙️ What can this application do?")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        ### 🧪 Chemical Composition

        Analyze the influence of:

        - Carbon
        - Manganese
        - Silicon
        - Chromium
        - Nickel
        - Molybdenum
        """)

    with col2:

        st.markdown("""
        ### 🔥 Heat Treatment

        Analyze:

        - Annealing
        - Normalizing
        - Quenching
        - Quenching + Tempering
        - Cooling rate
        - Tempering temperature
        """)

    st.divider()

    st.info(
        "Use the sidebar to explore the dataset, "
        "make predictions and analyze heat treatment."
    )


# ============================================================
# DATASET EXPLORER
# ============================================================

elif page == "📊 Dataset Explorer":

    st.title("📊 Dataset Explorer")

    st.write(
        "Explore the generated steel metallurgy dataset."
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    st.dataframe(
        df,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Statistical summary
    # --------------------------------------------------------

    st.subheader("📌 Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    # --------------------------------------------------------
    # Tensile strength distribution
    # --------------------------------------------------------

    st.subheader("📈 Tensile Strength Distribution")

    fig = px.histogram(
        df,
        x="Tensile Strength (MPa)",
        nbins=40,
        title="Distribution of Tensile Strength"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Heat treatment vs strength
    # --------------------------------------------------------

    st.subheader("🔥 Heat Treatment vs Strength")

    fig = px.box(
        df,
        x="Heat Treatment",
        y="Tensile Strength (MPa)",
        title="Heat Treatment Influence on Tensile Strength"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Correlation matrix
    # --------------------------------------------------------

    st.subheader("🧮 Correlation Matrix")

    numeric_df = df.select_dtypes(
        include=np.number
    )

    correlation = numeric_df.corr()

    fig = px.imshow(
        correlation,
        text_auto=True,
        title="Feature Correlation"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# PROPERTY PREDICTOR
# ============================================================

elif page == "🤖 Property Predictor":

    st.title("🤖 Steel Property Predictor")

    st.write(
        "Enter the chemical composition and heat-treatment "
        "parameters to predict mechanical properties."
    )

    # --------------------------------------------------------
    # Chemical composition
    # --------------------------------------------------------

    st.subheader("🧪 Chemical Composition")

    col1, col2, col3 = st.columns(3)

    with col1:

        carbon = st.number_input(
            "Carbon (%)",
            min_value=0.01,
            max_value=1.5,
            value=0.40,
            step=0.01
        )

        manganese = st.number_input(
            "Manganese (%)",
            min_value=0.0,
            max_value=3.0,
            value=0.80,
            step=0.01
        )

        silicon = st.number_input(
            "Silicon (%)",
            min_value=0.0,
            max_value=2.0,
            value=0.25,
            step=0.01
        )

    with col2:

        chromium = st.number_input(
            "Chromium (%)",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.01
        )

        nickel = st.number_input(
            "Nickel (%)",
            min_value=0.0,
            max_value=5.0,
            value=0.20,
            step=0.01
        )

        molybdenum = st.number_input(
            "Molybdenum (%)",
            min_value=0.0,
            max_value=2.0,
            value=0.15,
            step=0.01
        )

    with col3:

        treatment = st.selectbox(
            "Heat Treatment",
            [
                "Annealing",
                "Normalizing",
                "Quenching",
                "Quenching + Tempering"
            ]
        )

        cooling_rate = st.number_input(
            "Cooling Rate (°C/s)",
            min_value=0.1,
            max_value=150.0,
            value=20.0
        )

        tempering_temperature = st.number_input(
            "Tempering Temperature (°C)",
            min_value=100.0,
            max_value=750.0,
            value=500.0
        )

    st.divider()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Properties",
        type="primary"
    ):

        input_data = pd.DataFrame({

            "Carbon (%)": [carbon],

            "Manganese (%)": [manganese],

            "Silicon (%)": [silicon],

            "Chromium (%)": [chromium],

            "Nickel (%)": [nickel],

            "Molybdenum (%)": [molybdenum],

            "Cooling Rate (°C/s)": [cooling_rate],

            "Tempering Temperature (°C)": [
                tempering_temperature
            ]
        })

        # ----------------------------------------------------
        # Treatment columns
        # ----------------------------------------------------

        treatment_columns = [
            "Treatment_Annealing",
            "Treatment_Normalizing",
            "Treatment_Quenching",
            "Treatment_Quenching + Tempering"
        ]

        for column in treatment_columns:

            input_data[column] = 0

        input_data[
            "Treatment_" + treatment
        ] = 1

        # ----------------------------------------------------
        # Match training feature order
        # ----------------------------------------------------

        input_data = input_data[
            feature_columns
        ]

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predicted_strength = strength_model.predict(
            input_data
        )[0]

        predicted_hardness = hardness_model.predict(
            input_data
        )[0]

        predicted_elongation = elongation_model.predict(
            input_data
        )[0]

        predicted_yield = predicted_strength * 0.78

        # ----------------------------------------------------
        # Display predictions
        # ----------------------------------------------------

        st.subheader("🎯 Predicted Properties")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Yield Strength",
            f"{predicted_yield:.1f} MPa"
        )

        col2.metric(
            "Tensile Strength",
            f"{predicted_strength:.1f} MPa"
        )

        col3.metric(
            "Hardness",
            f"{predicted_hardness:.1f} HV"
        )

        col4.metric(
            "Elongation",
            f"{predicted_elongation:.1f} %"
        )

        st.divider()

        # ----------------------------------------------------
        # Metallurgical interpretation
        # ----------------------------------------------------

        st.subheader("🔬 Metallurgical Interpretation")

        if treatment == "Quenching":

            st.info(
                """
                Quenching produces rapid cooling from the
                austenitic region. This promotes martensitic
                transformation and generally increases hardness
                and strength while reducing ductility.
                """
            )

        elif treatment == "Annealing":

            st.info(
                """
                Annealing involves slow cooling and generally
                promotes a softer and more ductile microstructure.
                """
            )

        elif treatment == "Normalizing":

            st.info(
                """
                Normalizing uses air cooling after austenitization.
                It generally produces a finer microstructure than
                annealing and improves strength.
                """
            )

        else:

            st.info(
                """
                Quenching followed by tempering provides a
                compromise between strength and toughness.
                Tempering reduces the brittleness associated
                with as-quenched martensite.
                """
            )


# ============================================================
# HEAT TREATMENT
# ============================================================

elif page == "🔥 Heat Treatment":

    st.title("🔥 Heat Treatment Analysis")

    treatment = st.selectbox(
        "Select Heat Treatment",
        [
            "Annealing",
            "Normalizing",
            "Quenching",
            "Quenching + Tempering"
        ]
    )

    # --------------------------------------------------------
    # Treatment information
    # --------------------------------------------------------

    treatment_info = {

        "Annealing": {
            "cooling": "Furnace cooling",
            "structure": "Ferrite + Pearlite",
            "property": "Soft and ductile"
        },

        "Normalizing": {
            "cooling": "Air cooling",
            "structure": "Fine Ferrite + Pearlite",
            "property": "Higher strength than annealing"
        },

        "Quenching": {
            "cooling": "Water/Oil",
            "structure": "Martensite",
            "property": "High hardness and strength"
        },

        "Quenching + Tempering": {
            "cooling": "Quenching followed by tempering",
            "structure": "Tempered Martensite",
            "property": "High strength + improved toughness"
        }
    }

    info = treatment_info[treatment]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Cooling",
        info["cooling"]
    )

    col2.metric(
        "Typical Structure",
        info["structure"]
    )

    col3.metric(
        "Main Property",
        info["property"]
    )

    st.divider()

    # --------------------------------------------------------
    # Temperature-time curve
    # --------------------------------------------------------

    st.subheader("🌡️ Conceptual Temperature-Time Curve")

    if treatment == "Annealing":

        temperature = [
            25,
            400,
            850,
            850,
            700,
            500,
            300,
            25
        ]

    elif treatment == "Normalizing":

        temperature = [
            25,
            400,
            850,
            850,
            700,
            500,
            300,
            25
        ]

    elif treatment == "Quenching":

        temperature = [
            25,
            400,
            850,
            850,
            300,
            100,
            25,
            25
        ]

    else:

        temperature = [
            25,
            400,
            850,
            850,
            300,
            25,
            500,
            500,
            25
        ]

    time = list(range(len(temperature)))

    curve_df = pd.DataFrame({
        "Time": time,
        "Temperature": temperature
    })

    fig = px.line(
        curve_df,
        x="Time",
        y="Temperature",
        markers=True,
        title=treatment
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Metallurgical explanation
    # --------------------------------------------------------

    st.subheader("🔬 Metallurgical Explanation")

    if treatment == "Annealing":

        st.write(
            """
            In annealing, steel is heated above the critical
            transformation temperature and then cooled slowly,
            generally inside the furnace. This reduces hardness,
            relieves internal stresses and improves ductility.
            """
        )

    elif treatment == "Normalizing":

        st.write(
            """
            Normalizing consists of heating steel into the
            austenitic region followed by cooling in air.
            The relatively faster cooling compared with annealing
            produces a finer ferrite-pearlite structure.
            """
        )

    elif treatment == "Quenching":

        st.write(
            """
            Quenching rapidly cools austenitized steel.
            The high cooling rate suppresses diffusional
            transformations and can produce martensite.
            """
        )

    else:

        st.write(
            """
            Quenching produces hard martensite, but the structure
            can be brittle. Tempering at an intermediate temperature
            reduces brittleness and improves toughness.
            """
        )


# ============================================================
# METALLURGICAL CALCULATOR
# ============================================================

elif page == "🧪 Metallurgical Calculator":

    st.title("🧪 Metallurgical Calculator")

    # --------------------------------------------------------
    # Carbon Equivalent
    # --------------------------------------------------------

    st.subheader("🔥 Carbon Equivalent Calculator")

    st.write(
        """
        Using the commonly used IIW-type carbon equivalent:
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        C = st.number_input(
            "Carbon C (%)",
            0.0,
            2.0,
            0.40
        )

        Mn = st.number_input(
            "Manganese Mn (%)",
            0.0,
            3.0,
            0.80
        )

        Cr = st.number_input(
            "Chromium Cr (%)",
            0.0,
            5.0,
            1.0
        )

    with col2:

        Mo = st.number_input(
            "Molybdenum Mo (%)",
            0.0,
            2.0,
            0.15
        )

        V = st.number_input(
            "Vanadium V (%)",
            0.0,
            2.0,
            0.05
        )

    with col3:

        Ni = st.number_input(
            "Nickel Ni (%)",
            0.0,
            5.0,
            0.20
        )

        Cu = st.number_input(
            "Copper Cu (%)",
            0.0,
            5.0,
            0.10
        )

    # --------------------------------------------------------
    # IIW Carbon Equivalent Formula
    # --------------------------------------------------------

    CE = (
        C
        + Mn / 6
        + (Cr + Mo + V) / 5
        + (Ni + Cu) / 15
    )

    st.metric(
        "Carbon Equivalent",
        f"{CE:.3f}"
    )

    if CE < 0.40:

        st.success(
            "Relatively good weldability based on CE."
        )

    elif CE < 0.60:

        st.warning(
            "Moderate hardenability/weldability considerations."
        )

    else:

        st.error(
            "Higher carbon equivalent — preheating and "
            "welding procedure control may be important."
        )

    st.divider()

    # ========================================================
    # Rule of Mixtures Calculator
    # ========================================================

    st.subheader("📦 Rule of Mixtures Calculator")

    Vf = st.slider(
        "Volume fraction of reinforcement",
        0.0,
        1.0,
        0.30
    )

    Vm = 1 - Vf

    Ef = st.number_input(
        "Reinforcement Modulus (GPa)",
        1.0,
        1000.0,
        70.0
    )

    Em = st.number_input(
        "Matrix Modulus (GPa)",
        0.1,
        500.0,
        3.0
    )

    # --------------------------------------------------------
    # Composite modulus
    # --------------------------------------------------------

    Ec = Vf * Ef + Vm * Em

    st.metric(
        "Composite Elastic Modulus",
        f"{Ec:.2f} GPa"
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.title("📈 Machine Learning Model Performance")

    st.write(
        """
        Random Forest regression models were trained to predict
        tensile strength, hardness and elongation.
        """
    )

    # --------------------------------------------------------
    # R2 metrics
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Strength R²",
        f"{metrics['Strength R2']:.3f}"
    )

    col2.metric(
        "Hardness R²",
        f"{metrics['Hardness R2']:.3f}"
    )

    col3.metric(
        "Elongation R²",
        f"{metrics['Elongation R2']:.3f}"
    )

    st.divider()

    # --------------------------------------------------------
    # Results table
    # --------------------------------------------------------

    results = pd.DataFrame({

        "Target": [
            "Tensile Strength",
            "Hardness",
            "Elongation"
        ],

        "R²": [
            metrics["Strength R2"],
            metrics["Hardness R2"],
            metrics["Elongation R2"]
        ],

        "MAE": [
            metrics["Strength MAE"],
            metrics["Hardness MAE"],
            metrics["Elongation MAE"]
        ]
    })

    st.dataframe(
        results,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    st.subheader("🌳 Feature Importance")

    importance = pd.DataFrame({

        "Feature": feature_columns,

        "Importance": strength_model.feature_importances_

    }).sort_values(
        "Importance",
        ascending=True
    )

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance for Tensile Strength"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        """
        Feature importance indicates how strongly each input
        contributes to the Random Forest prediction. It should
        be interpreted as a model-based measure, not as a direct
        causal metallurgical relationship.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Metallurgy AI Lab | Built with Python + Streamlit"
)