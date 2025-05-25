import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF
from PIL import Image
import io

st.set_page_config(page_title="MediRank: A Diagnosis Support Tool Using Euclidean Distance-Based Similarity Analysis")
st.title("🧠 MediRank: Diagnosis Support Tool Using Euclidean Distance-Based Similarity Analysis")

with st.form("diagnosis_form"):
    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Patient Name", "John Doe")
        fever = st.number_input("Fever (C)", 35.0, 42.0, 38.5)
        platelet = st.number_input("Platelet Count", 10, 500, 55)
        wbc = st.number_input("WBC Count", 0.5, 30.0, 3.0)
    with col2:
        doctor_name = st.text_input("Doctor Name", "Dr. Zahari")
        bleeding = st.selectbox("Bleeding", ["No", "Yes"])
        fatigue = st.selectbox("Fatigue", ["None", "Mild", "Moderate", "Severe"])
        pain = st.selectbox("Pain", ["None", "Mild", "Moderate", "Severe"])
        nausea = st.selectbox("Nausea", ["None", "Slight", "Frequent"])
    submitted = st.form_submit_button("🩺 Diagnose")

if submitted:
    fuzzy_map = {"None": 0.0, "Mild": 0.3, "Moderate": 0.6, "Severe": 0.9, "Slight": 0.6, "Frequent": 0.9, "Yes": 1, "No": 0}
    patient_vector = [fever, platelet, wbc, fuzzy_map[bleeding], fuzzy_map[fatigue], fuzzy_map[pain], fuzzy_map[nausea]]

    diseases = {
        "Dengue DHF":    [38.5,  45,  4.0, 1.0, 0.6, 0.9, 0.9],
        "Sepsis":        [39.0, 180, 25.0, 0.0, 0.3, 0.6, 0.0],
        "Meningitis":    [39.2, 150, 12.0, 0.0, 0.6, 0.9, 0.6],
        "Leukemia":      [38.0,  90,  1.2, 1.0, 0.6, 0.9, 0.3],
        "Typhoid Fever": [38.8, 110,  5.0, 0.0, 0.6, 0.3, 0.9],
        "Malaria":       [39.5,  70,  6.5, 1.0, 0.9, 0.9, 0.9],
        "COVID-19":      [39.0, 140,  7.0, 0.0, 0.9, 0.6, 0.3],
        "Hepatitis A":   [38.0, 160,  5.5, 0.0, 0.6, 0.3, 0.9],
        "Appendicitis":  [38.2, 250, 10.0, 0.0, 0.3, 0.9, 0.6]
    }

    def euclidean_distance(v1, v2):
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

    distances = {k: euclidean_distance(patient_vector, v) for k, v in diseases.items()}
    max_dist = max(distances.values())
    similarities = {k: (1 - d / max_dist) * 100 for k, d in distances.items()}

    result_df = pd.DataFrame([{"Disease": k, "Distance": round(distances[k], 2), "Similarity (%)": round(similarities[k], 1)} for k in distances])
    result_df.sort_values(by="Similarity (%)", ascending=False, inplace=True)

    top_disease = result_df.iloc[0]["Disease"]

    explanations = {
        "Dengue DHF": ["Fever and platelet count match dengue profile closely.", "Bleeding and high pain levels are consistent with dengue hemorrhagic fever symptoms.", "Nausea and fatigue further support dengue likelihood."],
        "Sepsis": ["Elevated WBC and fever align with systemic infection.", "Fatigue and moderate pain also support sepsis diagnosis."],
        "Meningitis": ["High fever and pain suggest possible CNS involvement.", "Moderate nausea supports meningitis pattern."],
        "Leukemia": ["Low WBC and platelets indicate hematologic abnormality.", "Persistent fatigue and moderate pain are common."],
        "Typhoid Fever": ["Fever with GI-related symptoms like nausea aligns with typhoid.", "Moderate fatigue and pain reinforce the match."],
        "Malaria": ["High fever, low platelets, and fatigue are classic malaria signs.", "Pain and nausea strongly support this diagnosis."],
        "COVID-19": ["Fever and respiratory fatigue common in severe COVID.", "Mild nausea and moderate pain also seen in cases."],
        "Hepatitis A": ["Mild fever and fatigue typical in liver inflammation.", "GI symptoms like nausea are strong indicators."],
        "Appendicitis": ["High WBC and localized pain consistent with appendicitis.", "Fever and slight nausea support early-stage detection."]
    }

    st.subheader("📋 Patient Input Summary")
    input_labels = ["Fever", "Platelet", "WBC", "Bleeding", "Fatigue", "Pain", "Nausea"]
    for label, val in zip(input_labels, patient_vector):
        st.write(f"**{label}:** {val}")

    st.subheader("📊 Diagnosis Ranking")
    st.dataframe(result_df)
    st.success(f"✅ Most likely diagnosis: **{top_disease}**")

    st.markdown("### 🔍 Why this diagnosis?")
    for point in explanations.get(top_disease, ["No explanation available."]):
        st.markdown(f"- {point}")