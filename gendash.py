import pandas as pd
import plotly.express as px
import plotly.io as pio

# ==========================================
# 1. Load the AI's Extracted Data
# ==========================================
try:
    df = pd.read_csv("sspe_cohort_analysis.csv")
    print(f"Loaded {len(df)} detected bursts for clinical visualization.")
except FileNotFoundError:
    print("Error: 'sspe_cohort_analysis.csv' not found. Run the extraction engine first!")
    exit()

# Clean up data types for visualization
df['Morphology_Cluster'] = "Cluster " + df['Morphology_Cluster'].astype(str)

# --- THE FIX: Handle NaNs from the Rolling Window Math ---
df['IBI_Periodicity_CoV'] = df.groupby('Patient_ID')['IBI_Periodicity_CoV'].bfill().fillna(1.0)
df_ibi = df[df['Rolling_Median_IBI_s'].notna() & (df['Rolling_Median_IBI_s'] > 0)]
# ---------------------------------------------------------

# ==========================================
# TIER 1: Clinical Progression (Smoothed IBI Decay)
# ==========================================
fig1 = px.scatter(
    df_ibi, 
    x="Timestamp_s", 
    y="Rolling_Median_IBI_s",  
    color="Patient_ID",
    size="IBI_Periodicity_CoV", 
    trendline="ols", 
    title="Tier 1: Disease Progression (Smoothed Biological Trajectory)",
    labels={
        "Timestamp_s": "Time in Recording (Seconds)", 
        "Rolling_Median_IBI_s": "Smoothed Baseline IBI (Seconds)"
    },
    hover_data=["Inter_Burst_Interval_s", "Bursts_Per_Minute", "AI_Confidence"] 
)
fig1.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))

# ==========================================
# TIER 2: Neuro-Physiological View (Frequency/Chaos)
# ==========================================
fig2 = px.scatter(
    df, 
    x="Spectral_Entropy", 
    y="Relative_Delta_Ratio", 
    color="Peak_to_Peak_uV",
    size="Peak_to_Peak_uV", 
    color_continuous_scale="Viridis",
    title="Tier 2: Spectral Signature (Frequency Chaos vs. Slow-Wave Power)",
    labels={"Spectral_Entropy": "Spectral Entropy (Chaos)", "Relative_Delta_Ratio": "Delta/Alpha Ratio (Slowing)"},
    hover_data=["Patient_ID", "Timestamp_s"]
)
fig2.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))

# ==========================================
# TIER 3: AI Discovery View (Latent Space)
# ==========================================
fig3 = px.scatter(
    df, 
    x="Latent_X", 
    y="Latent_Y", 
    color="Morphology_Cluster",
    symbol="Patient_ID", 
    title="Tier 3: Deep Learning Discovery (t-SNE Morphological Sub-Types)",
    labels={"Latent_X": "Latent Dimension 1", "Latent_Y": "Latent Dimension 2"},
    hover_data=["Timestamp_s", "Peak_to_Peak_uV", "AI_Confidence"]
)
fig3.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))

# ==========================================
# TIER 3b: Cluster Biology Breakdown (The "Why")
# ==========================================
print("Calculating morphological cluster statistics...")
cluster_analysis = df.groupby('Morphology_Cluster').agg({
    'Peak_to_Peak_uV': 'mean',          
    'Relative_Delta_Ratio': 'mean',     
    'Spectral_Entropy': 'mean',         
    'Patient_ID': 'nunique'             
}).round(2).reset_index()

# Rename columns for the clinical table
cluster_analysis.rename(columns={
    'Morphology_Cluster': 'Latent Cluster ID',
    'Peak_to_Peak_uV': 'Avg Voltage (µV)',
    'Relative_Delta_Ratio': 'Avg Delta Ratio',
    'Spectral_Entropy': 'Avg Entropy',
    'Patient_ID': 'Unique Patients in Cluster'
}, inplace=True)

# Generate raw HTML for the table
cluster_table_html = cluster_analysis.to_html(index=False, classes='clinical-table', border=0)

# ==========================================
# Export to Interactive HTML Dashboard
# ==========================================
print("Compiling interactive dashboard...")

html_content = f"""
<html>
<head>
    <title>Computational Neurology: SSPE Analysis</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; margin: 0; padding: 20px; }}
        h1 {{ text-align: center; color: #2c3e50; }}
        h3 {{ text-align: center; color: #7f8c8d; font-weight: normal; margin-bottom: 40px; }}
        .chart-container {{ background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 40px; padding: 20px; }}
        
        /* CSS for the new Clinical Table */
        .clinical-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; text-align: left; }}
        .clinical-table th {{ background-color: #2c3e50; color: white; padding: 12px 15px; }}
        .clinical-table td {{ padding: 12px 15px; border-bottom: 1px solid #dddddd; }}
        .clinical-table tbody tr:nth-of-type(even) {{ background-color: #f9f9f9; }}
        .clinical-table tbody tr:hover {{ background-color: #f1f1f1; }}
    </style>
</head>
<body>
    <h1>Automated SSPE Diagnostics & Sub-Typing Engine</h1>
    <h3>Cohort Analysis Dashboard</h3>
    
    <div class="chart-container">
        {fig1.to_html(full_html=False, include_plotlyjs='cdn')}
        <p style="text-align: center; color: #555;"><b>Interpretation:</b> A downward trendline indicates advancing disease pathology. <i>Note: IBI has been smoothed using a Rolling Median filter to remove phantom spikes caused by temporary biological refractory variations. Bubble size represents the Coefficient of Variation (rigidity).</i></p>
    </div>
    
    <div class="chart-container">
        {fig2.to_html(full_html=False, include_plotlyjs='cdn')}
        <p style="text-align: center; color: #555;"><b>Interpretation:</b> Identifies the severity of the background slowing. Large, bright dots indicate massive voltage discharges associated with high delta-wave ratios.</p>
    </div>
    
    <div class="chart-container">
        {fig3.to_html(full_html=False, include_plotlyjs='cdn')}
        <p style="text-align: center; color: #555;"><b>Interpretation:</b> The Transformer's mathematical grouping of the bursts. Distinct spatial clusters represent completely different physical shapes/dipoles of the Radermecker complexes.</p>
        
        <hr style="border: 0; border-top: 1px solid #eee; margin: 30px 0;">
        <h4 style="color: #2c3e50; margin-bottom: 10px;">Morphological Cluster Analytics</h4>
        <p style="color: #555; font-size: 14px; margin-top: 0;">The table below explains the physical biology driving the AI's spatial groupings above:</p>
        {cluster_table_html}
    </div>
</body>
</html>
"""

with open("clinical_dashboard.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Success! Open 'clinical_dashboard.html' in your web browser to view the presentation.")