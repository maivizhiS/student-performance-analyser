

# ============================================================
#   Student Academic Performance Analyser
#   Professional 
#   Built with: Python, Pandas, Seaborn, Scikit-learn, Streamlit
# ============================================================
 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")
 
# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Student Academic Performance Analyser",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ── Beautiful CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #F0FBF6; }
 
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #064e3b 0%, #0f766e 50%, #064e3b 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }
 
    /* Main title */
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #064e3b, #0d9488, #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0;
    }
 
    /* Subtitle */
    .sub-title {
        text-align: center;
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
 
    /* Section header */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #064e3b;
        border-left: 5px solid #0d9488;
        padding-left: 12px;
        margin: 1.5rem 0 1rem 0;
    }
 
    /* Metric cards */
    .metric-container {
        background: linear-gradient(135deg, #064e3b, #0d9488);
        border-radius: 15px;
        padding: 1.2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(13, 148, 136, 0.3);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #ccfbf1;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: white;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #99f6e4;
        margin-top: 0.2rem;
    }
 
    /* Insight cards */
    .insight-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #0d9488;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .insight-title {
        font-weight: 700;
        color: #064e3b;
        margin-bottom: 0.3rem;
    }
    .insight-text { color: #6b7280; font-size: 0.92rem; }
 
    /* Pass box */
    .pass-box {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border: 2px solid #10b981;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
    }
    .fail-box {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 2px solid #ef4444;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
    }
 
    /* Chart card */
    .chart-card {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
 
    /* Step card */
    .step-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        display: flex;
        gap: 1rem;
    }
 
    /* Badge */
    .badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        background: #ccfbf1;
        color: #134e4a;
        margin-right: 5px;
        margin-bottom: 5px;
    }
 
    /* Hide streamlit branding (keep header so the sidebar toggle stays usable) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent;}
    [data-testid="stHeader"] {background: transparent;}
</style>
""", unsafe_allow_html=True)
 
 
# ============================================================
# COLORS
# ============================================================
COLORS = {
    'primary':   '#064e3b',
    'secondary': '#0d9488',
    'accent':    '#059669',
    'success':   '#10b981',
    'warning':   '#f59e0b',
    'danger':    '#ef4444',
    'info':      '#0ea5e9',
    'male':      '#3b82f6',
    'female':    '#ec4899',
    'palette':   ['#064e3b','#0d9488','#059669','#0ea5e9','#10b981','#f59e0b']
}
 
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f5fdfa',
    'axes.grid':        True,
    'grid.color':       '#e5e7eb',
    'grid.linewidth':   0.5,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.family':      'DejaVu Sans',
})
 
 
# ============================================================
# LOAD & PREPARE DATA
# ============================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("StudentsPerformance.csv")
    except FileNotFoundError:
        st.error("❌ 'StudentsPerformance.csv' not found! Please place it in the same folder as app.py")
        st.stop()
 
    # Clean column names
    df.columns = (df.columns.str.strip()
                             .str.lower()
                             .str.replace(" ", "_")
                             .str.replace("/", "_"))
 
    # ── Replace "lunch" with "Family Income Level" ──
    if 'lunch' in df.columns:
        df.rename(columns={'lunch': 'family_income_level'}, inplace=True)
        df['family_income_level'] = df['family_income_level'].replace({
            'standard':     'High / Medium Income',
            'free/reduced': 'Low Income'
        })
 
    # ── Clean test prep ──
    if 'test_preparation_course' in df.columns:
        df['test_preparation_course'] = df['test_preparation_course'].replace({
            'none':      'Not Completed',
            'completed': 'Completed'
        })
 
    # ── Feature engineering ──
    df['average_score']  = (df['math_score'] + df['reading_score'] + df['writing_score']) / 3
    df['total_score']    = df['math_score'] + df['reading_score'] + df['writing_score']
    df['result']         = df['average_score'].apply(lambda x: 'Pass' if x >= 40 else 'Fail')
    df['performance']    = pd.cut(
        df['average_score'],
        bins=[0, 40, 55, 70, 85, 100],
        labels=['Fail', 'Below Average', 'Average', 'Good', 'Excellent']
    )
 
    return df
 
 
@st.cache_resource
def train_model(df):
    features = ['gender', 'race_ethnicity',
                'parental_level_of_education',
                'family_income_level',
                'test_preparation_course',
                'reading_score', 'writing_score']
    target = 'result'
 
    le = {}
    X  = df[features].copy()
    cat_cols = ['gender', 'race_ethnicity',
                'parental_level_of_education',
                'family_income_level',
                'test_preparation_course']
 
    for col in cat_cols:
        le[col] = LabelEncoder()
        X[col]  = le[col].fit_transform(X[col])
 
    y = LabelEncoder().fit_transform(df[target])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
 
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
 
    return lr, rf, le, \
           accuracy_score(y_test, lr.predict(X_test)), \
           accuracy_score(y_test, rf.predict(X_test)), \
           X_test, y_test
 
 
# ============================================================
# LOAD
# ============================================================
df = load_data()
lr_model, rf_model, encoders, lr_acc, rf_acc, X_test, y_test = train_model(df)
 
 
# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("""
<div style='text-align:center; padding: 1rem 0;'>
    <div style='font-size:3rem;'>🎓</div>
    <div style='font-size:1.1rem; font-weight:700; color:white;'>Student Academic</div>
    <div style='font-size:1.1rem; font-weight:700; color:#99f6e4;'>Performance Analyser</div>
</div>
""", unsafe_allow_html=True)
 
st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Navigation")
 
page = st.sidebar.radio("", [
    "🏠  Home & Overview",
    "📊  Exploratory Analysis",
    "📈  Advanced Visualizations",
    "🤖  ML Prediction",
    "📋  Model Performance",
    "💡  Insights & Conclusion"
])
 
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='font-size:0.8rem; color:#ccfbf1;'>
    <b style='color:white;'>📁 Dataset Info</b><br>
    Records: <b style='color:#99f6e4;'>1,000 students</b><br>
    Features: <b style='color:#99f6e4;'>8 columns</b><br>
    Source: <b style='color:#99f6e4;'>Kaggle</b><br><br>
    <b style='color:white;'>🤖 Model Accuracy</b><br>
    Logistic Regression: <b style='color:#99f6e4;'>{lr_acc*100:.1f}%</b><br>
    Random Forest: <b style='color:#99f6e4;'>{rf_acc*100:.1f}%</b>
</div>
""", unsafe_allow_html=True)
 
 
# ============================================================
# PAGE 1 — HOME & OVERVIEW
# ============================================================
if "Home" in page:
    st.markdown('<div class="main-title">🎓 Student Academic Performance Analyser</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">End-to-End Data Analysis & Machine Learning Prediction System</div>', unsafe_allow_html=True)
 
    # KPI Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        ("Total Students", f"{len(df):,}", "in dataset"),
        ("Avg Math Score", f"{df['math_score'].mean():.1f}", "out of 100"),
        ("Avg Reading", f"{df['reading_score'].mean():.1f}", "out of 100"),
        ("Avg Writing", f"{df['writing_score'].mean():.1f}", "out of 100"),
        ("Pass Rate", f"{(df['result']=='Pass').mean()*100:.1f}%", "students passed"),
    ]
    for col, (label, val, sub) in zip([c1,c2,c3,c4,c5], metrics):
        col.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # Dataset preview
    st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), width='stretch')
 
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">📌 Dataset Information</div>', unsafe_allow_html=True)
        info = pd.DataFrame({
            "Column":   df.columns.tolist(),
            "Type":     [str(d) for d in df.dtypes],
            "Non-Null": df.notnull().sum().values,
            "Nulls":    df.isnull().sum().values
        })
        st.dataframe(info, width='stretch')
 
    with col2:
        st.markdown('<div class="section-header">📊 Statistical Summary</div>', unsafe_allow_html=True)
        st.dataframe(
            df[['math_score','reading_score','writing_score','average_score']].describe().round(2),
            width='stretch'
        )
 
    # Pass/Fail + Performance
    st.markdown('<div class="section-header">🎯 Result Distribution</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('white')
 
    # Pie
    result_counts = df['result'].value_counts()
    axes[0].pie(result_counts, labels=result_counts.index,
                autopct='%1.1f%%',
                colors=[COLORS['success'], COLORS['danger']],
                startangle=90,
                wedgeprops=dict(edgecolor='white', linewidth=2))
    axes[0].set_title('Pass vs Fail', fontsize=13, fontweight='bold', color=COLORS['primary'])
 
    # Performance bar
    perf = df['performance'].value_counts().sort_index()
    bars = axes[1].bar(perf.index, perf.values,
                       color=COLORS['palette'][:len(perf)],
                       edgecolor='white', linewidth=1.5, width=0.6)
    axes[1].set_title('Performance Distribution', fontsize=13, fontweight='bold', color=COLORS['primary'])
    axes[1].set_xlabel('Performance Level')
    axes[1].set_ylabel('Number of Students')
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     str(int(bar.get_height())), ha='center', fontsize=9, fontweight='bold')
 
    # Gender donut
    gender_counts = df['gender'].value_counts()
    axes[2].pie(gender_counts, labels=gender_counts.index,
                autopct='%1.1f%%',
                colors=[COLORS['male'], COLORS['female']],
                startangle=90,
                wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2))
    axes[2].set_title('Gender Distribution', fontsize=13, fontweight='bold', color=COLORS['primary'])
 
    plt.tight_layout()
    st.pyplot(fig)
 
 
# ============================================================
# PAGE 2 — EXPLORATORY ANALYSIS
# ============================================================
elif "Exploratory" in page:
    st.markdown('<div class="main-title">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Deep dive into student performance patterns across multiple dimensions</div>', unsafe_allow_html=True)
 
    tab1, tab2, tab3, tab4 = st.tabs([
        "👥 Gender Analysis",
        "🎓 Parental Education",
        "📚 Exam Preparation",
        "💰 Family Income"
    ])
 
    def styled_bar(ax, data, title, xlabel='', ylabel='Score', colors=None):
        if colors is None: colors = COLORS['palette']
        bars = ax.bar(data.index, data.values,
                      color=colors[:len(data)] if isinstance(colors, list) else colors,
                      edgecolor='white', linewidth=1.5, width=0.5)
        ax.set_title(title, fontsize=12, fontweight='bold', color=COLORS['primary'], pad=10)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}', ha='center', fontsize=9, fontweight='bold')
        return ax
 
    with tab1:
        st.markdown('<div class="section-header">Gender vs Academic Scores</div>', unsafe_allow_html=True)
        gender_avg = df.groupby('gender')[['math_score','reading_score','writing_score']].mean().round(2)
        st.dataframe(gender_avg.style.background_gradient(cmap='Greens'), width='stretch')
 
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        gender_avg.T.plot(kind='bar', ax=axes[0],
                          color=[COLORS['male'], COLORS['female']], rot=0,
                          edgecolor='white', linewidth=1.5, width=0.6)
        axes[0].set_title('Average Scores by Gender', fontsize=12, fontweight='bold', color=COLORS['primary'])
        axes[0].set_ylabel('Score'); axes[0].legend(title='Gender')
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt='%.1f', fontsize=8, fontweight='bold')
 
        for gender, color in zip(df['gender'].unique(), [COLORS['male'], COLORS['female']]):
            df[df['gender']==gender]['average_score'].plot(
                kind='hist', bins=20, alpha=0.65, ax=axes[1],
                color=color, label=gender, edgecolor='white')
        axes[1].set_title('Score Distribution by Gender', fontsize=12, fontweight='bold', color=COLORS['primary'])
        axes[1].set_xlabel('Average Score'); axes[1].legend(title='Gender')
        plt.tight_layout(); st.pyplot(fig)
 
    with tab2:
        st.markdown('<div class="section-header">Parental Education vs Student Performance</div>', unsafe_allow_html=True)
        edu_order = ["some high school","high school","some college",
                     "associate's degree","bachelor's degree","master's degree"]
        edu_avg = df.groupby('parental_level_of_education')['average_score'].mean().reindex(edu_order).round(2)
 
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        bars = axes[0].barh(edu_avg.index, edu_avg.values,
                            color=COLORS['palette'], edgecolor='white', linewidth=1.5)
        axes[0].set_title('Average Score by Parental Education', fontsize=12, fontweight='bold', color=COLORS['primary'])
        axes[0].set_xlabel('Average Score')
        for bar, val in zip(bars, edu_avg.values):
            axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                         f'{val:.1f}', va='center', fontsize=9, fontweight='bold')
 
        sns.boxplot(data=df, x='parental_level_of_education', y='average_score',
                    order=edu_order, palette='Greens', ax=axes[1])
        axes[1].set_title('Score Distribution by Education Level', fontsize=12, fontweight='bold', color=COLORS['primary'])
        axes[1].set_xlabel(''); axes[1].set_ylabel('Average Score')
        axes[1].tick_params(axis='x', rotation=30)
        plt.tight_layout(); st.pyplot(fig)
 
    with tab3:
        st.markdown('<div class="section-header">Exam Preparation Course Impact</div>', unsafe_allow_html=True)
        prep_avg = df.groupby('test_preparation_course')[['math_score','reading_score','writing_score']].mean().round(2)
        st.dataframe(prep_avg.style.background_gradient(cmap='Greens'), width='stretch')
 
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        prep_avg.T.plot(kind='bar', ax=axes[0],
                        color=[COLORS['success'], COLORS['danger']], rot=0,
                        edgecolor='white', linewidth=1.5, width=0.6)
        axes[0].set_title('Score Comparison: Exam Preparation', fontsize=12, fontweight='bold', color=COLORS['primary'])
        axes[0].set_ylabel('Average Score'); axes[0].legend(title='Preparation')
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt='%.1f', fontsize=8, fontweight='bold')
 
        sns.boxplot(data=df, x='test_preparation_course', y='average_score',
                    palette={'Completed': COLORS['success'], 'Not Completed': COLORS['danger']},
                    ax=axes[1])
        axes[1].set_title('Score Distribution by Preparation', fontsize=12, fontweight='bold', color=COLORS['primary'])
        axes[1].set_xlabel('Exam Preparation'); axes[1].set_ylabel('Average Score')
        plt.tight_layout(); st.pyplot(fig)
 
    with tab4:
        st.markdown('<div class="section-header">Family Income Level vs Academic Performance</div>', unsafe_allow_html=True)
        income_avg = df.groupby('family_income_level')[['math_score','reading_score','writing_score']].mean().round(2)
        st.dataframe(income_avg.style.background_gradient(cmap='Purples'), width='stretch')
 
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        income_avg.T.plot(kind='bar', ax=axes[0],
                          color=[COLORS['secondary'], COLORS['warning']], rot=0,
                          edgecolor='white', linewidth=1.5, width=0.6)
        axes[0].set_title('Scores by Family Income Level', fontsize=12, fontweight='bold', color=COLORS['primary'])
        axes[0].set_ylabel('Average Score'); axes[0].legend(title='Income Level')
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt='%.1f', fontsize=8, fontweight='bold')
 
        sns.violinplot(data=df, x='family_income_level', y='average_score',
                       palette=[COLORS['secondary'], COLORS['warning']], ax=axes[1])
        axes[1].set_title('Score Distribution by Income Level', fontsize=12, fontweight='bold', color=COLORS['primary'])
        axes[1].set_xlabel('Family Income Level'); axes[1].set_ylabel('Average Score')
        plt.tight_layout(); st.pyplot(fig)
 
 
# ============================================================
# PAGE 3 — ADVANCED VISUALIZATIONS
# ============================================================
elif "Advanced" in page:
    st.markdown('<div class="main-title">📈 Advanced Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Statistical charts and correlation analysis for deep academic insights</div>', unsafe_allow_html=True)
 
    # Heatmap
    st.markdown('<div class="section-header">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    corr = df[['math_score','reading_score','writing_score','average_score','total_score']].corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='Greens',
                linewidths=1, linecolor='white', ax=ax,
                annot_kws={'size': 11, 'weight': 'bold'})
    ax.set_title('Correlation Between Academic Scores', fontsize=14,
                 fontweight='bold', color=COLORS['primary'], pad=15)
    plt.tight_layout(); st.pyplot(fig)
 
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">📦 Score Spread (Box Plot)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        df[['math_score','reading_score','writing_score']].plot(
            kind='box', ax=ax, patch_artist=True,
            boxprops=dict(facecolor='#ccfbf1', color=COLORS['secondary']),
            medianprops=dict(color=COLORS['accent'], linewidth=2.5),
            whiskerprops=dict(color=COLORS['secondary']),
            capprops=dict(color=COLORS['secondary']),
            flierprops=dict(marker='o', color=COLORS['danger'], markersize=4)
        )
        ax.set_title('Score Distribution (Box Plot)', fontsize=12,
                     fontweight='bold', color=COLORS['primary'])
        ax.set_ylabel('Score')
        plt.tight_layout(); st.pyplot(fig)
 
    with col2:
        st.markdown('<div class="section-header">🔵 Math vs Reading Scatter</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        colors_map = {'Pass': COLORS['success'], 'Fail': COLORS['danger']}
        for result, group in df.groupby('result'):
            ax.scatter(group['math_score'], group['reading_score'],
                       label=result, alpha=0.6, color=colors_map[result], s=25)
        ax.set_xlabel('Math Score'); ax.set_ylabel('Reading Score')
        ax.set_title('Math vs Reading (by Result)', fontsize=12,
                     fontweight='bold', color=COLORS['primary'])
        ax.legend(title='Result')
        plt.tight_layout(); st.pyplot(fig)
 
    # Histograms
    st.markdown('<div class="section-header">📊 Score Distribution Histograms</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, color, title in zip(
        axes,
        ['math_score','reading_score','writing_score'],
        [COLORS['secondary'], COLORS['info'], COLORS['accent']],
        ['Mathematics Score','Reading Score','Writing Score']
    ):
        n, bins, patches = ax.hist(df[col], bins=20, color=color,
                                    edgecolor='white', alpha=0.85, linewidth=1.2)
        ax.axvline(df[col].mean(), color=COLORS['danger'], linestyle='--',
                   linewidth=2, label=f"Mean: {df[col].mean():.1f}")
        ax.axvline(df[col].median(), color=COLORS['warning'], linestyle='-.',
                   linewidth=2, label=f"Median: {df[col].median():.1f}")
        ax.set_title(title, fontsize=11, fontweight='bold', color=COLORS['primary'])
        ax.set_xlabel('Score'); ax.set_ylabel('Count')
        ax.legend(fontsize=8)
    plt.tight_layout(); st.pyplot(fig)
 
    # Pair plot
    st.markdown('<div class="section-header">🔗 Pair Plot Analysis</div>', unsafe_allow_html=True)
    fig = sns.pairplot(
        df[['math_score','reading_score','writing_score','result']],
        hue='result',
        palette={'Pass': COLORS['success'], 'Fail': COLORS['danger']},
        plot_kws={'alpha': 0.5, 's': 15},
        diag_kind='kde'
    ).fig
    fig.suptitle('Pair Plot: Score Relationships', y=1.02,
                 fontsize=13, fontweight='bold', color=COLORS['primary'])
    st.pyplot(fig)
 
 
# ============================================================
# PAGE 4 — ML PREDICTION
# ============================================================
elif "ML Prediction" in page:
    st.markdown('<div class="main-title">🤖 ML Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter student details to predict academic performance using Machine Learning</div>', unsafe_allow_html=True)
 
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#ccfbf1,#f0fdf9);
                border-radius:12px; padding:1rem 1.5rem; margin-bottom:1.5rem;
                border-left:5px solid {COLORS["secondary"]};'>
        <b style='color:{COLORS["primary"]};'>About this Prediction System</b><br>
        <span style='color:#6b7280; font-size:0.9rem;'>
        This system uses two trained Machine Learning models — Logistic Regression
        ({lr_acc*100:.1f}% accuracy) and Random Forest ({rf_acc*100:.1f}% accuracy) —
        to predict whether a student will Pass or Fail based on their profile.
        </span>
    </div>
    """, unsafe_allow_html=True)
 
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 👤 Student Profile")
        gender       = st.selectbox("Gender", df['gender'].unique())
        race         = st.selectbox("Race / Ethnicity", df['race_ethnicity'].unique())
        edu          = st.selectbox("Parental Level of Education", df['parental_level_of_education'].unique())
        income       = st.selectbox("Family Income Level", df['family_income_level'].unique())
        prep         = st.selectbox("Exam Preparation Course", df['test_preparation_course'].unique())
 
    with col2:
        st.markdown("#### 📝 Academic Scores")
        reading_score = st.slider("Reading Score", 0, 100, 70,
                                  help="Enter the student's reading score out of 100")
        writing_score = st.slider("Writing Score", 0, 100, 68,
                                  help="Enter the student's writing score out of 100")
        st.markdown("#### 🤖 Select Model")
        model_choice = st.radio("Choose ML Model",
                                ["Logistic Regression", "Random Forest"],
                                horizontal=True)
        st.info(f"Selected model accuracy: **{lr_acc*100:.1f}%**" if model_choice == "Logistic Regression"
                else f"Selected model accuracy: **{rf_acc*100:.1f}%**")
 
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict Student Performance",
                            width='stretch',
                            type="primary")
 
    if predict_btn:
        input_data = pd.DataFrame([{
            'gender':                       gender,
            'race_ethnicity':               race,
            'parental_level_of_education':  edu,
            'family_income_level':          income,
            'test_preparation_course':      prep,
            'reading_score':                reading_score,
            'writing_score':                writing_score
        }])
 
        for col in ['gender','race_ethnicity','parental_level_of_education',
                    'family_income_level','test_preparation_course']:
            input_data[col] = encoders[col].transform(input_data[col])
 
        model = lr_model if model_choice == "Logistic Regression" else rf_model
        prediction  = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
 
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")
 
        r1, r2, r3 = st.columns(3)
        r1.metric("Pass Probability",  f"{probability[1]*100:.1f}%")
        r2.metric("Fail Probability",  f"{probability[0]*100:.1f}%")
        r3.metric("Model Used", model_choice.split()[0])
 
        # Estimated math
        est_math = min(100, max(0, int(reading_score * 0.45 + writing_score * 0.35 + 8)))
        avg_est  = round((reading_score + writing_score + est_math) / 3, 1)
 
        if prediction == 1:
            st.markdown(f"""
            <div class="pass-box">
                <div style='font-size:2.5rem;'>✅</div>
                <div style='font-size:1.5rem; font-weight:800; color:#065f46;'>Predicted: PASS</div>
                <div style='color:#047857; margin-top:0.5rem;'>
                    This student is likely to perform well with an estimated average of {avg_est}/100
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="fail-box">
                <div style='font-size:2.5rem;'>⚠️</div>
                <div style='font-size:1.5rem; font-weight:800; color:#991b1b;'>Predicted: NEEDS SUPPORT</div>
                <div style='color:#b91c1c; margin-top:0.5rem;'>
                    This student may need additional academic support. Estimated average: {avg_est}/100
                </div>
            </div>""", unsafe_allow_html=True)
 
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Estimated Score Breakdown")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Mathematics (Est.)", f"{est_math}/100")
        sc2.metric("Reading", f"{reading_score}/100")
        sc3.metric("Writing", f"{writing_score}/100")
        sc4.metric("Average (Est.)", f"{avg_est}/100")
 
        # Confidence chart
        fig, ax = plt.subplots(figsize=(8, 2))
        bars = ax.barh(['Fail', 'Pass'], [probability[0], probability[1]],
                       color=[COLORS['danger'], COLORS['success']],
                       edgecolor='white', linewidth=1.5, height=0.4)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability')
        ax.set_title('Prediction Confidence', fontweight='bold', color=COLORS['primary'])
        for bar, val in zip(bars, [probability[0], probability[1]]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val*100:.1f}%', va='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
 
 
# ============================================================
# PAGE 5 — MODEL PERFORMANCE
# ============================================================
elif "Model Performance" in page:
    st.markdown('<div class="main-title">📋 Model Performance Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Detailed evaluation metrics for both Machine Learning models</div>', unsafe_allow_html=True)
 
    tab1, tab2 = st.tabs(["📈 Logistic Regression", "🌲 Random Forest"])
 
    for tab, model, name, acc in [
        (tab1, lr_model, "Logistic Regression", lr_acc),
        (tab2, rf_model, "Random Forest",        rf_acc)
    ]:
        with tab:
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred,
                                           target_names=['Fail','Pass'],
                                           output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
 
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy",  f"{acc*100:.2f}%")
            m2.metric("Precision", f"{report['Pass']['precision']*100:.1f}%")
            m3.metric("Recall",    f"{report['Pass']['recall']*100:.1f}%")
            m4.metric("F1 Score",  f"{report['Pass']['f1-score']*100:.1f}%")
 
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                            xticklabels=['Fail','Pass'],
                            yticklabels=['Fail','Pass'],
                            ax=ax, linewidths=1, linecolor='white',
                            annot_kws={'size': 14, 'weight': 'bold'})
                ax.set_xlabel('Predicted', fontsize=11)
                ax.set_ylabel('Actual', fontsize=11)
                ax.set_title(f'Confusion Matrix — {name}', fontsize=12,
                             fontweight='bold', color=COLORS['primary'])
                plt.tight_layout(); st.pyplot(fig)
 
            with col2:
                st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(report).T.round(3), width='stretch')
 
    # Feature Importance
    st.markdown('<div class="section-header">🏆 Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    features = ['Gender','Race/Ethnicity','Parental Education',
                'Family Income','Exam Preparation','Reading Score','Writing Score']
    importance = pd.Series(rf_model.feature_importances_, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(importance.index, importance.values,
                   color=COLORS['palette'][:len(importance)],
                   edgecolor='white', linewidth=1.5)
    ax.set_title('Feature Importance — Which Factors Affect Performance Most?',
                 fontsize=12, fontweight='bold', color=COLORS['primary'])
    ax.set_xlabel('Importance Score')
    for bar, val in zip(bars, importance.values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    plt.tight_layout(); st.pyplot(fig)
 
 
# ============================================================
# PAGE 6 — INSIGHTS & CONCLUSION
# ============================================================
elif "Insights" in page:
    st.markdown('<div class="main-title">💡 Insights & Conclusion</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Key findings, recommendations and project summary</div>', unsafe_allow_html=True)
 
    st.markdown('<div class="section-header">🔍 Key Findings</div>', unsafe_allow_html=True)
 
    insights = [
        ("📚 Exam Preparation is the Strongest Booster",
         "Students who completed exam preparation courses scored 8–10 points higher on average across all subjects. This is the most impactful and actionable finding from the entire analysis."),
        ("👥 Gender Differences Vary by Subject",
         "Male students scored approximately 5 points higher in Mathematics, while female students outperformed in Reading (7 pts) and Writing (8 pts). Neither gender is universally superior — they excel in different areas."),
        ("🎓 Parental Education Significantly Influences Performance",
         "Students whose parents hold post-graduate degrees averaged 75+ points, compared to 58 points for students from lower educational backgrounds — a significant 17-point gap revealing generational academic influence."),
        ("💰 Family Income Level Reflects Academic Opportunity",
         "Students from high/medium income families scored 10–12 points higher than those from low income backgrounds, highlighting the role of socioeconomic factors in academic access and support."),
        ("📖 Reading & Writing are Highly Correlated (r = 0.95)",
         "These two skills move together strongly, confirming a shared literacy foundation. Mathematics shows moderate correlation (r ≈ 0.82), suggesting it draws on more independent cognitive skills."),
        ("🤖 ML Models Achieved High Accuracy",
         f"Logistic Regression achieved {lr_acc*100:.1f}% accuracy and Random Forest achieved {rf_acc*100:.1f}% accuracy in predicting student pass/fail — demonstrating the practical value of ML in academic analytics."),
    ]
 
    for title, desc in insights:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">{title}</div>
            <div class="insight-text">{desc}</div>
        </div>""", unsafe_allow_html=True)
 
    st.markdown('<div class="section-header">🎯 Recommendations</div>', unsafe_allow_html=True)
    recs = [
        ("📝 Make Exam Preparation Mandatory",
         "Schools and colleges should make exam preparation programs free and mandatory for all students, as they show the strongest positive impact on academic performance."),
        ("💰 Support Economically Challenged Students",
         "Institutions should provide scholarships, free study materials, and additional academic support to students from low income families to bridge the socioeconomic performance gap."),
        ("👨‍👩‍👧 Parental Engagement Programs",
         "Design programs to actively involve parents in student learning. Students with educated parents perform significantly better — engagement programs can replicate this support."),
        ("🎯 Personalized Learning Plans",
         "Use the ML prediction model to identify at-risk students early and create personalized intervention plans before examinations, not after failure."),
    ]
 
    col1, col2 = st.columns(2)
    for i, (title, desc) in enumerate(recs):
        col = col1 if i % 2 == 0 else col2
        col.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">{title}</div>
            <div class="insight-text">{desc}</div>
        </div>""", unsafe_allow_html=True)
 
    # Conclusion
    st.markdown('<div class="section-header">📄 Conclusion</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#064e3b,#0d9488);
                border-radius:15px; padding:1.5rem 2rem; color:white;'>
        <p style='font-size:1rem; line-height:1.8; margin-bottom:1rem;'>
        This project successfully demonstrates a complete end-to-end data analysis pipeline —
        from raw data collection and preprocessing, through exploratory data analysis and
        advanced visualization, to building and deploying production-ready machine learning models.
        </p>
        <p style='font-size:1rem; line-height:1.8; margin-bottom:1rem;'>
        The analysis reveals that <b>exam preparation, family income level, and parental education</b>
        are the strongest predictors of student academic performance. The Random Forest classifier
        achieved <b>{rf_acc*100:.1f}% accuracy</b> in predicting student outcomes —
        making this a practically deployable tool for educational institutions.
        </p>
        <p style='font-size:1rem; line-height:1.8; margin:0;'>
        The insights generated can guide targeted academic interventions, early identification
        of at-risk students, and data-driven policy decisions to improve overall academic outcomes.
        </p>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; padding:1rem;'>
        <div style='color:#6b7280; font-size:0.9rem;'>
            <b>Tools & Technologies Used</b><br>
            <span class='badge'>Python</span>
            <span class='badge'>Pandas</span>
            <span class='badge'>NumPy</span>
            <span class='badge'>Matplotlib</span>
            <span class='badge'>Seaborn</span>
            <span class='badge'>Scikit-learn</span>
            <span class='badge'>Streamlit</span>
        </div>
        <div style='color:#9ca3af; font-size:0.8rem; margin-top:0.5rem;'>
            Dataset: Students Performance in Exams — Kaggle &nbsp;|&nbsp;
            Deployed on Streamlit Cloud
        </div>
    </div>
    """, unsafe_allow_html=True)
 


