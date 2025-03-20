import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 데이터 불러오기
df = pd.read_csv("data/Obesity Classification.csv")

# BMI 이상치 제거 (IQR 방법 적용)
Q1 = df['BMI'].quantile(0.25)
Q3 = df['BMI'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['BMI'] >= lower_bound) & (df['BMI'] <= upper_bound)]

# 레이블 인코딩
le_label = LabelEncoder()
df["Label"] = le_label.fit_transform(df["Label"])
le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])

# 특성과 타겟 변수 분리
X = df[['Age', 'Height', 'Weight', 'BMI', 'Gender']]
y = df['Label']

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=le_label.classes_)

# Streamlit UI 설정
st.set_page_config(page_title="Obesity Dashboard", layout="wide")
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "EDA", "Model Performance"])

# 홈 화면
def home():
    st.title("Obesity Classification Dashboard")
    st.markdown("""
    - **Age**: 나이
    - **Gender**: 성별 (Male, Female)
    - **Height**: 키 (Cm)
    - **Weight**: 몸무게 (Kg)
    - **BMI**: 체질량지수
    - **Label**: 비만 여부 (Normal Weight, Overweight, Obese, Underweight)
    """)

# EDA(데이터 시각화)
def eda():
    st.title("데이터 시각화")
    chart_tabs = st.tabs(["히스토그램", "박스플롯", "히트맵"])
    
    # 히스토그램 (연령, 키, 몸무게, BMI 분포)
    with chart_tabs[0]:
        st.subheader("연령, 키, 몸무게, BMI 분포")
        for col in ["Age", "Height", "Weight", "BMI"]:
            fig = px.histogram(df, x=col, nbins=20, title=f"{col} 분포",
                               labels={col: col}, marginal="box", color_discrete_sequence=["#636EFA"])
            st.plotly_chart(fig, use_container_width=True)
    
    # 박스플롯 (성별 및 비만 등급별 BMI)
    with chart_tabs[1]:
        st.subheader("BMI 박스플롯")
        fig = px.box(df, x="Gender", y="BMI", color="Label", 
                     title="성별 및 비만 등급별 BMI 분포",
                     labels={"Gender": "성별", "BMI": "BMI", "Label": "비만 등급"},
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    
    # 히트맵 (상관관계 분석)
    with chart_tabs[2]:
        st.subheader("변수 간 상관관계")
        corr_matrix = df.corr()
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="Blues",
                        title="변수 간 상관관계 히트맵")
        st.plotly_chart(fig, use_container_width=True)

# 모델 성능 평가
def model_performance():
    st.title("모델 성능 평가")
    st.write(f'### 모델 정확도: {accuracy:.2f}')
    st.text(classification_rep)
    
    # 특성 중요도 시각화
    feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
    fig = px.bar(feature_importance, x='Feature', y='Importance', title='특성 중요도',
                 labels={'Feature': '특성', 'Importance': '중요도'},
                 color='Importance', color_continuous_scale='Bluered_r')
    st.plotly_chart(fig, use_container_width=True)

# 메뉴 선택에 따른 화면 전환
if menu == "Home":
    home()
elif menu == "EDA":
    eda()
elif menu == "Model Performance":
    model_performance()
