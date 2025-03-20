import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

## 데이터 확인 및 전처리
# 데이터 불러오기 (pandas 활용)
df = pd.read_csv("data/Obesity Classification.csv")
# st.dataframe(df)

# Step1. BMI의 이상치를 IQR(사분위수 범위) 방법을 사용하여 제거
# 결측치, 이상치 확인 및 처리
Q1 = df['BMI'].quantile(0.25) # 1사분위수 (25% 지점)
Q3 = df['BMI'].quantile(0.75) # 3사분위수 (75% 지점)
IQR = Q3 - Q1 #사분위범위

# 이상치 경계 계산
lower_bound = Q1 - 1.5 * IQR # 이상치 하한값
upper_bound = Q3 + 1.5 * IQR # 이상치 상한값

# BMI 이상치 제거
df = df[(df['BMI'] >= lower_bound) & (df['BMI'] <= upper_bound)]
# st.dataframe(df)
# Step2. 데이터 타입 변환 및 정리
# Label Encoding(레이블 인코딩)은 문자열(범주형 데이터)을 숫자로 변환하는 방법입니다.
le_label = LabelEncoder()
df["Label"] = le_label.fit_transform(df["Label"])
le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])
# print(dict(zip(le_label.classes_, le_label.transform(le_label.classes_))))
# print(dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_))))

# Step3: 특성과 타겟변수 분리
# 특성과 타겟 분리
# X (입력 데이터, Feature Set):
# Age (나이)
# Height (키)
# Weight (몸무게)
# BMI (체질량지수)
# Gender (성별)

# y (타겟 데이터, Label):
# 비만 등급 (Underweight, Normal Weight, Overweight, Obese)
# 이제 X와 y를 나눴으므로, 모델이 학습할 데이터를 준비한 상태입니다.
X=df[['Age','Height','Weight','BMI','Gender']]
y=df['Label']

# Step4: 데이터 정규화(Scaling, 표준화)
# 머신러닝 모델은 특성(Feature) 값들의 크기가 다를 경우, 특정 값이 과도한 영향을 주는 문제(스케일링 문제)가 발생할 수 있음.
# 키(Height, cm)는 150~200 범위, 몸무게(Weight, kg)는 40~100 범위지만, BMI는 15~40 범위이므로 직접 비교가 어려움.
# StandardScaler()를 사용하여 모든 특성을 같은 범위(평균=0, 표준편차=1)로 변환하면 학습이 더 안정적임.
scaler = StandardScaler()
X_scaled =scaler.fit_transform(X)

# step5: 데이터셋 분할(train-test Split)
# 머신러닝 모델이 학습(Training) 과 평가(Testing) 를 위해 데이터를 분리해야 함.
# train_test_split() 함수를 사용하여 80%는 학습 데이터, 20%는 테스트 데이터로 분리합니다.
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

# step6: 머신러닝 모델 학습
# 우리가 사용하는 모델은 랜덤 포레스트(Random Forest) 분류 모델입니다.
# 랜덤 포레스트는 결정 트리(Decision Tree)를 여러 개 조합하여 예측 정확도를 높이는 모델입니다
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# if BMI >=25
#    if 나이 >=40
#       if 몸무게 >=90
#          비만
#       else 과체중
#    else 과체중      
# else 정상체중

# 특성 중요도 출력
# 결과에 영향을 가장 많이 미치는 컬럼을 찾는 방법
# 모델이 예측하는 데 중요한 변수를 찾는 것이 핵심!
# 1) Feature Importance (특성 중요도) 확인
# 랜덤 포레스트 같은 트리 기반 모델을 활용하면, 어떤 변수가 모델 예측에 중요한 역할을 하는지 평가 가능.
# feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
# feature_importances = feature_importances.sort_values(by="Importance", ascending=False)
# print(feature_importances)

#Step7: 예측 및 성능평가(Model Prediction & Evaluation)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names = le_label.classes_)
# st.write(f'### Model Accuracy: {accuracy:.2f}')
# st.text(classification_rep)
#precision(정밀도): 모델이 '비만'이라고 예측한 것 중 실제로 맞는 비율 => '과체중'을 예측했을때 실제로 과체중인 비율
#Recall(재현율): 실제'비만'중에서 모델이 맞춘 비율 => '비만'인 사람을 놓치지 않고 얼마나 예측했는가?
#F1-score : Precision과 Recall의 조화평균 => 모델의 전체적인 성능을 나타냄
#Support 각 클래스(Label)에 속하는 샘플의 개수 => 데이터셋 내에 존재하는 해당 클래스갯수

# Streamlit UI디자인
st.set_page_config(page_title="Obesity Dashbord", layout="wide")
# 사이드바 메뉴
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to",["Home","EDA","Model Performance"])

# 홈화면
def home():
    st.title("Obesity Classification Dashboard")
    st.markdown("""
    - **Age**: 나이
    - **Gender**: 성별(Male, Female)
    - **Height**: 키 (Cm)
    - **Weight**: 몸무게 (Kg)
    - **BMI**:  체질량지수
    - **Label**: 비만 여부 (Normal Weight, Overweight, Obese, Underweight)
    """)

# EDA(데이터시각화화면)
def eda():
    st.title("데이터 시각화")
    chart_tabs = st.tabs(["histgram","boxplot","hitmap"])
    
    # 히스토그램(Age, Height, Weight, BMI 분포)
    with chart_tabs[0]:
        st.subheader("연령, 키, 몸무게, BMI 분포")
        fig, axes = plt.subplots(2,2, figsize=(12,8))
        columns = ["Age","Height","Weight","BMI"]
        for i, col in enumerate(columns):
            ax = axes[i//2, i%2]
            sns.histplot(df[col],bins=20,kde=True, ax=ax)
            ax.set_title(col)
        st.pyplot(fig)

    # 박스플롯 (성별 및 비만 등급별 BMI)
    with chart_tabs[1]:
        st.subheader("BMI")
        fig, ax = plt.subplots(figsize=(8,5))
        sns.boxplot(data=df, x="Gender",y="BMI", hue="Label", palette="Set2",ax=ax)
        ax.set_title("BMI-BOXPLOT")
        st.pyplot(fig)




# 메뉴 선택에 따른 화면 전환
if menu == "Home":
    home()
elif menu == "EDA":
    eda()
elif menu == "Model Performance":
    model_performance()

