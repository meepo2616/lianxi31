import streamlit as st
import joblib
import pandas as pd

# 加载模型和选中的特征
try:
    model = joblib.load('XGB.pkl')
    selected_features = joblib.load('selected_features.pkl')  # 加载前4个特征名称（列表类型）
except Exception as e:
    st.error(f"加载失败: {str(e)}")
    st.stop()

# 完整的特征范围定义
full_feature_ranges = {
    "Vw": {"min": 0, "max": 1000000, "default": 10400},
    "Bave": {"min": 0, "max": 100, "default": 50},
    "hd": {"min": 0, "max": 100, "default": 50},
    "hb": {"min": 0, "max": 100, "default": 50},
    "S": {"min": 0, "max": 1000000, "default": 10400},
    "hw": {"min": 0, "max": 100, "default": 50},
}

# Streamlit 界面
st.title("Qp Prediction with XGBoost (Optimized Features)")

# 动态生成输入项（仅显示前4个重要特征）
st.header("Input Feature Values")
feature_values = {}  # 使用字典存储特征值
for feature in selected_features:  # 只遍历选中的特征
    props = full_feature_ranges[feature]
    value = st.number_input(
        f"{feature} ({props['min']} - {props['max']})",
        min_value=props["min"],
        max_value=props["max"],
        value=props["default"],
    )
    feature_values[feature] = value  # 正确使用字典赋值

# 预测逻辑
if st.button("Predict Qp"):
    try:
        # 确保顺序与模型训练时的特征顺序一致
        input_values = [feature_values[feat] for feat in selected_features]
        input_data = pd.DataFrame([input_values], columns=selected_features)
        prediction = model.predict(input_data)[0]
        st.success(f"**Predicted Qp:** {prediction:.2f}")

    except Exception as e:
        st.error(f"预测失败: {str(e)}")
