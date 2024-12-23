import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, accuracy_score
import joblib

st.title("AI-ML: Regression or Classification Models App")
st.write("Upload a dataset and choose a target column")

uploaded_file = st.file_uploader("Upload dataset", type = ["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)

    st.write("### Preview of Dataset")
    st.dataframe(df.head(5))

    st.write("### Correlation Heatmap")
    numeric_df = df.select_dtypes(include = [np.number])
    if numeric_df.shape[1] > 1:
        fig, ax = plt.subplots(figsize = (8, 6))
        sns.heatmap(numeric_df.corr(), annot = True, cmap = "coolwarm", fmt = ".2f", ax = ax)
        st.pyplot(fig)
    else:
        st.write("Not enough numeric columns for a correlation heatmap.")

    target_column = st.selectbox("Select Target Column (Prediction Target)", df.columns)

    if target_column:
        if df[target_column].nunique() <= 10:
            problem_type = "Classification"
        else:
            problem_type = "Regression"

        st.write(f"Problem Type: **{problem_type}**")

        
        st.write("### Select Features for Model Training")
        default_features = [col for col in df.columns if col != target_column]
        selected_features = st.multiselect(
            "Select features (remove unnecessary ones like IDs):",
            default_features,
            default_features,
        )

        if not selected_features:
            st.error("Please select at least one feature.")
        else:
            features = selected_features
            numeric_features = df[features].select_dtypes(include = [np.number])
            categorical_features = df[features].select_dtypes(include = ['object'])

            scaler = StandardScaler()
            encoder = OneHotEncoder(drop = 'first', sparse_output = False)

            #Handling missing values
            X = df[features]
            y = df[target_column]

            if X.isnull().any().any():
                numeric_cols = X.select_dtypes(include = [np.number]).columns
                categorical_cols = X.select_dtypes(exclude = [np.number]).columns

                if not numeric_cols.empty:
                    numeric_imputer = SimpleImputer(strategy = "mean")
                    X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])

                if not categorical_cols.empty:
                    categorical_imputer = SimpleImputer(strategy = "most_frequent")
                    X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])

            if not numeric_features.empty:
                scaled_numeric = scaler.fit_transform(X.select_dtypes(include = [np.number]))
                X_numeric = pd.DataFrame(scaled_numeric, columns = numeric_features.columns)
            else:
                X_numeric = pd.DataFrame()

            if not categorical_features.empty:
                encoded_categorical = encoder.fit_transform(X.select_dtypes(exclude = [np.number]))
                encoded_columns = encoder.get_feature_names_out(categorical_features.columns)
                X_categorical = pd.DataFrame(encoded_categorical, columns = encoded_columns, index = df.index)
            else:
                X_categorical = pd.DataFrame()

            X = pd.concat([X_numeric, X_categorical], axis = 1)

            feature_names = X.columns.tolist()
            joblib.dump(feature_names, "feature_names.pkl")

            if problem_type == "Classification" and y.dtype == 'object':
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)

            if X.empty:
                st.error("No valid features found for training. Please upload a suitable dataset.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

                if problem_type == "Regression":
                    st.write("## Training Regression Models")
                    lin_reg = LinearRegression()
                    lin_reg.fit(X_train, y_train)
                    y_pred_lr = lin_reg.predict(X_test)
                    r2_lr = r2_score(y_test, y_pred_lr)

                    rf_reg = RandomForestRegressor()
                    rf_reg.fit(X_train, y_train)
                    y_pred_rf = rf_reg.predict(X_test)
                    r2_rf = r2_score(y_test, y_pred_rf)

                    st.write("### Model Evaluation Metrics")
                    st.write(f"**Linear Regression**: R² = {r2_lr:.2f}")
                    st.write(f"**Random Forest Regressor**: R² = {r2_rf:.2f}")

                    joblib.dump(lin_reg, "linear_regression_model.pkl")
                    joblib.dump(rf_reg, "random_forest_regressor.pkl")

                elif problem_type == "Classification":
                    st.write("## Training Classification Models")
                    log_reg = LogisticRegression(max_iter = 1000)
                    log_reg.fit(X_train, y_train)
                    acc_lr = accuracy_score(y_test, log_reg.predict(X_test))

                    rf_clf = RandomForestClassifier()
                    rf_clf.fit(X_train, y_train)
                    acc_rf = accuracy_score(y_test, rf_clf.predict(X_test))

                    st.write("### Model Evaluation Metrics")
                    st.write(f"**Logistic Regression**: Accuracy = {acc_lr:.2f}")
                    st.write(f"**Random Forest Classifier**: Accuracy = {acc_rf:.2f}")

                    joblib.dump(log_reg, "logistic_regression_model.pkl")
                    joblib.dump(rf_clf, "random_forest_classifier.pkl")

                st.write("## Make Predictions on New Data")
                user_input = {}
                for feature in selected_features:
                    if np.issubdtype(df[feature].dtype, np.number):
                        if np.issubdtype(df[feature].dtype, np.integer):
                            user_input[feature] = st.number_input(
                                f"Enter value for {feature}",
                                value=int(df[feature].mean()),
                                step=1,
                            )
                        elif np.issubdtype(df[feature].dtype, np.floating):
                            user_input[feature] = st.number_input(
                                f"Enter value for {feature}",
                                value = float(df[feature].mean()),
                                format = "%.2f",
                            )
                    else:
                        user_input[feature] = st.selectbox(
                            f"Select value for {feature}", 
                            options = df[feature].dropna().unique(),
                        )

                input_df = pd.DataFrame([user_input])

                numeric_input = input_df.select_dtypes(include = [np.number])
                categorical_input = input_df.select_dtypes(exclude = [np.number])

                if not numeric_input.empty:
                    numeric_input_scaled = scaler.transform(numeric_input)
                else:
                    numeric_input_scaled = np.array([])

                if not categorical_input.empty:
                    categorical_input_encoded = encoder.transform(categorical_input)
                else:
                    categorical_input_encoded = np.array([])

                input_df_scaled = (
                    np.hstack([numeric_input_scaled, categorical_input_encoded])
                    if numeric_input_scaled.size > 0 and categorical_input_encoded.size > 0
                    else numeric_input_scaled
                    if numeric_input_scaled.size > 0
                    else categorical_input_encoded
                )

                input_df_scaled = pd.DataFrame(input_df_scaled, columns = feature_names, index = input_df.index)

                for col in feature_names:
                    if col not in input_df_scaled.columns:
                        input_df_scaled[col] = 0
                input_df_scaled = input_df_scaled[feature_names]

                if problem_type == "Regression":
                    st.write("### Regression Predictions")
                    lin_reg_pred = lin_reg.predict(input_df_scaled)[0]
                    rf_reg_pred = rf_reg.predict(input_df_scaled)[0]
                    st.write(f"**Linear Regression Prediction**: {lin_reg_pred:.2f}")
                    st.write(f"**Random Forest Regressor Prediction**: {rf_reg_pred:.2f}")

                elif problem_type == "Classification":
                    st.write("### Classification Predictions")
                    log_reg_pred = log_reg.predict(input_df_scaled)[0]
                    rf_clf_pred = rf_clf.predict(input_df_scaled)[0]
                    st.write(f"**Logistic Regression Prediction**: {log_reg_pred}")
                    st.write(f"**Random Forest Classifier Prediction**: {rf_clf_pred}")

else:
    st.info("Upload a dataset to proceed.")

