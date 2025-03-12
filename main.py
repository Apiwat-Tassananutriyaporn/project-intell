from sklearn.calibration import LinearSVC
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow import keras

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

import os
from tensorflow.keras.utils import to_categorical
from keras import layers, models

from PIL import Image



st.title("Intelligent System")

file_path2 = "health_dataset2.csv"
df2 = pd.read_csv(file_path2)

# Insert containers separated into tabs:
tab1, tab3, tab5, tab4, tab6 = st.tabs(["Machine Learing Explain","KNN & SVM","Neural Network explain", "Neural Network", "Reference"])
tab1.write("# Machine Learing Explain ")
tab4.write("# CNN Model")
tab5.write("# Neural Network Explain")



with tab1:
    st.write("## Data Cleansing")
    st.write(" ##### Missing Values: ")
    st.write("We handle Missing Values ​​by calculating the average of that column and replacing the Missing Values ​​with the average.")
    
    st.write(" ######  How to handle Missing Values: ")
    code_MissingValues = ''' 
        for col in ["BMI", "BloodPressure", "Cholesterol", "ExerciseHours"]:
            df2[col].fillna(df2[col].mean(), inplace=True)
    '''
    st.code(code_MissingValues)

    st.write(" #####  Outlier: ")
    st.write("We handle Outlier Values ​​by calculating the median of that column and replacing the Outlier Values ​​with the median.")
    st.write(" ######  How to handle Outlier: ")
    code_Outlier = ''' 
        Q1 = df2["BloodPressure"].quantile(0.25)
        Q3 = df2["BloodPressure"].quantile(0.75)
        IQR = Q3 - Q1
        outlier_condition = (df2["BloodPressure"] < (Q1 - 1.5 * IQR)) | (df2["BloodPressure"] > (Q3 + 1.5 * IQR))
        medianBP = df2["BloodPressure"].median()
        df2.loc[outlier_condition, "BloodPressure"] = medianBP

        Q1 = df2["Cholesterol"].quantile(0.25)
        Q3 = df2["Cholesterol"].quantile(0.75)
        IQR = Q3 - Q1
        outlier_condition = (df2["Cholesterol"] < (Q1 - 1.5 * IQR)) | (df2["Cholesterol"] > (Q3 + 1.5 * IQR))
        medianCH = df2["Cholesterol"].median()
        df2.loc[outlier_condition, "Cholesterol"] = medianCH
    '''
    st.code(code_Outlier)

    

    st.write("## Dataset")
    st.write(" ### Patient Health Information")
    st.dataframe(df2)
    st.write("by ChatGPT")
    st.write(" ### Feather")
    st.write(" ##### Age:")
    st.write("Age of each patient")
    st.write(" ##### BloodPressure:")
    st.write("Blood pressure value of each patient ")
    st.write(" ##### Cholesterol:")
    st.write("Cholesterol value of each patient. A waxy substance throughout the body of each patient")
    st.write(" ##### BMI:")
    st.write("Body mass index of each patient. BMI is measurement tool to estimate the level of body fat.")
    st.write(" ##### ExerciseHours:")
    st.write("Number of Hours that patient takes for exercise per week")
    st.write(" ##### Diabetes:")
    st.write("Status using for telling patient has Diabete or not. By '0' is not haveing Diabetes '1' is haveing Diabetes ")


# Cleansing Data KNN & SVM

file_path1 = "C:\\Users\\Admin\\Downloads\\workout_fitness_tracker_data.csv"
df1 = pd.read_csv(file_path1)

blood_pressure_avg = df2["BloodPressure"].mean()
ExerciseHours_avg = df2["ExerciseHours"].mean()

#Missing Value Managing
for col in ["BMI", "BloodPressure", "Cholesterol", "ExerciseHours"]:
    df2[col].fillna(df2[col].mean(), inplace=True)

#     # df2["BloodPressure"].fillna(blood_pressure_avg, inplace=True)    

#     #Outlier Managing
Q1 = df2["BloodPressure"].quantile(0.25)
Q3 = df2["BloodPressure"].quantile(0.75)
IQR = Q3 - Q1
outlier_condition = (df2["BloodPressure"] < (Q1 - 1.5 * IQR)) | (df2["BloodPressure"] > (Q3 + 1.5 * IQR))
medianBP = df2["BloodPressure"].median()
df2.loc[outlier_condition, "BloodPressure"] = medianBP


Q1 = df2["Cholesterol"].quantile(0.25)
Q3 = df2["Cholesterol"].quantile(0.75)
IQR = Q3 - Q1
outlier_condition = (df2["Cholesterol"] < (Q1 - 1.5 * IQR)) | (df2["Cholesterol"] > (Q3 + 1.5 * IQR))
medianCH = df2["Cholesterol"].median()
df2.loc[outlier_condition, "Cholesterol"] = medianCH






with tab3:
    st.title("🎯 KNN & SVM Model") 
    

    file_path = "health_dataset2.csv"
    df = pd.read_csv(file_path)

    for col in ["BMI", "BloodPressure", "Cholesterol", "ExerciseHours"]:
        df[col].fillna(df[col].mean(), inplace=True)  # Replace NaN with column mean


    X = df2[["Age", "BloodPressure"]]
    y = df2["Diabetes"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    knn_predicions = knn.predict(X_test)

    # Streamlit UI
    st.title("KNN Classification: Age & Blood Pressure")

    
    # Create meshgrid for decision boundary
    x_min, x_max = X["Age"].min() - 5, X["Age"].max() + 5
    y_min, y_max = X["BloodPressure"].min() - 10, X["BloodPressure"].max() + 10
    # x_min, x_max = X_scaled[:, 0].min() - 0.1, X_scaled[:, 0].max() + 0.1
    # y_min, y_max = X_scaled[:, 1].min() - 0.1, X_scaled[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))  # This creates fig and ax properly


    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    sns.scatterplot(x=X["Age"], y=X["BloodPressure"], hue=y, palette=["blue", "red"], edgecolor="k", ax=ax)

    ax.set_xlabel("Age (Normalized)")
    ax.set_ylabel("Blood Pressure (Normalized)")
    ax.set_title("KNN Decision Boundary")

    st.pyplot(fig)

    

    # Evaluate metrics
    def evaluate_model(name, y_true, y_pred):
        st.write(f" {name}:")
        st.write(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
        st.write(f"Precision: {precision_score(y_true, y_pred):.2f}")
        st.write(f"Recall: {recall_score(y_true, y_pred):.2f}")
        st.write(f"F1-Score: {f1_score(y_true, y_pred):.2f}")
        
    # Evaluate SVM
    evaluate_model("KNN", y_test, knn_predicions)



    # SVMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM

    features_svm = df2[["Age", "BloodPressure"]]
    target_svm = df2["Diabetes"]  # 0 = No, 1 = Yes

    # Normalize features for SVM
    scaler_svm = MinMaxScaler()
    features_svm_scaled = scaler_svm.fit_transform(features_svm)

    # Split dataset
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(features_svm_scaled, target_svm, test_size=0.2, random_state=42)

    # Train SVM model
    model_svm = SVC(kernel="linear")  # Linear Kernel for classification
    model_svm.fit(X_train_svm, y_train_svm)
    svm_predicions = model_svm.predict(X_test_svm)

    # Streamlit UI
    st.title("SVM Classification: Age & Blood Pressure")

    

    # Visualization
    st.subheader("SVM Decision Boundary")
    fig_svm, ax_svm = plt.subplots(figsize=(8, 6))

    real_min = scaler_svm.data_min_
    real_max = scaler_svm.data_max_

    # Create meshgrid for decision boundary
    x_min_svm, x_max_svm = real_min[0] - 5, real_max[0] + 5
    y_min_svm, y_max_svm = real_min[1] - 10, real_max[1] + 10
    xx_svm, yy_svm = np.meshgrid(np.linspace(x_min_svm, x_max_svm, 100), np.linspace(y_min_svm, y_max_svm, 100))

    grid_points = np.c_[xx_svm.ravel(), yy_svm.ravel()]
    grid_real_values = scaler_svm.inverse_transform(grid_points)
    Z_svm = model_svm.predict(grid_real_values)

    Z_svm = Z_svm.reshape(xx_svm.shape)

    ax_svm.contourf(xx_svm, yy_svm, Z_svm, alpha=0.3, cmap="coolwarm")
    real_features_svm = scaler_svm.inverse_transform(features_svm_scaled)

    sns.scatterplot(
        x=real_features_svm[:, 0], 
        y=real_features_svm[:, 1], 
        hue=target_svm, 
        palette=["blue", "red"], 
        edgecolor="w", 
        linewidth=0.8, 
        ax=ax_svm
    )

    ax_svm.set_xlabel("Age (Normalized)")
    ax_svm.set_ylabel("Blood Pressure (Normalized)")
    ax_svm.set_title("SVM Decision Boundary")

    st.pyplot(fig_svm)


    # Evaluate metrics
    def evaluate_model(name, y_true, y_pred):
        st.write(f" {name}:")
        st.write(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
        st.write(f"Precision: {precision_score(y_true, y_pred):.2f}")
        st.write(f"Recall: {recall_score(y_true, y_pred):.2f}")
        st.write(f"F1-Score: {f1_score(y_true, y_pred):.2f}")
        
    # Evaluate SVM
    evaluate_model("SVM", y_test, svm_predicions)

    
    st.write("# Dataset")
    st.write("##### **After cleansing")
    st.dataframe(df2)


with  tab4:
    

    model = tf.keras.models.load_model("cnn_model.keras")
    def preprocess_image(image):
        image = image.resize((128, 128))  # ปรับขนาดให้ตรงกับที่โมเดลต้องการ
        image = np.array(image) / 255.0  # Normalize [0,1]
        image = np.expand_dims(image, axis=0)  # เพิ่มมิติสำหรับ batch
        return image

    st.title("🎯 Multitask CNN: Age Prediction")

    # 📤 Upload image
    # uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "png"])
    uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)

        # 🔍 พรีโปรเซสภาพ
        img_input = preprocess_image(image)

        # 🔥 ทำนายอายุ
        age_pred = model.predict(img_input)[0][0]  # ใช้ [0][0] เพราะ output เป็น single value

        # 🎯 แสดงผลลัพธ์
        st.write(f"🔢 **Predicted Age:** {int(age_pred)} years")

    
with tab5:
    st.write("## Data Preparing")
    #st.write(" ##### Missing Values: ")
    st.write("data that we  use is picture that is picture.jpg because I will predict the age by using people's face. That is a reason why I use picture for training.")
    st.write("Dataset by UTKFace")
    
    st.write(" ##### Loaded data from part1 folder")
    code_dataPrepare = '''
        dataset_path = "part1"
    '''
    st.code(code_dataPrepare)

    st.write(" ##### Separate data from file name")
    st.write("I'm using file name to be tell people's age for each file such as 1_0_0_20161219140623097.jpg. So file name is age_gender_race.jpg. I need to use age for prediction")
    code_separate = '''
        for file in os.listdir(dataset_path):
            if file.endswith(".jpg"):
                try:
                    parts = file.split("_")
                    if len(parts) >= 3:
                        age = int(parts[0])
                        image_paths.append(os.path.join(dataset_path, file))
                        labels.append(age)
                    else:
                        print(f"Skipping file {file} due to incorrect format.")
                except ValueError as e:
                    print(f"Skipping file {file} due to error: {e}")
    '''
    st.code(code_separate)

    st.write(" ##### CNN Model Architecture")
    
    code_Model_Architecture='''
        inputs = keras.Input(shape=(128, 128, 3))

        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)

    '''
    st.code(code_Model_Architecture)

    st.write(" ##### Output layer")
    st.write(" Output is flattened and passed through a fully connected (Dense) layer with 128 neurons.")
    
    code_output_layer = '''
        age_output = layers.Dense(1, activation="linear", name="age")(x)
        model = keras.Model(inputs=inputs, outputs=age_output)
    '''
    st.code(code_output_layer)

    st.write(" ##### Compile Model")
    code_complie = '''
        model.compile(optimizer="adam",
        loss="mean_absolute_error",
        metrics=["mae"])
    '''
    st.code(code_complie)

    st.write(" ##### Train Model")  
    st.write("I use 10 epochs for training cnn model")
    code_train = '''
        model.fit(train_dataset, epochs=10, validation_data=test_dataset)
    '''
    st.code(code_train)
    
    st.write(" ##### Save Model") 
    st.write("Save Model to be file.keras")
    code_saveMdel = '''
        model.save("cnn_model.keras")

    '''
    st.code(code_saveMdel)
    st.write("")
        
with tab6:
    st.write("# ☄️ Dataset for training Machine Learning")  
    st.write("#### By 🎠")
    GPT_button = st.button("Chat GPT")
    if GPT_button:
        st.page_link("https://chat.openai.com", label="Click here to go to ChatGPT.com", icon="🔗")

    st.write("# ☄️ Dataset for training Neural network")  
    st.write("#### By 🎠")
    UTKFace_button = st.button("UTKFace")
    if UTKFace_button:
        st.page_link("https://susanqq.github.io/UTKFace/", label="Click here to go to UTKFace", icon="🔗")


