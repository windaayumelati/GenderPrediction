import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import statistics

from web_functions import train_model

# Fungsi untuk membuat heatmap confusion matrix
def plot_confusion_matrix_heatmap(y_true, y_pred, title=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Female', 'Male'], yticklabels=['Female', 'Male'])

    # Set the title if provided
    if title:
        plt.title(title)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot()

def plot_kneighbors_scatter_statistic(df, k, features, x1, x2):
    if len(features) != 2:
        st.warning("Pilih tepat dua fitur untuk visualisasi K-Nearest Neighbors.")
        return

    euclidean_distance = []

    for i in range(df.shape[0]):
        dist = np.sqrt((df[features[0]].iloc[i] - x1) ** 2 + (df[features[1]].iloc[i] - x2) ** 2)
        euclidean_distance.append(dist)

    index = np.argsort(euclidean_distance)
    index = index[:k]
    label = [df.gender.iloc[i] for i in index]
    label = statistics.mode(label)

    palette = sns.color_palette("husl", 2)
    colors = {0: palette[0], 1: palette[1]}

    fig, ax = plt.subplots(figsize=(15, 8))
    sns.scatterplot(data=df, x=features[0], y=features[1], hue='gender', alpha=0.9, s=250, palette=palette, ax=ax)

    for i in index:
        target_value = df.gender.iloc[i]
        if isinstance(target_value, (int, float)):
            color = colors[int(target_value)]
        else:
            color = 'gray'
        ax.scatter(x=df[features[0]].iloc[i], y=df[features[1]].iloc[i], s=250, alpha=0.6, linewidth=2, edgecolor='k', color=color)

    ax.scatter(x=x1, y=x2, s=400, marker='*', color='k')
    ax.set_title(label=f'K-Nearest Neighbor with K = {k}', fontsize=14)
    ax.set_axis_off()
    st.pyplot()

    return f'Predictions: {label}'


# Fungsi untuk membuat ROC Curve
def plot_roc_curve(model, x, y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    y_probs = model.predict_proba(x)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_encoded, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    st.pyplot()

# Fungsi untuk membuat bar plot akurasi
def plot_accuracy(model, x_train, y_train, x_test, y_test):
    train_accuracy = model.score(x_train, y_train) * 100
    test_accuracy = model.score(x_test, y_test) * 100

    plt.figure(figsize=(10, 6))
    plt.bar(['Training Accuracy', 'Testing Accuracy'], [train_accuracy, test_accuracy], color=['skyblue', 'lightgreen'])
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    st.pyplot()

    st.write(f"Training Accuracy: {train_accuracy:.2f}%")
    st.write(f"Testing Accuracy: {test_accuracy:.2f}%")

# Fungsi untuk membuat plot akurasi dan grafik lainnya untuk k-NN
def plot_knn_accuracy(x_train, y_train, x_test, y_test, max_neighbors):
    neighbors = range(1, max_neighbors + 1)
    train_accuracy = []
    test_accuracy = []

    for neighbor in neighbors:
        model = KNeighborsClassifier(n_neighbors=neighbor)
        model.fit(x_train, y_train)
        train_accuracy.append(model.score(x_train, y_train) * 100)
        test_accuracy.append(model.score(x_test, y_test) * 100)

    # Plot k-NN Accuracy
    plt.figure(figsize=(10, 6))
    plt.title('k-NN Varying number of neighbors')
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy', marker='o', linestyle='-', color='orange')
    plt.plot(neighbors, train_accuracy, label='Training accuracy', marker='o', linestyle='-', color='lightblue')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    st.pyplot()

    # Print the accuracy values
    for neighbor, test_acc, train_acc in zip(neighbors, test_accuracy, train_accuracy):
        st.write(f"Neighbors: {neighbor}, Testing Accuracy: {test_acc:.2f}%, Training Accuracy: {train_acc:.2f}%")

    # Plot Confusion Matrix for k-NN with max_neighbors
    model_k = KNeighborsClassifier(n_neighbors=max_neighbors)
    model_k.fit(x_train, y_train)
    y_pred_k = model_k.predict(x_test)

    # Set the title for Confusion Matrix
    conf_matrix_title = f'KNN Classifier Confusion Matrix (Neighbors={max_neighbors})'

    # Plot the confusion matrix
    plot_confusion_matrix_heatmap(y_test, y_pred_k, title=conf_matrix_title)

    # Plot ROC Curve for k-NN with max_neighbors
    model = KNeighborsClassifier(n_neighbors=max_neighbors)
    model.fit(x_train, y_train)
    plot_roc_curve(model, x_test, y_test)

    # Print AUC ROC for k-NN with max_neighbors
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)
    y_probs = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test_encoded, y_probs)
    roc_auc = auc(fpr, tpr)
    st.write(f"AUC ROC for k-NN with {max_neighbors} neighbors: {roc_auc:.4f}")

    # Plot Error Rate vs K
    error_rate = [1 - acc / 100 for acc in test_accuracy]
    plt.figure(figsize=(10, 6))
    plt.plot(neighbors, error_rate, marker='o', linestyle='--', color='red')
    plt.title('Error Rate vs K')
    plt.xlabel('Number of neighbors')
    plt.ylabel('Error Rate')
    st.pyplot()

    # Print Error Rate for each K
    for k, error in zip(neighbors, error_rate):
        st.write(f"K = {k}, Error Rate = {error:.4f}")

# Fungsi utama aplikasi
def app(df, x, y):
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Halaman Visualisasi Prediksi Gender")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

    # Kode untuk mengukur akurasi pada rentang jumlah tetangga dari 1 hingga 9
    test_accuracies = []
    train_accuracies = []

    for n_neighbors in range(1, 10):
        knn = KNeighborsClassifier(n_neighbors)
        knn.fit(x_train, y_train)
        train_accuracies.append(knn.score(x_train, y_train))
        test_accuracies.append(knn.score(x_test, y_test))

    # Generate plots for training and testing accuracy where x is the number of neighbors and y is accuracy
    plt.figure(figsize=(11, 6))
    plt.plot(range(1, 10), train_accuracies, marker='*', label='Train Score')
    plt.plot(range(1, 10), test_accuracies, marker='o', label='Test Score')
    plt.xlabel('Number of neighbors', size='15')
    plt.ylabel('Accuracy', size='15')
    plt.text(7.7, 0.75, 'Here!')
    plt.grid()
    plt.legend()

    if st.checkbox("Plot Confusion Matrix"):
        st.write("Menggunakan data yang dihasilkan dari model")
        model, score = train_model(x_train, y_train)
        y_pred = model.predict(x_test)
        plot_confusion_matrix_heatmap(y_test, y_pred)


    if st.checkbox("Plot K-Nearest Neighbors Statistic"):
        st.write("Menggunakan implementasi k-NN dengan modul statistics")
        st.write("Kustomisasi plot sesuai")

       # Pilih fitur untuk visualisasi
        feature_options = x.columns.tolist()
        selected_features = st.multiselect('Pilih fitur untuk visualisasi',
                                        feature_options,
                                        default=[feature_options[0], feature_options[1]],
                                        key="multiselect_features")  # Add a unique key here

        # Input nilai K yang diinginkan
        k_value = st.slider('Pilih Nilai K', 1, 20, 3)

        # Check the number of selected features
        if len(selected_features) != 2:
            st.warning("Pilih tepat dua fitur untuk visualisasi K-Nearest Neighbors.")
        else:
            # Input nilai x1 dan x2
            x1_value = st.slider('Nilai value x1', float(x[selected_features[0]].min()),
                                float(x[selected_features[0]].max()), float(x[selected_features[0]].min()))  # Set the initial value to the minimum
            x2_value = st.slider('Nilai value x2', float(x[selected_features[1]].min()),
                                float(x[selected_features[1]].max()), float(x[selected_features[1]].min()))  # Set the initial value to the minimum

            result = plot_kneighbors_scatter_statistic(df, k_value, selected_features, x1_value, x2_value)
            st.write(result)


    if st.checkbox("Plot ROC Curve"):
        st.write("Menggunakan data yang dihasilkan dari model")
        model, score = train_model(x_train, y_train)
        plot_roc_curve(model, x_test, y_test)

    if st.checkbox("Plot Accuracy Model"):
        st.write("Menggunakan data yang dihasilkan dari model")
        model, score = train_model(x_train, y_train)
        plot_accuracy(model, x_train, y_train, x_test, y_test)


    if st.checkbox("Plot Berdasarkan Input Nilai K untuk k-NN"):
        st.write("Input Nilai K yang Diinginkan")
        max_neighbors = st.slider('Select nilai K of neighbors', 1, 20, 3)
        plot_knn_accuracy(x_train, y_train, x_test, y_test, max_neighbors)