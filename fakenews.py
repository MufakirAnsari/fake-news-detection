# Fake News Detection with Advanced Analysis and Creative Visualizations

# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from collections import Counter
import itertools
import os

# Create a directory to save images
os.makedirs("images", exist_ok=True)

# Step 2: Load the dataset
def load_data(file_path):
    """
    Load the dataset from the given file path.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Step 3: Preprocess the data
def preprocess_data(df):
    """
    Preprocess the dataset by extracting labels and splitting into train/test sets.
    """
    # Drop missing values
    df.dropna(inplace=True)
    
    # Extract labels
    labels = df['label']
    print("Labels extracted successfully.")
    
    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        df['text'], labels, test_size=0.2, random_state=7
    )
    print("Data split into training and testing sets.")
    return x_train, x_test, y_train, y_test

# Step 4: Vectorize the text data
def vectorize_text(x_train, x_test):
    """
    Convert text data into TF-IDF features.
    """
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2))
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)
    print("Text data vectorized successfully.")
    return tfidf_train, tfidf_test, tfidf_vectorizer

# Step 5: Hyperparameter Tuning
def hyperparameter_tuning(x_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('clf', PassiveAggressiveClassifier())
    ])
    
    param_grid = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__max_iter': [50, 100],
        'clf__C': [0.1, 1.0, 10.0]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x_train, y_train)  # Pass raw text data here
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.2f}")
    return grid_search.best_estimator_

# Step 6: Feature Importance Analysis
def analyze_feature_importance(vectorizer, model):
    """
    Analyze the importance of features (words) using model coefficients.
    """
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.named_steps['clf'].coef_[0]
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': coefficients
    }).sort_values(by='Importance', ascending=False)
    
    # Plot top 20 most important features
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(20), x='Importance', y='Feature', palette='viridis')
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig("images/feature_importance.png")  # Save the plot
    plt.show()

# Step 7: Error Analysis
def error_analysis(model, X_test, y_test):
    """
    Analyze misclassified examples to understand model weaknesses.
    """
    y_pred = model.predict(X_test)
    errors = X_test[y_test != y_pred]
    error_labels = y_test[y_test != y_pred]
    
    print(f"Number of Misclassified Examples: {len(errors)}")
    for i, (text, label) in enumerate(zip(errors[:5], error_labels[:5])):
        print(f"\nMisclassified Example {i+1}:\nText: {text}\nTrue Label: {label}")

# Step 8: Generate Advanced Word Clouds
def generate_word_clouds(df):
    """
    Generate advanced word clouds for real and fake news.
    """
    stopwords = set(STOPWORDS)
    real_news = " ".join(text for text in df[df['label'] == 'REAL']['text'])
    fake_news = " ".join(text for text in df[df['label'] == 'FAKE']['text'])
    
    wordcloud_real = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(real_news)
    wordcloud_fake = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(fake_news)
    
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(wordcloud_real, interpolation='bilinear')
    plt.axis('off')
    plt.title('Real News Word Cloud')
    
    plt.subplot(1, 2, 2)
    plt.imshow(wordcloud_fake, interpolation='bilinear')
    plt.axis('off')
    plt.title('Fake News Word Cloud')
    
    plt.tight_layout()
    plt.savefig("images/word_clouds.png")  # Save the plot
    plt.show()

# Step 9: Evaluate the Model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance using various metrics and plots.
    """
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {round(score * 100, 2)}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
    print("Confusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig("images/confusion_matrix.png")  # Save the plot
    plt.show()
    
    # ROC-AUC Curve
    y_prob = model.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(y_test.map({'FAKE': 0, 'REAL': 1}), y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("images/roc_auc_curve.png")  # Save the plot
    plt.show()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test.map({'FAKE': 0, 'REAL': 1}), y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve', color='green')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig("images/precision_recall_curve.png")  # Save the plot
    plt.show()

# Main function to execute the project
def main():
    # Path to the dataset (update this path as per your local setup)
    file_path = "news.csv"
    
    # Step 1: Load the dataset
    df = load_data(file_path)
    if df is None:
        return
    
    # Step 2: Preprocess the data
    x_train, x_test, y_train, y_test = preprocess_data(df)
    
    # Step 3: Vectorize the text data
    tfidf_train, tfidf_test, tfidf_vectorizer = vectorize_text(x_train, x_test)
    
    # Step 4: Hyperparameter tuning
    best_model = hyperparameter_tuning(x_train, y_train)  # Pass raw text data here
    
    # Step 5: Feature importance analysis
    analyze_feature_importance(tfidf_vectorizer, best_model)
    
    # Step 6: Evaluate the model
    evaluate_model(best_model, x_test, y_test)  # Pass raw text data here
    
    # Step 7: Error analysis
    error_analysis(best_model, x_test, y_test)
    
    # Step 8: Generate word clouds
    generate_word_clouds(df)

# Run the project
if __name__ == "__main__":
    main()