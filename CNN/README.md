# Projet Final : Hands-On Machine Learning for Cybersecurity

Ce dépôt contient le code source et les résultats du projet final pour le cours "Hands-On Machine Learning for Cybersecurity".

## 🎯 Objectif du Projet

L'objectif principal de ce projet est de comparer les performances de trois architectures de modèles hybrides appliquées à une problématique de cybersécurité (ici la detection de malware dans un fichier image).

Les trois algorithmes étudiés et comparés sont :

1.  **CNN - SVM** (Convolutional Neural Network - Support Vector Machine)
2.  **GRU - SVM** (Gated Recurrent Unit - Support Vector Machine)
3.  **MLP - SVM** (Multi-Layer Perceptron - Support Vector Machine)

## 🛠️ Ma Contribution

Dans le cadre de ce projet de groupe, ma contribution personnelle s'est concentrée sur l'implémentation, l'entraînement et l'évaluation du modèle **GRU-SVM**.

Le code spécifique à ce modèle (incluant le prétraitement des données, la définition du modèle GRU et l'intégration avec le classificateur SVM) se trouve dans le dossier `/gru_svm` (ou à l'emplacement pertinent de votre projet).

## 🚀 Utilisation

Pour tester ou répliquer les résultats de l'implémentation GRU-SVM :

1.  **Clonez le dépôt :**
    ```bash
    git clone [(https://github.com/ArthurTouati/Machine-learning-for-Cybersecurity/tree/final-project_AT)]
    cd [NOM_DU_DEPOT]
    ```

2.  **Installez les dépendances** (il est recommandé d'utiliser un environnement virtuel) :
    ```bash
    pip install -r requirements.txt
    ```

3.  **Lancez l'expérimentation** (adaptez cette commande à votre script) :
    ```bash
    python ML.py
    ```

## 📊 Résultats


| Modèle | Accuracy | Précision | Rappel | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| CNN-SVM | À compl. | À compl. | À compl. | À compl. |
| **GRU-SVM** | **0.8721** | **0.8950** | **0.8721** | **0.8738** |
| MLP-SVM | À compl. | À compl. | À compl. | À compl. |
