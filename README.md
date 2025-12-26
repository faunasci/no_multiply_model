# Shift-Add Neural Networks Comparison

[English](#english) | [Português](#português) | [Français](#français)

---

## English

This project explores the implementation of **Shift-Add Neural Networks** as an alternative to standard multiplication-based neural networks. Inspired by the Ancient Egyptian multiplication method (multiplication by doubling and adding), this model replaces standard linear layers with a quantized version where weights are powers of two.

### 🚀 Key Concepts
- **Multiplication-free Layers**: By constraining weights to be powers of two ($sign \times 2^n$), multiplications in the forward pass can be replaced by more efficient bit-shift operations and additions.
- **Quantization**: Weights are quantized to the nearest power of two during the forward pass using a Straight-Through Estimator (STE).
- **Hardware Efficiency**: Shift-add operations are significantly more energy-efficient and faster on specialized hardware (FPGAs, custom ASICs).

### 📁 Project Structure
- **`src/`**: Source code containing `add_model.py`.
- **`results/`**: Output directory for metrics and plots.
- `.gitignore` & `requirements.txt`: Standard project configuration.

### �️ Usage
1. `pip install -r requirements.txt`
2. `python src/add_model.py`

---

## Português

Este projeto explora a implementação de **Redes Neurais Shift-Add** como uma alternativa às redes neurais padrão baseadas em multiplicação. Inspirado no método de multiplicação egípcio (multiplicação por dobros e adições), este modelo substitui as camadas lineares padrão por uma versão quantizada onde os pesos são potências de dois.

### 🚀 Conceitos Chave
- **Camadas sem Multiplicação**: Ao restringir os pesos a potências de dois ($sinal \times 2^n$), as multiplicações no "forward pass" podem ser substituídas por operações de bit-shift e adições mais eficientes.
- **Quantização**: Os pesos são quantizados para a potência de dois mais próxima durante o processamento usando um Straight-Through Estimator (STE).
- **Eficiência de Hardware**: Operações de shift-add são significativamente mais eficientes energeticamente e rápidas em hardware especializado (FPGAs, ASICs customizados).

### 📁 Estrutura do Projeto
- **`src/`**: Código fonte contendo `add_model.py`.
- **`results/`**: Diretório de saída para métricas e gráficos.
- `.gitignore` & `requirements.txt`: Configuração padrão do projeto.

---

## Français

Ce projet explore l'implémentation des **Réseaux de Neurones Shift-Add** comme alternative aux réseaux de neurones standard basés sur la multiplication. Inspiré par la méthode de multiplication égyptienne (multiplication par doublement et addition), ce modèle remplace les couches linéaires standard par une version quantifiée où les poids sont des puissances de deux.

### � Concepts Clés
- **Couches sans Multiplication**: En contraignant les poids à être des puissances de deux ($signe \times 2^n$), les multiplications peuvent être remplacées par des opérations de décalage de bits (bit-shift) et des additions plus efficaces.
- **Quantification**: Les poids sont quantifiés à la puissance de deux la plus proche pendant la passe avant en utilisant un "Straight-Through Estimator" (STE).
- **Efficacité Matérielle**: Les opérations shift-add sont nettement plus économes en énergie et plus rapides sur du matériel spécialisé (FPGAs, ASICs personnalisés).

### � Structure du Projet
- **`src/`**: Code source contenant `add_model.py`.
- **`results/`**: Répertoire de sortie pour les métriques et les graphiques.
- `.gitignore` & `requirements.txt`: Configuration standard du projet.
