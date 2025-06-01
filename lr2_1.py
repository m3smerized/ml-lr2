import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics

voice_data = pd.read_csv('voice.csv')
voice_data.head()

voice_data.isnull().sum().sum()


X = voice_data.drop('label', axis=1)  
y = voice_data['label']  

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# Создание и обучение модели
model = tree.DecisionTreeClassifier(
    max_depth=1, 
    criterion='entropy', 
    random_state=42
)
model.fit(X_train, y_train)

# Визуализация дерева
plt.figure(figsize=(12, 8))
tree.plot_tree(
    model, 
    feature_names=X_train.columns, 
    class_names=['female', 'male'], 
    filled=True, 
    rounded=True
)
plt.savefig('decision_tree.png')
plt.show()

# Анализ дерева
root_feature = X_train.columns[model.tree_.feature[0]]
root_threshold = round(model.tree_.threshold[0], 3)

# Расчет процента наблюдений
root_samples = model.tree_.n_node_samples[0]
left_samples = model.tree_.n_node_samples[1]
percentage_left = round((left_samples / root_samples) * 100, 1)

# Оценка точности
y_pred = model.predict(X_test)
accuracy = round(metrics.accuracy_score(y_test, y_pred), 3)

# Вывод результатов
print("1. Фактор в корневой вершине:", root_feature)
print("2. Пороговое значение:", root_threshold)
print("3. Процент наблюдений:", percentage_left)
print("4. Accuracy на тесте:", accuracy)
