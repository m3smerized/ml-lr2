import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics

# Загрузка данных
voice_data = pd.read_csv('voice.csv')
X = voice_data.drop('label', axis=1)
y = voice_data['label']

# Разделение данных
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Задаем сетку параметров
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [3, 4, 5, 10]
}

# Задаем метод кросс-валидации
cv = model_selection.StratifiedKFold(n_splits=5)

# Создаем модель для поиска
grid_search = model_selection.GridSearchCV(
    estimator=tree.DecisionTreeClassifier(random_state=0),
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

# Выполняем поиск
grid_search.fit(X_train, y_train)

# Получаем лучшую модель
best_model = grid_search.best_estimator_

# Анализируем результаты
best_params = grid_search.best_params_

# Ответы на вопросы
criterion_map = {
    'gini': 'Критерий Джини',
    'entropy': 'Энтропия Шеннона'
}

# Предсказания и точность
train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)

train_accuracy = round(metrics.accuracy_score(y_train, train_pred), 3)
test_accuracy = round(metrics.accuracy_score(y_test, test_pred), 3)

# Вывод результатов
print("1. Критерий информативности:", criterion_map[best_params['criterion']])
print("2. Оптимальная глубина:", best_params['max_depth'])
print("3. Оптимальное min_samples_split:", best_params['min_samples_split'])
print("4. Accuracy на обучающей выборке:", train_accuracy)
print("   Accuracy на тестовой выборке:", test_accuracy)