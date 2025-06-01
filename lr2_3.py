import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics

# Загрузка данных
voice_data = pd.read_csv('voice.csv')

# Подготовка данных
X = voice_data.drop('label', axis=1)
y = voice_data['label']

# Разделение данных
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Создание и обучение модели без ограничения глубины
unrestricted_model = tree.DecisionTreeClassifier(
    criterion='entropy',
    random_state=0  # Указываем random_state=0 по заданию
)
unrestricted_model.fit(X_train, y_train)

# 1. Глубина дерева
depth = unrestricted_model.get_depth()

# 2. Количество листьев
n_leaves = unrestricted_model.get_n_leaves()

# 3. Расчет точности
train_pred = unrestricted_model.predict(X_train)
test_pred = unrestricted_model.predict(X_test)

train_accuracy = round(metrics.accuracy_score(y_train, train_pred), 3)
test_accuracy = round(metrics.accuracy_score(y_test, test_pred), 3)

# Вывод результатов
print("1. Глубина дерева:", depth)
print("2. Количество листьев:", n_leaves)
print("3. Accuracy на обучающей выборке:", train_accuracy)
print("   Accuracy на тестовой выборке:", test_accuracy)