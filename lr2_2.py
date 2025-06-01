import pandas as pd
import matplotlib.pyplot as plt
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

# Создание и обучение модели глубины 2
model_depth2 = tree.DecisionTreeClassifier(
    max_depth=2, 
    criterion='entropy', 
    random_state=42
)
model_depth2.fit(X_train, y_train)

# Визуализация дерева
plt.figure(figsize=(16, 10))
tree.plot_tree(
    model_depth2, 
    feature_names=X_train.columns, 
    class_names=model_depth2.classes_,  # Автоматическое определение классов
    filled=True, 
    rounded=True,
    proportion=True
)
plt.savefig('decision_tree_depth2.png')
plt.show()

# Анализ дерева
n_nodes = model_depth2.tree_.node_count
children_left = model_depth2.tree_.children_left
children_right = model_depth2.tree_.children_right

# Подсчет листьев с классом female
female_leaves = 0
for i in range(n_nodes):
    # Проверяем, является ли узел листом
    if children_left[i] == children_right[i]:
        # Получаем предсказанный класс в листе
        class_distribution = model_depth2.tree_.value[i][0]
        predicted_class = model_depth2.classes_[class_distribution.argmax()]
        
        if predicted_class == 'female':
            female_leaves += 1

# Оценка точности
y_pred = model_depth2.predict(X_test)
accuracy = round(metrics.accuracy_score(y_test, y_pred), 3)

# Определение используемых признаков
used_features = set()
for i in range(n_nodes):
    if children_left[i] != children_right[i]:  # Если не лист
        feature = X_train.columns[model_depth2.tree_.feature[i]]
        used_features.add(feature)

print("1. Используемые признаки:", ', '.join(used_features))
print("2. Количество листьев с классом 'female':", female_leaves)
print("3. Accuracy на тесте:", accuracy)