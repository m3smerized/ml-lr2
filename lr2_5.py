import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn import model_selection

voice_data = pd.read_csv('voice.csv')
X = voice_data.drop('label', axis=1)
y = voice_data['label']


X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


best_model = tree.DecisionTreeClassifier(
    criterion='gini',
    max_depth=7,
    min_samples_split=3,
    random_state=0
)
best_model.fit(X_train, y_train)


feature_importances = best_model.feature_importances_
features = X.columns


importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)


plt.figure(figsize=(12, 8))
sns.barplot(
    x='Importance', 
    y='Feature', 
    data=importance_df,
    palette='viridis'
)
plt.title('Важность признаков в оптимальном дереве решений')
plt.xlabel('Важность')
plt.ylabel('Признак')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()


target_features = ['meanfreq', 'median', 'IQR', 'meanfun', 'minfun', 'Q25', 'sfm']
target_importance = importance_df[importance_df['Feature'].isin(target_features)]
top3_features = target_importance.nlargest(3, 'Importance')['Feature'].tolist()

print("Топ-3 важных признаков:")
for i, feature in enumerate(top3_features, 1):
    print(f"{i}. {feature}")