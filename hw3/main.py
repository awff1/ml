import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree


from sklearn.metrics import (
    accuracy_score,# Доля правильных ответов
    precision_score,# Точность
    recall_score,# Полнота (из всех больных, сколько мы нашли)
    f1_score, # Среднее между Precision и Recall
    roc_auc_score, # Площадь под ROC-кривой (чем ближе к 1 — тем лучше)
    confusion_matrix, # Матрица ошибок
    ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('heart.csv')
df = df.drop_duplicates()

print(f"Строк (пациентов): {df.shape[0]}")
print(f"Столбцов (признаков): {df.shape[1]}")
print("\nПервые 5 строк:")
print(df.head())
print("\nОписание признаков:")
print("""
  age      — возраст
  sex      — пол (1=мужчина, 0=женщина)
  cp       — тип боли в груди (0-3)
  trestbps — артериальное давление в покое
  chol     — холестерин
  fbs      — сахар в крови > 120 мг/дл (1=да)
  restecg  — результаты ЭКГ в покое
  thalach  — максимальная ЧСС
  exang    — стенокардия при нагрузке (1=да)
  oldpeak  — депрессия ST при нагрузке
  slope    — наклон ST-сегмента
  ca       — кол-во крупных сосудов (0-3)
  thal     — тип таласемии
  target   — ЦЕЛЬ: 1=болен, 0=здоров
""")


X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Обучающая выборка: {X_train.shape[0]} пациентов")
print(f"Тестовая выборка:  {X_test.shape[0]} пациентов")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


def print_metrics(name, y_true, y_pred, y_prob=None):
    """Печатает все метрики для одной модели"""
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(f"  Accuracy  (точность):  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision (прецизия):  {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall    (полнота):   {recall_score(y_true, y_pred):.4f}")
    print(f"  F1-score:              {f1_score(y_true, y_pred):.4f}")
    if y_prob is not None:
        print(f"  ROC-AUC:               {roc_auc_score(y_true, y_prob):.4f}")


print("\n" + "=" * 60)
print("ЧАСТЬ 1: МОДЕЛИ С ПАРАМЕТРАМИ ПО УМОЛЧАНИЮ")
print("=" * 60)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)

lr_pred = lr.predict(X_test_sc)
lr_prob = lr.predict_proba(X_test_sc)[:, 1]
print_metrics("Логистическая регрессия (дефолт)", y_test, lr_pred, lr_prob)

svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train_sc, y_train)

svm_pred = svm.predict(X_test_sc)
svm_prob = svm.predict_proba(X_test_sc)[:, 1]
print_metrics("SVM linear (дефолт)", y_test, svm_pred, svm_prob)

svm_rbf = SVC(kernel='rbf', probability=True, random_state=42)
svm_rbf.fit(X_train_sc, y_train)
svm_rbf_pred = svm_rbf.predict(X_test_sc)
svm_rbf_prob = svm_rbf.predict_proba(X_test_sc)[:, 1]
print_metrics("SVM rbf (дефолт)", y_test, svm_rbf_pred, svm_rbf_prob)


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
dt_prob = dt.predict_proba(X_test)[:, 1]
print_metrics("Дерево решений (дефолт)", y_test, dt_pred, dt_prob)


#   TN = правильно определили здоровых (True Negative)
#   FP = здоровых назвали больными (False Positive) — "ложная тревога"
#   FN = больных пропустили (False Negative) — "опасная ошибка"
#   TP = правильно определили больных (True Positive)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Матрицы ошибок (дефолт)', fontsize=14, fontweight='bold')

for ax, (name, pred) in zip(axes, [
    ("Логистич. регрессия", lr_pred),
    ("SVM (linear)",        svm_pred),
    ("Дерево решений",      dt_pred),
]):
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm, display_labels=["Здоров", "Болен"]).plot(ax=ax)
    ax.set_title(name)

plt.tight_layout()
plt.show()


print("ЧАСТЬ 2: ПОДБОР ГИПЕРПАРАМЕТРОВ (GridSearchCV)")


param_grid_lr = {
    'C':       [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver':  ['liblinear', 'saga']
}

grid_lr = GridSearchCV(
    LogisticRegression(max_iter=2000, random_state=42),
    param_grid_lr,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_lr.fit(X_train_sc, y_train)
print(f"\nЛогистич. регрессия — лучшие параметры: {grid_lr.best_params_}")
print(f"  Лучший ROC-AUC (CV): {grid_lr.best_score_:.4f}")

lr_best = grid_lr.best_estimator_
lr_best_pred = lr_best.predict(X_test_sc)
lr_best_prob = lr_best.predict_proba(X_test_sc)[:, 1]
print_metrics("Логистич. регрессия (лучшая)", y_test, lr_best_pred, lr_best_prob)


param_grid_svm = {
    'C':      [0.1, 1, 10, 100],
    'gamma':  ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf', 'poly']
}

grid_svm = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid_svm,
    cv=5, scoring='roc_auc', n_jobs=-1
)
grid_svm.fit(X_train_sc, y_train)
print(f"\nSVM — лучшие параметры: {grid_svm.best_params_}")
print(f"  Лучший ROC-AUC (CV): {grid_svm.best_score_:.4f}")

svm_best = grid_svm.best_estimator_
svm_best_pred = svm_best.predict(X_test_sc)
svm_best_prob = svm_best.predict_proba(X_test_sc)[:, 1]
print_metrics("SVM (лучшая)", y_test, svm_best_pred, svm_best_prob)


param_grid_dt = {
    'max_depth':        [3, 5, 10, None],
    'min_samples_split':[2, 5, 10],
    'criterion':        ['gini', 'entropy']
}

grid_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid_dt,
    cv=5, scoring='roc_auc', n_jobs=-1
)
grid_dt.fit(X_train, y_train)
print(f"\nДерево решений — лучшие параметры: {grid_dt.best_params_}")
print(f"  Лучший ROC-AUC (CV): {grid_dt.best_score_:.4f}")

dt_best = grid_dt.best_estimator_
dt_best_pred = dt_best.predict(X_test)
dt_best_prob = dt_best.predict_proba(X_test)[:, 1]
print_metrics("Дерево решений (лучшее)", y_test, dt_best_pred, dt_best_prob)


print("\n" + "=" * 60)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 60)

results = []
for name, y_pred, y_prob in [
    ("LR дефолт",   lr_pred,       lr_prob),
    ("LR лучшая",   lr_best_pred,  lr_best_prob),
    ("SVM дефолт",  svm_pred,      svm_prob),
    ("SVM лучшая",  svm_best_pred, svm_best_prob),
    ("DT дефолт",   dt_pred,       dt_prob),
    ("DT лучшая",   dt_best_pred,  dt_best_prob),
]:
    results.append({
        'Модель':    name,
        'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall':    round(recall_score(y_test, y_pred), 4),
        'F1':        round(f1_score(y_test, y_pred), 4),
        'ROC-AUC':   round(roc_auc_score(y_test, y_prob), 4),
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))


print("\n" + "=" * 60)
print("ЧАСТЬ 3: ИНТЕРПРЕТАЦИЯ МОДЕЛЕЙ")
print("=" * 60)

coef = lr_best.coef_[0]
feature_names = X.columns.tolist()
coef_df = pd.DataFrame({
    'feature':    feature_names,
    'coefficient': coef,
    'abs_coef':   np.abs(coef)
}).sort_values('abs_coef', ascending=False).head(10)

print("\nТоп-10 признаков (лог. регрессия):")
print(coef_df[['feature', 'coefficient']].to_string(index=False))


fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#d62728' if c > 0 else '#1f77b4' for c in coef_df['coefficient']]
ax.barh(coef_df['feature'][::-1], coef_df['coefficient'][::-1], color=colors[::-1])
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Коэффициент (знак = направление влияния)')
ax.set_title('Топ-10 признаков по влиянию\nЛогистическая регрессия\n'
             '(красный = повышает риск, синий = снижает)')
plt.tight_layout()
plt.show()

svm_linear = SVC(kernel='linear', C=1, probability=True, random_state=42)
svm_linear.fit(X_train_sc, y_train)

svm_coef = svm_linear.coef_[0]
svm_coef_df = pd.DataFrame({
    'feature':  feature_names,
    'weight':   svm_coef,
    'abs':      np.abs(svm_coef)
}).sort_values('abs', ascending=False).head(10)

print("\nТоп-10 признаков (SVM linear):")
print(svm_coef_df[['feature', 'weight']].to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 6))
colors_svm = ['#d62728' if w > 0 else '#1f77b4' for w in svm_coef_df['weight']]
ax.barh(svm_coef_df['feature'][::-1], svm_coef_df['weight'][::-1], color=colors_svm[::-1])
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Вес признака')
ax.set_title('Топ-10 признаков по влиянию\nSVM (linear kernel)')
plt.tight_layout()
plt.show()

dt_viz = DecisionTreeClassifier(max_depth=4, random_state=42,
                                 criterion=grid_dt.best_params_['criterion'])
dt_viz.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(24, 10))
plot_tree(
    dt_viz,
    feature_names=feature_names,
    class_names=['Здоров', 'Болен'],
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax,
    max_depth=4
)
ax.set_title('Дерево решений (max_depth=4)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

root_feature = feature_names[dt_viz.tree_.feature[0]]
root_threshold = dt_viz.tree_.threshold[0]
print(f"\nКорневой признак дерева: '{root_feature}' (порог: {root_threshold:.2f})")
print("Сравниваем с топ-признаками логрег:")
print(coef_df['feature'].tolist())
print(f"→ '{root_feature}' {'входит' if root_feature in coef_df['feature'].values else 'не входит'} в топ-10 лог.регрессии")


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Матрицы ошибок — лучшие модели', fontsize=14, fontweight='bold')

for ax, (name, pred) in zip(axes, [
    ("Логистич. регрессия", lr_best_pred),
    ("SVM (лучшая)",        svm_best_pred),
    ("Дерево решений",      dt_best_pred),
]):
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm, display_labels=["Здоров", "Болен"]).plot(ax=ax)
    ax.set_title(name)

plt.tight_layout()
plt.show()
