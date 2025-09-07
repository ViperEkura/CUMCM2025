import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils.data_util import preprocess

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 

def run_decision_tree_classification(X, y, use_grid_search=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=19, stratify=y)
    
    if use_grid_search:
        print("=== 开始网格搜索 ===")
        param_grid = {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 5, 10],
            'criterion': ['gini', 'entropy']
        }
        
        dt = DecisionTreeClassifier(random_state=3407)
        
        grid_search = GridSearchCV(
            estimator=dt,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证分数 (F1): {grid_search.best_score_:.4f}")
        model = grid_search.best_estimator_
        
    else:
        # 使用默认参数
        model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=3407,
            criterion='gini'
        )
        model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, precision, recall, f1, cm, y_pred_proba

def plot_decision_tree_results(model, x_col):
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    sorted_features = [x_col[i] for i in sorted_idx]
    sorted_importance = feature_importance[sorted_idx]

    plt.figure(figsize=(6, 6))
    bars = plt.barh(range(len(sorted_features)), sorted_importance, align='center')
    plt.yticks(range(len(sorted_features)), sorted_features, fontsize=10)
    plt.xlabel('特征重要性', fontsize=12)
    plt.title('决策树特征重要性排序', fontsize=14)
    plt.grid(axis='x', alpha=0.3)

    for i, (bar, importance) in enumerate(zip(bars, sorted_importance)):
        if importance > 0.01:
            plt.text(importance + 0.005, i, f'{importance:.3f}', 
                    va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    if model.tree_.node_count <= 50:
        plt.figure(figsize=(12, 8))
        tree.plot_tree(model, 
                       feature_names=x_col, 
                       class_names=['正常', '非整倍体'],
                       filled=True, 
                       rounded=True,
                       fontsize=8)
        plt.title("最优决策树结构")
        plt.show()
    else:
        print(f"决策树节点数较多 ({model.tree_.node_count})，跳过可视化")

    print("\n=== 最优决策树规则（文本表示）===")
    tree_rules = tree.export_text(model, feature_names=x_col)
    print(tree_rules)

if __name__ == "__main__":
    x_col = ["检测孕周", "孕妇BMI", "原始读段数", "在参考基因组上比对的比例", "重复读段的比例","唯一比对的读段数",
            "GC含量", "13号染色体的Z值", "18号染色体的Z值",  "21号染色体的Z值", "X染色体的Z值",
            "X染色体浓度", "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量", "被过滤掉读段数的比例"]

    y_col = "染色体的非整倍体"
    
    df = pd.read_excel('附件.xlsx', sheet_name=1)
    df = preprocess(df).dropna(subset=["孕妇BMI"])

    print("目标变量分布:")
    print(df[y_col].value_counts())
    print(f"正样本比例: {df[y_col].mean():.4f}")
    
    X, y = df[x_col], df[y_col]
    
    model, accuracy, precision, recall, f1, cm, y_pred_proba = run_decision_tree_classification(
        X, y, use_grid_search=True
    )
    
    print("\n最优模型评估指标:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"\n混淆矩阵:\n{cm}")
    
    plot_decision_tree_results(model, x_col)