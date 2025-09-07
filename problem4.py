import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # NA/空值 -> False (0), 有值 -> True (1)
    df["染色体的非整倍体"] = df["染色体的非整倍体"].notna().astype(int)
    df = df.dropna(subset=["孕妇BMI"])
    return df

def run_decision_tree_classification(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3407, stratify=y)
    
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

if __name__ == "__main__":
    x_col = ["孕妇BMI", "原始读段数", "在参考基因组上比对的比例", "重复读段的比例","唯一比对的读段数",
            "GC含量", "13号染色体的Z值", "18号染色体的Z值",  "21号染色体的Z值", "X染色体的Z值",
            "X染色体浓度", "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量", "被过滤掉读段数的比例"]

    y_col = "染色体的非整倍体"
    
    df = pd.read_excel('附件.xlsx', sheet_name=1)
    df = preprocess(df)
    
    # 检查数据分布
    print("目标变量分布:")
    print(df[y_col].value_counts())
    print(f"正样本比例: {df[y_col].mean():.4f}")
    
    X, y = df[x_col], df[y_col]
    model, accuracy, precision, recall, f1, cm, y_pred_proba = run_decision_tree_classification(X, y)
    
    print("\n模型评估指标:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"\n混淆矩阵:\n{cm}")
    
    print("\n=== 特征重要性 ===")
    feature_importance = model.feature_importances_
    
    importance_df = pd.DataFrame({
        '特征': x_col,
        '重要性': feature_importance
    })
    importance_df = importance_df.sort_values('重要性', ascending=False)
    print(importance_df.head())
    
    print(f"\n最重要的5个特征:")
    for i, row in importance_df.head().iterrows():
        print(f"{row['特征']}: {row['重要性']:.4f}")
    

    plt.figure(figsize=(20, 12))
    tree.plot_tree(model, 
                   feature_names=x_col, 
                   class_names=['正常', '非整倍体'],
                   filled=True, 
                   rounded=True) 
    plt.title("决策树结构")
    plt.show()
    

    print("\n=== 决策树规则（文本表示）===")
    tree_rules = tree.export_text(model, feature_names=x_col)
    print(tree_rules)