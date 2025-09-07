import os
import pandas as pd

from utils.data_util import preprocess, shapiro_test, spearman_test, custom_statistical_tests
from utils.plot_util import plot_distribution, plot_boxplot, plot_spearman_heatmap, plot_qq, plot_predicted_vs_actual
from utils.regression import BetaRegression
from sklearn.model_selection import train_test_split


plot_save_path = 'analyze_plot'
table_save_path = 'analyze_table'
os.makedirs(plot_save_path, exist_ok=True)
os.makedirs(table_save_path, exist_ok=True)


def analyze_data(df: pd.DataFrame):

    plot_distribution(df, '年龄', save_path=plot_save_path)
    plot_distribution(df, '孕妇BMI', save_path=plot_save_path)
    plot_boxplot(df, ['X染色体浓度', 'Y染色体浓度'], save_path=plot_save_path)
    plot_boxplot(df, ['X染色体的Z值', 'Y染色体的Z值'], save_path=plot_save_path)
    plot_boxplot(df, ['13号染色体的Z值', '18号染色体的Z值', "21号染色体的Z值"], save_path=plot_save_path)
    plot_boxplot(df, ['13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量'], save_path=plot_save_path)
    
    selected_col = ['年龄', '孕妇BMI', '原始读段数','唯一比对的读段数', '被过滤掉读段数的比例', '在参考基因组上比对的比例', '重复读段的比例', 
                    'GC含量', 'X染色体浓度', 'Y染色体浓度', 'X染色体的Z值', 'Y染色体的Z值', 
                    '13号染色体的Z值', '18号染色体的Z值', "21号染色体的Z值", 
                    '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量']
    
    shapiro_test_res = shapiro_test(df, selected_col)
    shapiro_test_res.to_excel(os.path.join(table_save_path,'shapiro_test_res.xlsx'))

    selected_col = ["检测孕周"] + selected_col
    plot_spearman_heatmap(df, column_names=selected_col, save_path=plot_save_path)
    spearman_test_res = spearman_test(df[selected_col], 'Y染色体浓度')
    spearman_test_res.to_excel(os.path.join(table_save_path,'spearman_test_res.xlsx'))


def beta_regression(df: pd.DataFrame):
    x_col = ['年龄', '检测孕周', '孕妇BMI', '原始读段数','唯一比对的读段数', '被过滤掉读段数的比例', 
             '在参考基因组上比对的比例', '重复读段的比例', 'X染色体浓度', 'Y染色体的Z值', '13号染色体的Z值', '18号染色体的Z值']
    y_col = ['Y染色体浓度']

    X = df[x_col].values
    y = df[y_col].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    beta_model = BetaRegression()
    beta_model.fit(X_train, y_train)
    
    train_metrics = beta_model.evaluate(X_train, y_train)
    test_metrics = beta_model.evaluate(X_test, y_test)
    
    test_residuals = y_test - test_metrics['y_pred']
    test_residuals_df = pd.DataFrame(test_residuals, columns=['Residuals'])
    plot_qq(test_residuals_df['Residuals'], save_path=plot_save_path)
    shapiro_test_res_test = shapiro_test(test_residuals_df, ['Residuals'])
    shapiro_test_res_test.to_excel(os.path.join(table_save_path, 'shapiro_test_residuals.xlsx'))
    plot_predicted_vs_actual(y_test, test_metrics['y_pred'], '预测值与真实值的关系', '预测值', '真实值', save_path=plot_save_path)
    
    print("\n" + "="*50)
    print("Beta回归 模型评估结果")
    print("="*50)
    print(f"训练集评估: R2={train_metrics['R2']:.4f}, MSE={train_metrics['MSE']:.4f}, MAE={train_metrics['MAE']:.4f}")
    print(f"测试集评估: R2={test_metrics['R2']:.4f}, MSE={test_metrics['MSE']:.4f}, MAE={test_metrics['MAE']:.4f}")
    print("-"*50)
    
    print("\nBeta回归 统计检验结果")
    coef, intercept, feature_names = beta_model.get_coefficients()
    custom_statistical_tests(beta_model, X_test, y_test, feature_names, coef, intercept)
    return beta_model


if __name__ == "__main__":
    df = pd.read_excel('附件.xlsx', sheet_name=0)
    df = preprocess(df)
    
    analyze_data(df)
    beta_regression(df)