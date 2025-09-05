import os
import pandas as pd
import statsmodels.api as sm

from utils.data_uitl import preprocess, shapiro_test, spearman_test
from utils.plot_util import plot_distribution, plot_boxplot, plot_spearman_heatmap
from utils.regression import BetaRegression
from sklearn.model_selection import train_test_split

from pygam import LinearGAM, s
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def analyze_data(df: pd.DataFrame):
    plot_save_path = 'analyze_plot'
    table_save_path = 'analyze_table'
    os.makedirs(plot_save_path, exist_ok=True)
    os.makedirs(table_save_path, exist_ok=True)
    
    
    plot_distribution(df, '年龄', safe_path=plot_save_path)
    plot_distribution(df, '孕妇BMI', safe_path=plot_save_path)
    plot_boxplot(df, ['X染色体浓度', 'Y染色体浓度'], safe_path=plot_save_path)
    plot_boxplot(df, ['X染色体的Z值', 'Y染色体的Z值'], safe_path=plot_save_path)
    plot_boxplot(df, ['13号染色体的Z值', '18号染色体的Z值', "21号染色体的Z值"], safe_path=plot_save_path)
    plot_boxplot(df, ['13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量'], safe_path=plot_save_path)
    
    selected_col = ['年龄', '孕妇BMI', '原始读段数','唯一比对的读段数', '被过滤掉读段数的比例', '在参考基因组上比对的比例', '重复读段的比例', 
                    'GC含量', 'X染色体浓度', 'Y染色体浓度', 'X染色体的Z值', 'Y染色体的Z值', 
                    '13号染色体的Z值', '18号染色体的Z值', "21号染色体的Z值", 
                    '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量']
    
    shapiro_test_res = shapiro_test(df, selected_col)
    shapiro_test_res.to_excel(os.path.join(table_save_path,'shapiro_test_res.xlsx'))

    selected_col = ["检测孕周"] + selected_col
    plot_spearman_heatmap(df, column_names=selected_col, safe_path=plot_save_path)
    spearman_test_res = spearman_test(df[selected_col], 'Y染色体浓度')
    spearman_test_res.to_excel(os.path.join(table_save_path,'spearman_test_res.xlsx'))


def beta_regression(df: pd.DataFrame):
    x_col = ['年龄', '检测孕周', '孕妇BMI', '原始读段数','唯一比对的读段数', '被过滤掉读段数的比例', 
             '在参考基因组上比对的比例', '重复读段的比例', 'X染色体浓度', 'Y染色体的Z值', '13号染色体的Z值', '18号染色体的Z值']
    y_col = ['Y染色体浓度']

    X = df[x_col].values
    y = df[y_col].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    beta_model = BetaRegression(degree=1)
    beta_model.fit(X_train, y_train)
    
    train_metrics = beta_model.evaluate(X_train, y_train)
    test_metrics = beta_model.evaluate(X_test, y_test)
    print("\n" + "="*50)
    print("Beta回归 模型评估结果")
    print("="*50)
    print(f"训练集评估: R2={train_metrics['R2']:.4f}, MSE={train_metrics['MSE']:.4f}, MAE={train_metrics['MAE']:.4f}")
    print(f"测试集评估: R2={test_metrics['R2']:.4f}, MSE={test_metrics['MSE']:.4f}, MAE={test_metrics['MAE']:.4f}")
    print("-"*50)
    
    print("\nBeta回归 统计检验结果")
    print("="*50)
    
    X_train_sm = sm.add_constant(X_train)
    y_train_flat = y_train.flatten()
    
    model_sm = sm.OLS(y_train_flat, X_train_sm)
    results = model_sm.fit()
    
    f_value = results.fvalue
    f_pvalue = results.f_pvalue
    print(f"F检验: F值={f_value:.4f}, p值={f_pvalue:.4f}")
    
    print("t检验结果:")
    print(results.summary())
    
    return beta_model, results

def gam_regression(df: pd.DataFrame):
    x_col = ['年龄', '检测孕周', '孕妇BMI', '原始读段数','唯一比对的读段数', '被过滤掉读段数的比例', 
             '在参考基因组上比对的比例', '重复读段的比例', 'X染色体浓度', 'Y染色体的Z值', '13号染色体的Z值', '18号染色体的Z值']
    y_col = ['Y染色体浓度']

    X = df[x_col].values
    y = df[y_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gam = LinearGAM(terms=s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + 
                          s(8) + s(9) + s(10) + s(11))

    gam.fit(X_train, y_train)

    y_pred_train = gam.predict(X_train)
    y_pred_test = gam.predict(X_test)

    train_metrics = {
        'R2': r2_score(y_train, y_pred_train),
        'MSE': mean_squared_error(y_train, y_pred_train),
        'MAE': mean_absolute_error(y_train, y_pred_train)
    }
    test_metrics = {
        'R2': r2_score(y_test, y_pred_test),
        'MSE': mean_squared_error(y_test, y_pred_test),
        'MAE': mean_absolute_error(y_test, y_pred_test)
    }

    print("\n" + "="*50)
    print("GAM 模型评估结果")
    print("="*50)
    print(f"训练集: R2={train_metrics['R2']:.4f}, MSE={train_metrics['MSE']:.4f}, MAE={train_metrics['MAE']:.4f}")
    print(f"测试集: R2={test_metrics['R2']:.4f}, MSE={test_metrics['MSE']:.4f}, MAE={test_metrics['MAE']:.4f}")
    print("-"*50)
    
    print("\nGAM 统计检验结果")
    print("="*50)
    
    X_train_sm = sm.add_constant(X_train)
    y_train_flat = y_train.flatten()
    
    model_sm = sm.OLS(y_train_flat, X_train_sm)
    results = model_sm.fit()
    
    # F检验
    f_value = results.fvalue
    f_pvalue = results.f_pvalue
    print(f"F检验: F值={f_value:.4f}, p值={f_pvalue:.4f}")
    
    # t检验
    print("t检验结果:")
    print(results.summary())
    
    return gam, results


if __name__ == "__main__":
    df = pd.read_excel('附件.xlsx', sheet_name=0)
    df = preprocess(df)
    
    analyze_data(df)
    
    # 建立beta 回归模型
    beta_model, beta_results = beta_regression(df)
    
   # 建立 GAM 模型
    gam_model, gam_results = gam_regression(df)