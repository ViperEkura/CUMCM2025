import pandas as pd
from utils.data_uitl import preprocess, shapiro_test
from utils.plot_util import plot_distribution, plot_boxplot

def plot(df, safe_path = 'plots'):
    plot_distribution(df, '年龄', safe_path=safe_path)
    plot_distribution(df, '孕妇BMI', safe_path=safe_path)
    
    plot_boxplot(df, ['X染色体浓度', 'Y染色体浓度'], safe_path=safe_path)
    plot_boxplot(df, ['X染色体的Z值', 'Y染色体的Z值'], safe_path=safe_path)
    plot_boxplot(df, ['13号染色体的Z值', '18号染色体的Z值', "21号染色体的Z值"], safe_path=safe_path)
    plot_boxplot(df, ['13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量'], safe_path=safe_path)


if __name__ == "__main__":
    df = pd.read_excel('附件.xlsx', sheet_name=0)
    df = preprocess(df)
    
    plot(df)
    
    selectd_col = ['年龄', '孕妇BMI', '原始读段数','唯一比对的读段数', '被过滤掉读段数的比例', '在参考基因组上比对的比例', '重复读段的比例', 
                    'GC含量', 'X染色体浓度', 'Y染色体浓度', 'X染色体的Z值', 'Y染色体的Z值', 
                    '13号染色体的Z值', '18号染色体的Z值', "21号染色体的Z值", 
                    '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量']
    
    shapiro_test_res = shapiro_test(df, selectd_col)
    shapiro_test_res.to_excel('shapiro_test_res.xlsx')
    
    
    
    
    
