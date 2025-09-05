import pandas as pd
from utils.data_uitl import preprocess
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
    
