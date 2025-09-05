import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 

def plot_distribution(df, column_name, show=False, save=True, safe_path=None):
    """绘制指定列的分布直方图"""
    plt.figure(figsize=(10, 6))
    plt.hist(df[column_name], bins='auto', color='blue', edgecolor='black', alpha=0.7, density=True)
    plt.ylabel('密度')
    

    kde = gaussian_kde(df[column_name].dropna())
    x = np.linspace(df[column_name].min(), df[column_name].max(), 1000)
    plt.plot(x, kde(x), 'r-', lw=2, label='密度曲线(KDE)')
    plt.legend()
    
    plt.title(f'分布直方图：{column_name}')
    plt.xlabel(column_name)
    plt.grid(axis='y', alpha=0.75)
    
    if show:
        plt.show()
        
    if save:
        if safe_path is None:
            safe_path = '.\\'
        plt.savefig(f'{safe_path}{column_name}.png')