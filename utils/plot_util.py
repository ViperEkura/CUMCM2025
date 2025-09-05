import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 

def plot_distribution(df, column_name, show=False, save=True, safe_path=None):
    """绘制指定列的分布直方图"""
    plt.figure(figsize=(10, 6))
    plt.hist(df[column_name], bins='auto', color='#3282F6', edgecolor='black', alpha=0.7, density=True)
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
    
    plt.close()

def plot_boxplot(df, column_names, show=False, save=True, safe_path=None, colors=None, show_all_points=True):
    """绘制多列对比箱线图"""
    plt.figure(figsize=(12, 7))
    
    if isinstance(column_names, str):
        column_names = [column_names]
        
    data = [df[col].dropna() for col in column_names]
    
    if colors is None:
        colors = ['#3282F6', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD']
    
    boxplots = plt.boxplot(data, positions=range(1, len(data)+1), 
                          vert=True, patch_artist=True,
                          labels=column_names, showfliers=not show_all_points)
    
    for patch, color in zip(boxplots['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    if show_all_points:
        scatter_handles = []
        for i, col_data in enumerate(data):

            x = np.random.normal(i+1, 0.02, size=len(col_data))
            scatter = plt.scatter(x, col_data, color=colors[i % len(colors)], 
                                alpha=0.8, s=25, edgecolors='white', linewidth=0.5, zorder=3)
            scatter_handles.append(scatter)
        

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                      markersize=8, label='全部数据点'),
            boxplots["boxes"][0]
        ]
        plt.legend(handles=legend_elements, labels=["数据点分布", "箱体统计"], loc='best')
    else:
        plt.legend([boxplots["boxes"][0]], ["数据分布"], loc='best')
    
    plt.title('多列数据对比箱线图', fontsize=14, pad=20)
    plt.ylabel('数值分布', fontsize=12)
    plt.xlabel('数据列', fontsize=12)
    plt.grid(axis='y', alpha=0.5, linestyle='--')
    plt.xticks(rotation=45 if len(column_names) > 5 else 0)
    
    plt.tight_layout()
    
    if show:
        plt.show()
        
    if save:
        if safe_path is None:
            safe_path = '.\\'
        col_suffix = '_'.join(column_names)
        plt.savefig(f'{safe_path}{col_suffix}_boxplot.png')
    
    plt.close()