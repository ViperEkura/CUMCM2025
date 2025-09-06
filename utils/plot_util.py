import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from scipy.stats import gaussian_kde


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 

def plot_distribution(df, column_name, show=False, save=True, save_path=None):
    """绘制指定列的分布直方图"""
    plt.figure(figsize=(9, 6))
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
        if save_path is None:
            save_path = os.path.join('.', 'plots')
            
        os.makedirs(save_path, exist_ok=True)
        col_suffix = column_name + '_distplot.png'
        plt.savefig(os.path.join(save_path, col_suffix))
    
    plt.close()

def plot_boxplot(df, column_names, show=False, save=True, save_path=None, colors=None, show_all_points=True):
    """绘制多列对比箱线图"""
    plt.figure(figsize=(9, 6))
    
    if isinstance(column_names, str):
        column_names = [column_names]
        
    data = [df[col].dropna() for col in column_names]
    
    if colors is None:
        colors = ["#0066FF", "#FF7700", "#008900", '#D62728', '#9467BD']
    
    boxplots = plt.boxplot(data, positions=range(1, len(data)+1), 
                          vert=True, patch_artist=True,
                          labels=column_names, widths=0.5, showfliers=not show_all_points)
    
    for patch, color in zip(boxplots['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    if show_all_points:
        scatter_handles = []
        for i, col_data in enumerate(data):

            x = np.random.normal(i+1, 0.03, size=len(col_data))
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
        if save_path is None:
            save_path = os.path.join('.', 'plots')
            
        os.makedirs(save_path, exist_ok=True)
        col_suffix = '_'.join(column_names) + '_boxplot.png'
        plt.savefig(os.path.join(save_path, col_suffix))

    plt.close()
    
def plot_spearman_heatmap(df, column_names, show=False, save=True, save_path=None):
    """绘制斯皮尔曼相关系数热力图"""
    corr_matrix = df[column_names].corr(method='spearman')
    fig_size = (len(column_names) * 0.8, len(column_names) * 0.8)

    plt.figure(figsize=fig_size)
    
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt=".2f", 
                cmap='coolwarm',
                square=True, 
                cbar_kws={"shrink": .8},
                annot_kws={"size": 10})
    
    plt.title('斯皮尔曼相关系数热力图', fontsize=14, pad=20)
    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.tight_layout()
    
    if show:
        plt.show()
        
    if save:
        if save_path is None:
            save_path = os.path.join('.', 'plots')
            
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'spearman_corr_heatmap.png'))
    
    plt.close()


def plot_qq(residuals, show=False, save=True, save_path=None):
    plt.figure(figsize=(6, 6))
    sm.qqplot(residuals, line='s', ax=plt.gca())
    plt.title('残差QQ图')
    plt.xlabel('理论分位数')
    plt.ylabel('样本分位数')
    
    if show:
        plt.show()
    
    if save:
        if save_path is None:
            save_path = os.path.join('.', 'plots')
            
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'qq_plot_residuals.png'))
    
    plt.close()
    
def plot_predicted_vs_actual(y_true, y_pred, title, xlabel, ylabel, show=False, save=True, save_path=None):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, color='blue')

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-', linewidth=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis('equal')

    if save:
        if save_path is None:
            save_path = os.path.join('.', 'plots')
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{title}.png'), dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    
    plt.close()

def plot_confidence_intervals(df, column_names, confidence_level=0.95, show=False, save=True, save_path=None):
    """绘制多列的置信区间图"""
    if isinstance(column_names, str):
        column_names = [column_names]
    
    means = []
    cis = []
    for col in column_names:
        data = df[col].dropna()
        n = len(data)
        if n == 0:
            raise ValueError(f"Column {col} contains only NaN values.")
        mean = np.mean(data)
        std_err = np.std(data, ddof=1) / np.sqrt(n)
        # 使用t分布计算临界值
        t_crit = scipy.stats.t.ppf((1 + confidence_level) / 2, df=n-1)
        ci_lower = mean - t_crit * std_err
        ci_upper = mean + t_crit * std_err
        means.append(mean)
        cis.append((ci_lower, ci_upper))
    
    plt.figure(figsize=(8, 6))
    x = np.arange(len(column_names))
    # 计算误差范围
    lower_errors = [mean - ci[0] for mean, ci in zip(means, cis)]
    upper_errors = [ci[1] - mean for mean, ci in zip(means, cis)]
    yerr = [lower_errors, upper_errors]
    
    plt.errorbar(x, means, yerr=yerr, fmt='o', capsize=5, color='#0066FF', ecolor='gray', elinewidth=2, alpha=0.8)
    
    plt.xticks(x, column_names)
    plt.title(f'{int(confidence_level*100)}% 置信区间图')
    plt.ylabel('均值及置信区间')
    plt.grid(True, axis='y', alpha=0.5)
    plt.tight_layout()
    
    if show:
        plt.show()
    
    if save:
        if save_path is None:
            save_path = os.path.join('.', 'plots')
        os.makedirs(save_path, exist_ok=True)
        filename = f'confidence_intervals_{confidence_level:.2f}.png'
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_scatter(df, x_col, y_col, show=False, save=True, save_path=None, 
                title=None, xlabel=None, ylabel=None, color="#005794", alpha=0.8):

    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df[y_col], c=color, alpha=alpha, edgecolors='w', linewidth=0.5)

    plt.title(title if title else f'{x_col} vs {y_col}')
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if show:
        plt.show()
        
    if save:
        if save_path is None:
            save_path = os.path.join('.', 'plots')
        os.makedirs(save_path, exist_ok=True)
        filename = f'{x_col}_vs_{y_col}_scatter.png'
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    
    plt.close()
