import pandas as pd
from utils.data_uitl import preprocess, calcu_first_over_week
from utils.plot_util import plot_scatter
from sklearn.cluster import KMeans
import os

plot_save_path = 'analyze_plot'
table_save_path = 'analyze_table'

if __name__ == '__main__':
    df = pd.read_excel('附件.xlsx', sheet_name=0)
    df = preprocess(df)
    over_week_df = calcu_first_over_week(df, "Y染色体浓度", 0.04)
    
    # 新增聚类分析代码
    # 提取身高体重数据并进行KMeans聚类
    cluster_data = df[['孕妇身高', '孕妇体重']].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(cluster_data)
    
    # 保存聚类结果到表格文件
    os.makedirs(table_save_path, exist_ok=True)
    df.to_excel(os.path.join(table_save_path, 'clustered_data.xlsx'), index=False)
    
    # 可视化聚类结果
    plt.figure(figsize=(8, 6))
    for cluster in df['cluster'].unique():
        subset = df[df['cluster'] == cluster]
        plt.scatter(subset['孕妇身高'], subset['孕妇体重'], label=f'Cluster {cluster}', alpha=0.8)
    
    plt.title('孕妇身高与体重聚类分析')
    plt.xlabel('孕妇身高')
    plt.ylabel('孕妇体重')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    os.makedirs(plot_save_path, exist_ok=True)
    plt.savefig(os.path.join(plot_save_path, 'cluster_scatter.png'))
    plt.close()