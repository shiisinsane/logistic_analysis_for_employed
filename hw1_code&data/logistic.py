import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)
import os
from matplotlib.font_manager import FontProperties

plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False
chinese_font = FontProperties(family=["SimHei"], size=10)

# 结果保存路径
RESULT_DIR = "./logistic_results/"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)



def load_and_split_data(file_path):
    """划分训练集和测试集"""
    df = pd.read_csv(file_path)

    # 定义X,y
    X = df.drop('Employed', axis=1)
    y = df['Employed']

    # 划分训练集和测试集，八二开
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"训练集：{X_train.shape[0]} 样本，测试集：{X_test.shape[0]} 样本")
    return X, y, X_train, X_test, y_train, y_test, X.columns


def train_predict_model(X_train, X_test, y_train):
    """训练逻辑回归模型并生成预测结果"""
    model = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 预测结果
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # 录用概率

    return model, y_pred, y_prob


def evaluate_model(y_test, y_pred, y_prob):
    """评估模型性能"""
    # 指标
    metrics = {
        "指标": ["准确率", "精确率", "召回率", "F1分数", "AUC"],
        "数值": [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred),
            roc_auc_score(y_test, y_prob)
        ]
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_df["数值"] = metrics_df["数值"].round(4)
    # 保存指标结果表格
    metrics_df.to_csv(f"{RESULT_DIR}评估指标.csv", index=False, encoding='utf-8-sig')
    print(metrics_df)

    # 分类详细报告
    class_report = classification_report(y_test, y_pred, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_df.to_csv(f"{RESULT_DIR}分类详细报告.csv", index=True, encoding='utf-8-sig')
    print(classification_report(y_test, y_pred))

    # 混淆矩阵画图
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['未录用', '录用'],
                yticklabels=['未录用', '录用'])
    plt.xlabel('预测结果', fontproperties=chinese_font)
    plt.ylabel('实际结果', fontproperties=chinese_font)
    plt.title('混淆矩阵', fontproperties=chinese_font)
    plt.savefig(f"{RESULT_DIR}混淆矩阵.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ROC曲线画图
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {metrics_df[metrics_df["指标"] == "AUC"]["数值"].values[0]}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('假正例率（FPR）', fontproperties=chinese_font)
    plt.ylabel('真正例率（TPR）', fontproperties=chinese_font)
    plt.title('ROC曲线', fontproperties=chinese_font)
    plt.legend(prop=chinese_font)
    plt.savefig(f"{RESULT_DIR}ROC曲线.png", dpi=300, bbox_inches='tight')
    plt.close()

    return metrics_df



def analyze_influencing_factors(model, feature_names):
    """分析特征影响"""
    # 提取特征权重
    coefficients = pd.DataFrame({
        '特征名称': feature_names,
        '权重系数': model.coef_[0],  # 逻辑回归系数
        '影响强度（绝对值）': abs(model.coef_[0])
    })

    # 按影响强度排序
    coefficients = coefficients.sort_values(by='影响强度（绝对值）', ascending=False)

    # 保存特征权重表格
    coefficients.to_csv(f"{RESULT_DIR}特征影响因素分析.csv", index=False, encoding='utf-8-sig')
    print("\n特征影响因素前10名：")
    print(coefficients[['特征名称', '权重系数']].head(10))

    # 可视化特征权重，前15
    top_n = 15
    top_features = coefficients.head(top_n)

    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='权重系数', y='特征名称', data=top_features,
        color='skyblue'
    )

    plt.axvline(x=0, color='black', linestyle='--')  # 零线区分正负影响
    plt.title(f'对录用结果影响最大的{top_n}个特征', fontproperties=chinese_font)
    plt.xlabel('权重系数', fontproperties=chinese_font)
    plt.ylabel('特征名称', fontproperties=chinese_font)
    plt.yticks(fontproperties=chinese_font)  # 确保y轴特征名中文显示
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}特征影响强度.png", dpi=300, bbox_inches='tight')
    plt.close()

    return coefficients


if __name__ == "__main__":
    file_path = "data_total.csv"

    # 加载数据
    X, y, X_train, X_test, y_train, y_test, feature_names = load_and_split_data(file_path)

    # 训练模型并预测
    model, y_pred, y_prob = train_predict_model(X_train, X_test, y_train)

    # 评估模型
    evaluate_model(y_test, y_pred, y_prob)

    # 分析特征
    analyze_influencing_factors(model, feature_names)
