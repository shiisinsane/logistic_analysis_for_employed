# 基于逻辑回归的求职影响因素分析

## 项目简介

本项目旨在通过对相关数据进行预处理、特征工程，并构建逻辑回归模型，分析影响求职市场上一个个体是否被录用（`Employed`）的关键因素。项目流程包括数据清洗、特征转换、描述性统计及模型训练与评估，最终输出模型预测结果和影响是否被录用的主要特征及其权重。

## 目录结构

```
/
├── data.csv # 原始数据文件
├── process1.py # 数据预处理第一阶段：缺失值处理、特征筛选、处理重要特征
├── process2.py # 数据预处理第二阶段：矛盾数据清洗、特征编码与标准化
├── process_HaveWorkedWith.py # 数据预处理第零阶段：统计HaveWorkedWith列编程语言出现次数
├── logistic.py # 逻辑回归模型训练、评估与特征影响分析
├── data_trans1.csv # process1处理后的中间数据
├── data_ready.csv # process2中清洗后未编码的中间数据
├── data_total.csv # 特征工程完成后的最终用于机器学习训练的数据
├── HaveWorkedWith_calculate.csv # 编程语言统计结果
└── logistic_results/ # 模型评估结果（指标、图表等）
```

## 环境依赖

运行本项目需安装以下重点的Python库：

```
pandas>=1.0.0
scikit-learn>=1.0.0
matplotlib>=3.0.0
seaborn>=0.11.0
```

## 运行步骤

1. **准备原始数据** 
  将原始数据命名为`data.csv`，放置于项目根目录下。
  
2. **第零阶段探索性分析：统计编程语言出现次数（process_HaveWorkedWith.py）** 
  观察数据集可发现这一列较为特殊，长字符串中以“;”隔开了不同的编程语言/技能，因此对这一列进行单独的统计和处理——执行脚本统计`HaveWorkedWith`列中各编程语言的出现次数：
  
  ```bash
  python process_HaveWorkedWith.py
  ```
  
  输出：`HaveWorkedWith_calculate.csv`（编程语言统计结果）。
  
3. **第一阶段预处理（process1.py）** 
  执行脚本完成缺失值处理、无关列删除、`HaveWorkedWith`列特征转换等：
  
  ```bash
  python process1.py
  ```
  
  输出：`data_trans1.csv`（第一阶段处理后的中间数据）。
  
4. **第二阶段预处理（process2.py）** 
  执行脚本完成矛盾数据清洗、去重、描述性统计、特征编码（标签编码/独热编码）及数值特征标准化：
  
  ```bash
  python process2.py
  ```
  
  输出：
  
  `data_ready.csv`（清洗后未编码的中间数据）
  
  `data_total.csv`（特征工程完成后的最终用于机器学习训练的数据）
  
5. **逻辑回归模型训练与评估（logistic.py）** 
  执行脚本完成数据划分、模型训练、性能评估及特征影响分析：
  
  ```bash
  python logistic.py
  ```
  
  输出：
  
  `logistic_results/`目录下的评估指标（准确率、精确率等）、混淆矩阵、ROC曲线及特征影响分析结果。
  

## 模块功能说明

### 1. process_HaveWorkedWith.py

- 解析`HaveWorkedWith`列中的技能列表，统计每种编程语言的出现次数并排序。
  
- 输出统计结果`HaveWorkedWith_calculate.csv`。
  

### 2. process1.py

- **缺失值处理**：统计并删除含缺失值的行，确保数据完整性。
  
- **特征筛选**：删除无关列（`Unnamed: 0`、`Country`等）。
  
- **技能列转换**：将`HaveWorkedWith`列（技能列表）转换为7个二进制特征（`codingLge`、`frontSkills`等），标识是否掌握某类技能。
  
- 输出第一阶段处理后的数据`data_trans1.csv`。
  

### 3. process2.py

- **数据清洗**：
  
  --检查并删除`Age`与`YearsCode`的逻辑矛盾行，判断规则：`Age="<35"`且`YearsCode≥35`。
  
  --检查并删除`YearsCode`与`YearsCodePro`的矛盾行，判断规则：总编码年限<专业编码年限。
  
  --去除重复行。
  
- **描述性统计**：对类别特征（`Age`、`Gender`等）和数值特征（如`YearsCode`、`PreviousSalary`等）进行分布统计。
  
- **特征工程**：
  
  --标签编码：将二元类别特征（`Age`、`MainBranch`）转换为0/1编码。
  
  --独热编码：将多元类别特征（`EdLevel`、`Gender`）转换为独热向量，并删除基准类别避免多重共线性。
  
  --标准化：对数值特征（`YearsCode`等）进行Z-score标准化。
  
- 输出最终建模数据`data_total.csv`。
  

### 4. logistic.py

- **数据划分**：将`data_total.csv`按8:2划分为训练集和测试集。
  
- **模型训练**：使用逻辑回归模型`LogisticRegression`训练，预测`Employed`（是否被录用）。
  
- **模型评估**：计算准确率、精确率、召回率、F1分数、AUC等指标，生成混淆矩阵和ROC曲线。
  
- **特征分析**：提取模型权重，分析对就业影响最大的特征并可视化。
  
- 输出结果至`logistic_results/`目录。
  

## 其他说明

- 预处理阶段生成的中间数据是用于数据质量检查和临时存储，便于后续进一步的统计分析。
  
- 如需调整预处理逻辑或模型参数，可修改对应脚本中的相关函数，如`process2.py`中的编码规则、`logistic.py`中的正则化参数等。
