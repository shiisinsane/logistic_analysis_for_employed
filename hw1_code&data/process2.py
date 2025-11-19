import pandas as pd
from process1 import missing_values, drop_feature
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 以下是数据预处理，接续process1=============================================================

def check_AgeAndYear(df):
    """
    检查Age列与YearsCode列的逻辑矛盾并清洗：
    Age="<35" 的行，YearsCode不能≥35；否则视为矛盾数据，直接drop
    :return: 删除矛盾数据后的清洗版DataFrame
    """

    # 筛选矛盾行（Age="<35" 且 YearsCode≥35）
    conflict_mask = (df['Age'] == '<35') & (df['YearsCode'] >= 35)
    conflict_count = conflict_mask.sum()

    print(f"Age=<35 且 YearsCode≥35的矛盾行数量：{conflict_count}")

    # 删除矛盾行，保留非矛盾行
    df_cleaned = df[~conflict_mask].copy()
    cleaned_rows = len(df_cleaned)

    print(f"清洗后剩余行数：{cleaned_rows}")


    return df_cleaned

def check_CodingYear(df):
    """
    检查YearsCode与YearsCodePro列的逻辑矛盾并清洗：
    YearsCode（总coding年限）不能小于YearsCodePro（专业coding年限）；否则视为矛盾数据，直接drop
    :return: 删除矛盾数据后的清洗版DataFrame
    """
    # 筛选矛盾行（总编码年限 < 专业编码年限）
    conflict_mask = df['YearsCode'] < df['YearsCodePro']
    conflict_count = conflict_mask.sum()

    print(f"YearsCode < YearsCodePro 的矛盾行数量：{conflict_count}")

    # 删除矛盾行，保留非矛盾行
    df_cleaned = df[~conflict_mask].copy()
    cleaned_rows = len(df_cleaned)

    print(f"清洗后剩余行数：{cleaned_rows}")

    return df_cleaned


def unique(df):
    """
    统计DataFrame中的重复行数，并对数据去重
    :return: 去重后的DataFrame
    """
    # 统计重复行数，duplicated()返回布尔序列，sum()得到重复行总数
    duplicate_count = df.duplicated().sum()
    total_rows_before = len(df)

    print(f"重复行数：{duplicate_count}")
    print(f"重复行占比：{(duplicate_count / total_rows_before * 100):.2f}%")

    # 去重，保留重复行中的第一行
    df_unique = df.drop_duplicates(keep='first')
    total_rows_after = len(df_unique)

    print(f"去重后总行数：{total_rows_after}")

    return df_unique

# 以下是描述性统计=======================================================

def count_classFeatures(df, categorical_cols):
    """
    批量对多个类别列做描述性统计（统计每个列的类别数量、分布）
    :param categorical_cols: 需要统计的类别列列表
    """
    print("类别变量描述性统计")

    for feature in categorical_cols:
        print(f"\n【{feature}列】")

        # 1. 统计唯一类别（去重后查看所有类别）
        unique_vals = df[feature].unique()
        print(f"1. 所有类别：{sorted(unique_vals)}")
        print(f"2. 类别总数：{len(unique_vals)}")

        # 2. 统计每个类别的数量和占比
        counts = df[feature].value_counts(dropna=False)  # 包含NaN的统计
        ratio = df[feature].value_counts(normalize=True, dropna=False) * 100
        stats = pd.DataFrame({
            '数量': counts,
            '占比(%)': ratio.round(2)
        })
        print(stats)

def count_numericFeatures(df, numeric_cols):
    """
    对所有数值变量进行描述性统计（均值、标准差、中位数、最值等）
    :param numeric_cols: 需要统计的数值列列表
    """

    print("数值变量描述性统计")

    stats = df[numeric_cols].describe().round(2)
    # 中位数
    stats.loc['median'] = df[numeric_cols].median().round(2)
    print(stats)
    # 偏度、峰度
    skewness = df[numeric_cols].skew().round(2)
    kurtosis = df[numeric_cols].kurt().round(2)
    print(f"\n偏度（Skewness）：{skewness.to_dict()}")
    print(f"峰度（Kurtosis）：{kurtosis.to_dict()}")


# 以下是特征工程===========================================================

def trans_Age(df):
    """
    对Age列进行标签编码
    编码规则：">35"为1，"<35"为0
    :return: 新增编码列后的DataFrame
    """

    # 编码规则函数
    def encode_age(age):
        if age == ">35":
            return 1
        elif age == "<35":
            return 0

    df['Age'] = df['Age'].apply(encode_age)

    return df

def trans_MainBranch(df):
    """
    对MainBranch列进行标签编码
    编码规则："Dev"为1，"NotDev"为0
    :return: 新增编码列后的DataFrame
    """

    # 编码规则函数
    def encode_MainBranch(MainBranch):
        if MainBranch == "Dev":
            return 1
        elif MainBranch == "NotDev":
            return 0

    df['MainBranch'] = df['MainBranch'].apply(encode_MainBranch)

    return df

def trans_Edlevel(df):
    """
    将EdLevel列转化为独热编码：先新增编码列，再删除原列
    目标类别：['Master', 'NoHigherEd', 'Other', 'PhD', 'Undergraduate']
    :return: 新增独热编码列、删除原EdLevel列后的DataFrame
    """
    # 目标类别
    target_categories = ['Master', 'NoHigherEd', 'Other', 'PhD', 'Undergraduate']

    # 先将EdLevel列转为指定类别的分类数据
    df['EdLevel'] = pd.Categorical(
        df['EdLevel'],
        categories=target_categories,
        ordered=False  # 无顺序关系，仅用于限定类别
    )

    # 生成独热编码
    dummies = pd.get_dummies(
        df['EdLevel'],
        prefix='EdLevel',  # 编码列名前缀：EdLevel_Master、EdLevel_NoHigherEd等
        drop_first=False
    )

    # 新增编码列到原始df
    df = pd.concat([df, dummies], axis=1)
    # 删除原EdLevel列
    df = df.drop(columns=['EdLevel'])

    return df

def trans_Gender(df):
    """
    将Gender列转化为独热编码：先新增编码列，再删除原列
    目标类别：['Man', 'NonBinary', 'Woman']
    :return: 新增独热编码列、删除原Gender列后的DataFrame
    """
    target_categories = ['Man', 'NonBinary', 'Woman']

    df['Gender'] = pd.Categorical(
        df['Gender'],
        categories=target_categories,
        ordered=False
    )

    dummies = pd.get_dummies(
        df['Gender'],
        prefix='Gender',
        drop_first=False
    )

    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=['Gender'])

    return df


def zscore_standardize(df):
    """
    对指定数值列进行Z-score标准化处理
    目标列：YearsCode, YearsCodePro, PreviousSalary, ComputerSkills
    :return: 新增标准化列后的DataFrame
    """
    # 需要标准化的目标列
    target_cols = ['YearsCode', 'YearsCodePro', 'PreviousSalary', 'ComputerSkills']

    # 初始化标准化器
    scaler = StandardScaler()

    # 标准化，返回numpy数组
    scaled_data = scaler.fit_transform(df[target_cols])

    # 将标准化结果转为DataFrame，列名添加前缀z_
    scaled_df = pd.DataFrame(
        scaled_data,
        columns=[f'z_{col}' for col in target_cols],
        index=df.index  # 保持索引与原数据一致
    )

    # drop掉原始数值列，拼接标准化之后的列
    df = df.drop(columns=target_cols).join(scaled_df)

    return df


if __name__ == "__main__":
    csv_file_path = "data_trans1.csv"
    file_encoding = "utf-8"

    df = pd.read_csv(csv_file_path, encoding=file_encoding)
    print(f"原始数据：{df.shape[0]} 行，{df.shape[1]} 列\n")

    # 执行缺失值统计
    miss_result = missing_values(df)
    print(miss_result)

    print(df.head())

    # 清洗不合逻辑的矛盾数据
    df = check_AgeAndYear(df)
    df = check_CodingYear(df)

    # 去重
    df = unique(df)

    # 存入新的csv（此时还未对类别列进行标签/独热编码，以及数值列的标准化）
    df.to_csv("data_ready.csv", encoding=file_encoding, index=False)

    # 对类别变量进行描述性统计
    categorical_cols = [
        'Age', 'EdLevel', 'Gender', 'MainBranch',
        'codingLge', 'frontSkills', 'backFrame', 'db',
        'deploy', 'tool', 'targetTask', 'Employed'
    ]
    count_classFeatures(df, categorical_cols)

    # 对数值变量进行描述性统计
    numeric_cols = ['YearsCode', 'YearsCodePro', 'PreviousSalary', 'ComputerSkills']
    count_numericFeatures(df, numeric_cols)


    # 转换二元类别列为标签编码
    df = trans_Age(df)
    df = trans_MainBranch(df)
    print(df.head())
    # 转换多元类别列为独热编码
    df = trans_Edlevel(df)
    df = trans_Gender(df)
    # 对于独热编码列，drop基准类别
    drop_col = ['Gender_NonBinary', 'EdLevel_Other']
    df = drop_feature(df, drop_col)
    print(df.head())

    # 对数值列进行标准化
    df = zscore_standardize(df)
    print(df.head())

    print(f"编码完成的最终数据：{df.shape[0]} 行，{df.shape[1]} 列\n")

    # 存入新的csv
    df.to_csv("data_total.csv", encoding=file_encoding, index=False)





