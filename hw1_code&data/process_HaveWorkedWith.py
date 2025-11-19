import pandas as pd
from collections import Counter

file_path = "data.csv"
df = pd.read_csv(file_path)

# 提取HaveWorkedWith列
have_worked_with = df['HaveWorkedWith']

# 收集所有出现的编程语言
all_languages = []
for item in have_worked_with:
    # 跳过空值
    if pd.notna(item):
        # 按分号分割字符串，获取单个语言
        languages = item.split(';')
        all_languages.extend(languages)

# 统计每种语言的出现次数
language_counts = Counter(all_languages)

# 转换为DataFrame并按出现次数降序排序
result_df = pd.DataFrame(language_counts.items(), columns=['语言', '出现次数'])
result_df = result_df.sort_values(by='出现次数', ascending=False).reset_index(drop=True)

print(f"共{len(result_df)}种不同的语言")
print(result_df.head())

result_df.to_csv('HaveWorkedWith_calculate.csv', index=False, encoding='utf-8-sig')