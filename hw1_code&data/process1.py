import pandas as pd
pd.set_option('display.max_columns', None)  # 显示所有列

def missing_values(df):
    """
    缺失值情况
    :return: 包含每列缺失值数量和占比的DataFrame
    """
    print("缺失值统计"+"="*50)
    miss_cnt = df.isnull().sum()
    miss_ratio = (df.isnull().mean() * 100).round(2)

    miss_stats = pd.DataFrame({
        '缺失值数量': miss_cnt,
        '缺失值占比': miss_ratio
    }).sort_values('缺失值数量', ascending=False)

    total_cells = df.shape[0] * df.shape[1]
    total_miss = miss_cnt.sum()
    total_miss_ratio = (total_miss / total_cells * 100).round(2)
    print(f"总单元格数：{total_cells}")
    print(f"总缺失值数：{total_miss}")
    print(f"总缺失率：{total_miss_ratio}%")

    return miss_stats

def drop_feature(df, col_todrop):
    """
    删除指定列
    :param col_todrop: 要删除的列名列表
    :return: 删除列后的DataFrame
    """
    print(f"删除指定列"+"="*50)

    # 删除列（errors='ignore'表示若列不存在则忽略,避免报错）
    df_dropped = df.drop(columns=col_todrop, errors='ignore')
    after_cols = df_dropped.shape[1]

    print(f"删除后列数：{after_cols}")

    return df_dropped

def checkMatch_numberOfSkills(df):
    """
    检查HaveWorkedWith列和ComputerSkills列的匹配
    :return: bool
    """
    for _, row in df.iterrows():
        # 获取当前行的 HaveWorkedWith / ComputerSkills数据
        have_worked_with = row['HaveWorkedWith']
        ComputerSkills = row['ComputerSkills']

        # 用 ; 分割字符串，生成技能列表（同时去除前后空格）
        skills_list = [skill.strip() for skill in have_worked_with.split(';')]
        if len(skills_list) != int(ComputerSkills):
            return False

    return True

def trans_HaveWorkedWith(df, class_list):
    """
    处理HaveWorkedWith列，生成重新编码的列
    :param class_list: 目标技能集合
    :return: 新增列后的DataFrame
    """

    # 新建列，默认值为 0
    df['codingLge'] = 0
    df['frontSkills'] = 0
    df['backFrame'] = 0
    df['db'] = 0
    df['deploy'] = 0
    df['tool'] = 0
    df['targetTask'] = 0

    # 遍历每一行，处理 HaveWorkedWith列
    for idx, row in df.iterrows():
        # 获取当前行的 HaveWorkedWith 数据
        have_worked_with = row['HaveWorkedWith']

        # 用 ; 分割字符串，生成技能列表（同时去除前后空格）
        skills_list = [skill.strip() for skill in have_worked_with.split(';')]
        skills_set = set(skills_list)

        if bool(skills_set & class_list[0]):
            df.at[idx, 'codingLge'] = 1
        if bool(skills_set & class_list[1]):
            df.at[idx, 'frontSkills'] = 1
        if bool(skills_set & class_list[2]):
            df.at[idx, 'backFrame'] = 1
        if bool(skills_set & class_list[3]):
            df.at[idx, 'db'] = 1
        if bool(skills_set & class_list[4]):
            df.at[idx, 'deploy'] = 1
        if bool(skills_set & class_list[5]):
            df.at[idx, 'tool'] = 1
        if bool(skills_set & class_list[6]):
            df.at[idx, 'targetTask'] = 1

    return df


if __name__ == "__main__":
    csv_file_path = "data.csv"
    file_encoding = "utf-8"

    df = pd.read_csv(csv_file_path, encoding=file_encoding)
    print(f"原始数据：{df.shape[0]} 行，{df.shape[1]} 列\n")

    # 执行缺失值统计
    miss_result = missing_values(df)
    print(miss_result)

    # drop所有缺失数据
    df = df.dropna()
    # 再次执行缺失值统计检查
    miss_result = missing_values(df)
    print(miss_result)

    # 删除某些特征列
    col_todrop = ['Unnamed: 0', 'Accessibility', 'Employment', 'MentalHealth', 'Country']
    df = drop_feature(df, col_todrop)
    print(df.head())

    # 检查ComputerSkills和HaveWorkedWith的match
    print(f"ComputerSkills和HaveWorkedWith的match结果：{checkMatch_numberOfSkills(df)}")

    # 为HaveWorkedWith列编码
    codingLgeSet = {
            'JavaScript','TypeScript','Python',
        'Java','C#','C++','C','Go','PHP','Ruby',
        'Rust','Kotlin','Swift','Dart','Objective-C',
        'Groovy','R','Scala','Elixir','Erlang',
        'Crystal','F#','Haskell',
        'LISP','Julia','APL','OCaml','Perl','Lua','COBOL',
        'Fortran','Assembly','SAS'
        }
    frontSkillsSet = {
            'HTML/CSS', 'React.js', 'Vue.js', 'Angular', 'Angular.js',
        'jQuery', 'Next.js', 'Nuxt.js', 'Svelte', 'Gatsby', 'Blazor'
        }
    backFrameSet = {
        'Node.js', 'Express', 'Django', 'Flask', 'FastAPI', 'Spring', 'Laravel',
        'Ruby on Rails', 'Symfony', 'ASP.NET',
        'ASP.NET Core', 'Play Framework', 'Phoenix', 'Fastify', 'Deno'
        }
    dbSet = {
        'SQL', 'PostgreSQL', 'MySQL', 'Microsoft SQL Server', 'SQLite', 'MongoDB',
        'Redis', 'Elasticsearch', 'MariaDB',
        'Cassandra', 'DynamoDB', 'IBM DB2', 'Oracle', 'Neo4j', 'Couchbase', 'CouchDB'
        }
    deploySet = {
        'Docker', 'AWS', 'Microsoft Azure', 'Google Cloud Platform', 'Google Cloud',
        'Oracle Cloud Infrastructure',
        'IBM Cloud or Watson', 'OVH', 'Linode', 'DigitalOcean', 'Heroku', 'Firebase',
        'Firebase Realtime Database',
        'Cloud Firestore', 'Kubernetes', 'Terraform', 'Ansible', 'Puppet', 'Chef',
        'VMware', 'OpenStack', 'Pulumi',
        'Managed Hosting', 'Colocation'
        }
    toolSet = {
        'Git', 'npm', 'Yarn', 'Homebrew', 'Bash/Shell', 'PowerShell', 'Flow'
        }
    targetTaskSet = {
        'Unreal Engine', 'Solidity', 'MATLAB', 'Matlab', 'VBA', 'Delphi', 'Xamarin',
        'Drupal', 'Clojure'
        }
    class_lst = [codingLgeSet, frontSkillsSet, backFrameSet, dbSet, deploySet, toolSet, targetTaskSet]
    # 处理HaveWorkedWith列，生成为该列重新编码后的7列
    df = trans_HaveWorkedWith(df, class_list=class_lst)
    #print(df.head())

    # 删除原有的HaveWorkedWith列
    df = drop_feature(df, ['HaveWorkedWith'])
    print(df.head())

    # 存入新的csv
    df.to_csv("data_trans1.csv", encoding=file_encoding, index=False)