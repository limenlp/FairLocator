import os
import json
import pandas as pd
import argparse


def process_json_to_txt(input_folder, output_folder, num_files):
    """
    读取 input_folder 中的 JSON 文件，提取 'content' 字段并保存为文本文件
    """
    os.makedirs(output_folder, exist_ok=True)

    for i in range(1, num_files + 1):
        input_file_name = f'question{i}.json'
        input_file_path = os.path.join(input_folder, input_file_name)

        try:
            # 检查文件是否存在
            if not os.path.exists(input_file_path):
                raise FileNotFoundError(f"文件 {input_file_name} 不存在")

            # 读取 JSON 文件
            with open(input_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # 提取 content 字段中的内容
            content = data['choices'][0]['message']['content']

            # 生成输出文件名
            output_file_name = f'result{i}.txt'
            output_file_path = os.path.join(output_folder, output_file_name)

            # 写入内容到新的文本文件
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(content)

            print(f"Question{i} Done!!!")

        except (FileNotFoundError, ValueError, KeyError, IOError) as e:
            print(f"处理文件 {input_file_name} 时发生错误: {e}")


def extract_json_data(input_folder, output_folder, num_files):
    """
    从txt文件中提取内容，获取两部分交集，并保存为新的 JSON 文件
    """
    os.makedirs(output_folder, exist_ok=True)

    for i in range(1, num_files + 1):
        filename = f'{input_folder}/result{i}.txt'
        if not os.path.exists(filename):
            print(f'Skipped {filename}: File does not exist')
            continue

        try:
            # 读取原始txt文件
            with open(filename, 'r') as file:
                content = file.read()

            # 去除换行符
            content = content.replace('\n', '').replace('\r', '')

            # 寻找最左侧的 '{' 右边的内容
            left_index = content.find('{')
            if left_index == -1:
                print(f'Skipped {filename}: No left curly brace found')
                continue

            left_content = content[left_index + 1:]

            # 寻找最右侧的 '}' 左边的内容
            right_index = content.rfind('}')
            if right_index == -1:
                print(f'Skipped {filename}: No right curly brace found')
                continue

            right_content = content[:right_index]

            # 取左右内容的交集
            intersection = ''
            left_index = 0
            right_index = 0

            while left_index < len(left_content) and right_index < len(right_content):
                if left_content[left_index] == right_content[right_index]:
                    intersection += left_content[left_index]
                    left_index += 1
                    right_index += 1
                else:
                    right_index += 1

            # 组合提取的内容并添加大括号
            extracted_data = f'{{{intersection}}}'
            output_filename = f'{output_folder}/extracted{i}.json'

            with open(output_filename, 'w', encoding='utf-8') as output_file:
                output_file.write(extracted_data)
            print(f'Processed {filename} and saved extracted data to {output_filename}')

        except Exception as e:
            print(f"处理文件 {filename} 时发生错误: {e}")


def generate_excel_from_json(input_folder, output_file, num_files):
    """
    从JSON文件夹中读取数据并保存为Excel
    """
    data_list = []

    for i in range(1, num_files + 1):
        filename = f"{input_folder}/extracted{i}.json"
        if os.path.exists(filename):
            with open(filename, "r", encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    lower_case_data = {k.lower(): v for k, v in data.items()}

                    # 移除 'analysis' 键并添加其他键值对到列表中
                    filtered_data = {k: v for k, v in lower_case_data.items() if k != 'analysis'}

                    data_list.append(filtered_data)
                except Exception as e:
                    print(f"Error processing file {filename}: {str(e)}")
                    data_list.append({"status": "Error", "message": str(e)})
        else:
            # 标记文件不存在的情况
            data_list.append({"status": "Not exist"})

    # 创建DataFrame
    df = pd.DataFrame(data_list)

    # 保存为Excel文件
    df.to_excel(output_file, index=False)
    print("提取完成并保存为Excel文件。")


def convert_yes_no_to_1_0(input_file, output_file):
    """
    将 Excel 中的 'Yes' / 'No' 替换为 1 / 0
    """
    # 读取 Excel 文件
    df = pd.read_excel(input_file, engine='openpyxl')

    # 定义替换函数
    def replace_yes_no(value):
        if isinstance(value, str):
            if value.lower() == 'yes':
                return 1
            elif value.lower() == 'no':
                return 0
        return value

    # 应用替换函数到整个 DataFrame
    df = df.applymap(replace_yes_no)

    # 保存处理后的结果到新的 Excel 文件
    df.to_excel(output_file, index=False, engine='openpyxl')

    print(f"处理完成并保存到: {output_file}")


def merge_excel_files(a_file, b_file, output_file):
    # 读取 A.xlsx 和 B.xlsx
    df_a = pd.read_excel(a_file, engine='openpyxl')
    df_b = pd.read_excel(b_file, engine='openpyxl')

    # 创建一个新的 Excel writer
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 将 A.xlsx 的内容写入 C.xlsx 的 Sheet1
        df_a.to_excel(writer, index=False, sheet_name='Sheet1')

        # 找到 A.xlsx 数据的列数，即最后一个空白列的位置
        last_col = len(df_a.columns)

        # 将 B.xlsx 的内容从空白列开始写入 C.xlsx 的 Sheet1
        df_b.to_excel(writer, index=False, startcol=last_col, sheet_name='Sheet1')

    print(f"数据已成功合并到 {output_file}")


def evaluate_exact_match(file_path, output_file_path):
    """
    评估 exact match，并保存结果到新的 Excel 文件
    """
    # 读取 Excel 文件
    df = pd.read_excel(file_path)

    # 进行评估，忽略大小写
    df['continent_match'] = (df['continent'].str.lower() == df['continent_answer'].str.lower()).astype(int)
    df['country_match'] = (df['country'].str.lower() == df['country_answer'].str.lower()).astype(int)
    df['city_match'] = (df['city'].str.lower() == df['city_answer'].str.lower()).astype(int)
    df['street_match'] = (df['street'].str.lower() == df['street_answer'].str.lower()).astype(int)

    # 保存评估结果到新的 Excel 文件
    df.to_excel(output_file_path, index=False)
    print(f"评估结果已保存到新的Excel文件: {output_file_path}")

def compute_union(file_path, output_file_path):
    """
    计算并集，并保存结果到新的 Excel 文件
    """
    # 读取 Excel 文件
    df = pd.read_excel(file_path)

    # 处理 NaN 值，确保所有列都是整数类型
    df['continent_match'] = df['continent_match'].fillna(0).astype(int)
    df['country_match'] = df['country_match'].fillna(0).astype(int)
    df['city_match'] = df['city_match'].fillna(0).astype(int)
    df['street_match'] = df['street_match'].fillna(0).astype(int)

    df['correct_continent'] = df['correct_continent'].fillna(0).astype(int)
    df['correct_country'] = df['correct_country'].fillna(0).astype(int)
    df['correct_city'] = df['correct_city'].fillna(0).astype(int)
    df['correct_street'] = df['correct_street'].fillna(0).astype(int)

    # 取并集
    df['continent_union'] = df['correct_continent'] | df['continent_match']
    df['country_union'] = df['correct_country'] | df['country_match']
    df['city_union'] = df['correct_city'] | df['city_match']
    df['street_union'] = df['correct_street'] | df['street_match']

    # 只保留所需的列
    required_columns = [
        'ID', 'DataPoint', 'continent', 'country', 'city', 'street', 'economies development status',
        'Population', 'CountryGroup', 'continent_answer', 'country_answer', 'city_answer', 'street_answer',
        'continent_union', 'country_union', 'city_union', 'street_union'
    ]

    # 选择需要保留的列
    df = df[required_columns]

    # 保存结果到新的 Excel 文件
    df.to_excel(output_file_path, index=False)
    print(f"并集结果已保存到新的Excel文件: {output_file_path}")


def rename_columns_in_excel(file_name, modified_file_name):
    # 读取Excel文件
    df = pd.read_excel(file_name)
    df.columns = [col.replace('_union', '_correct') if '_union' in col else col for col in df.columns]
    df.to_excel(modified_file_name, index=False)

    print(f"文件已保存为 {modified_file_name}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images and get geographical information.")
    parser.add_argument('--model', choices=['GPT4o', 'Gemini', "Llama", 'Llava'], required=True, help="The model to use.")
    parser.add_argument('--work', choices=['Breadth.xlsx', 'Depth.xlsx'], required=True, help="The Excel file to use (e.g., Breadth.xlsx).")

    return parser.parse_args()

def main():
    global modelname, work_name
    args = parse_arguments()
    modelname = args.model
    work = args.work
    # 原始文件夹名称列表
    work_name = os.path.splitext(work)[0]
    input_folder_step1 = f'Evaluate_Step1_V2_{modelname}_{work_name}'
    output_folder_step0 = f'Evaluate_path_to_step0_result_{modelname}_{work_name}'
    output_folder_step3 = f'Evaluate_Step3_{modelname}_{work_name}'
    num_files = 600

    # 步骤 1: 从 JSON 转到 TXT
    process_json_to_txt(input_folder_step1, output_folder_step0, num_files)

    # 步骤 2: 提取 JSON 数据并保存为新的文件
    extract_json_data(output_folder_step0, output_folder_step3, num_files)

    # 步骤 3: 生成 Excel 文件
    excel_output_file = f'evaluate_result_{modelname}_{work_name}.xlsx'
    generate_excel_from_json(output_folder_step3, excel_output_file, num_files)

    # 步骤 4: 将 Excel 中的 'Yes'/'No' 替换为 1/0
    final_output_file = f'evaluate_result_One_Zero_{modelname}_{work_name}.xlsx'
    convert_yes_no_to_1_0(excel_output_file, final_output_file)

    a_file = f'{modelname}_{work_name}QueryResult.xlsx'
    b_file = f'evaluate_result_One_Zero_{modelname}_{work_name}.xlsx'
    output_file = f'{modelname}_{work_name}Evaluate.xlsx'

    merge_excel_files(a_file, b_file, output_file)
    os.remove(excel_output_file)
    os.remove(final_output_file)


    input_file_path = f'{modelname}_{work_name}Evaluate.xlsx'
    exact_match_output_path = f'{modelname}_{work_name}ExactMatch.xlsx'
    evaluate_exact_match(input_file_path, exact_match_output_path)

    # 计算并集并保存结果
    union_output_path = f'{modelname}_{work_name}EvaluateResultRaw.xlsx'
    compute_union(exact_match_output_path, union_output_path)
    FinalexcelPath = f'{modelname}_{work_name}EvaluateResult.xlsx'
    rename_columns_in_excel(union_output_path, FinalexcelPath)
    os.remove(f'{modelname}_{work_name}Evaluate.xlsx')
    os.remove(f'{modelname}_{work_name}ExactMatch.xlsx')
    os.remove(f'{modelname}_{work_name}EvaluateResultRaw.xlsx')


if __name__ == "__main__":
    main()
