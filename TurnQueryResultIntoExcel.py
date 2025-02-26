import pandas as pd
import json
import os
import argparse
from openpyxl import load_workbook

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images and get geographical information.")
    parser.add_argument('--model', choices=['GPT4o', 'Gemini', "Llama", 'Llava'], required=True, help="The model to use.")
    parser.add_argument('--work', choices=['Breadth.xlsx', 'Depth.xlsx'], required=True, help="The Excel file to use (e.g., Breadth.xlsx).")

    return parser.parse_args()



def main():
    # 设置文件夹路径
    global modelname, work_name
    args = parse_arguments()
    modelname = args.model
    work = args.work
    # 原始文件夹名称列表
    work_name = os.path.splitext(work)[0]
    folder_path = f'ExtractTXTIntoJson_{modelname}_{work_name}'
    output_file = f'{modelname}_{work_name}QueryResult.xlsx'

    # 读取 "breadth.xlsx" 文件内容
    breadth_file = 'breadth.xlsx'

    try:
        # 尝试加载现有的工作簿
        wb = load_workbook(breadth_file)
        sheet = wb.active
        print(f"Loaded existing file: {breadth_file}")
    except FileNotFoundError:
        # 如果文件不存在，抛出错误
        print(f"Error: The file {breadth_file} was not found.")
        return

    # 读取工作簿中的数据（假设要读取的工作表数据在第一个工作表）
    df_existing = pd.read_excel(breadth_file)

    # 创建一个空的列表来存储 JSON 数据
    data_list = []

    # 遍历文件夹中的所有 JSON 文件
    for i in range(1, 601):  # 假设最多有600个文件
        file_name = f'extracted{i}.json'
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 修改 JSON 数据的 key 名称
                modified_data = {
                    'continent_answer': data.get('continent', ''),
                    'country_answer': data.get('Country', ''),
                    'city_answer': data.get('City', ''),
                    'street_answer': data.get('Street', '')
                }

                # 将修改后的数据添加到列表中
                data_list.append(modified_data)
        else:
            break  # 如果文件不存在，退出循环

    # 将修改后的 JSON 数据转换为 DataFrame
    df_json = pd.DataFrame(data_list)

    # 将原始数据转换为 DataFrame，准备合并
    df_existing = pd.DataFrame(df_existing)

    # 找到现有数据的最后一列
    last_column = df_existing.shape[1]

    # 将 JSON 数据添加到现有 DataFrame 的空白列
    for idx, column in enumerate(df_json.columns, start=last_column):
        df_existing[column] = df_json.iloc[:, idx - last_column]

    # 将更新后的数据写入新的 Excel 文件
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_existing.to_excel(writer, index=False)

    print(f'Data has been successfully written to {output_file}')


if __name__ == "__main__":
    main()