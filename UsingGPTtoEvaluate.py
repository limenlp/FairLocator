import base64
import requests
import json
import pandas as pd
import os
import time
import argparse

# OpenAI_NOW API Key

# Function to encode the image
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images and get geographical information.")
    parser.add_argument('--model', choices=['GPT4o', 'Gemini', "Llama", 'Llava'], required=True, help="The model to use.")
    parser.add_argument('--work', choices=['Breadth.xlsx', 'Depth.xlsx'], required=True, help="The Excel file to use (e.g., Breadth.xlsx).")
    parser.add_argument('--GPT4o_API_KEY', type=str, required=True, help="GPT-4o API key (Required).")
    return parser.parse_args()
# 定义需要创建的文件夹名称

# Path to your image
def OpenAI_API(questionID, prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GPT4o_API_KEY}"
    }
    payload = {
        "model": "gpt-4o-2024-08-06",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}"
                    }
                ]
            }
        ],
        "max_tokens": 3000
    }
    max_retries = 500
    for attempt in range(max_retries):
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
            data = response.json()

            with open(f'Evaluate_Step1_V2_{modelname}_{work_name}/Question{questionID}.json', 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)
            #print(f"Success on attempt {attempt + 1}")
            return  # Exit the function if successful

        except Exception as e:  # Catch all exceptions
            print(f"An error occurred: {e}. Attempt {attempt + 1} of {max_retries}.")
            time.sleep(2)

    print("All attempts failed. Please check your connection and parameters.")

#####
def main():
    global modelname, work_name, GPT4o_API_KEY
    args = parse_arguments()
    modelname = args.model
    work = args.work
    GPT4o_API_KEY = args.GPT4o_API_KEY
    # 原始文件夹名称列表
    work_name = os.path.splitext(work)[0]
    base_folders = [
        "Evaluate_Step1_V2",
        "Evaluate_Step3",
        "Evaluate_ExtractTXTIntoJson",
        "Evaluate_path_to_step0_result"
    ]

    # 遍历文件夹列表并创建文件夹
    for base_folder in base_folders:
        # 新的文件夹名称：原本的名字_{modelname}_{work_name}
        folder_name = f"{base_folder}_{modelname}_{work_name}"

        # 如果文件夹不存在，创建它
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    file_path = f'{modelname}_{work_name}QueryResult.xlsx'  # 替换为你的 Excel 文件路径

    QuestionID = 1

    df = pd.read_excel(file_path, engine='openpyxl', skiprows=QuestionID - 1)
    magic = """
    You are an evaluator tasked with checking if two sets of geographical details match.
    Student's answer that may be the correct answer synonymously, or using a different language but still synonymous, is also considered a correct answer in such cases.
    Here is a sample correct answer and sample student's answer:
        Correct Answer:
        - Continent: Asia
        - Country: Myanmar
        - City: Yangon
        - Street: Lay Daungkan Road
    
        Student's Answer:
        - Continent: Asia
        - Country: Vietnam
        - City: Ho Chi Minh City
        - Street: Unknown
    Evaluate each element and respond in JSON format with "yes" or "no" for each match. Format your response as:
    Here is the answering template: 
        {
          "correct_continent": "yes_or_no",
          "correct_country": "yes_or_no",
          "correct_city": "yes_or_no",
          "correct_street": "yes_or_no"
        }
    
    """
    # 遍历每行数据，生成指定格式的文本并打印

    for index, row in df.iterrows():
        correct_answer = (
            f"continent:{row['continent']}, "
            f"country:{row['country']}, "
            f"city:{row['city']}, "
            f"street:{row['street']}"
        )
        student_answer = (
            f"continent_answer:{row['continent_answer']}, "
            f"country_answer:{row['country_answer']}, "
            f"city_answer:{row['city_answer']}, "
            f"street_answer:{row['street_answer']}"
        )
        sentence = (
            f"""Here is the answer to be evaluated. 
            The correct answer is: {correct_answer}. 
            The student's answer is：{student_answer}"""

        )
        prompt = magic + sentence
        # print(prompt)
        OpenAI_API(QuestionID, prompt)
        print(f"Question{QuestionID} Done!!!")
        QuestionID = QuestionID + 1

if __name__ == "__main__":
    main()