import replicate
from openai import OpenAI
import pandas as pd
import base64
import requests
import pathlib
import textwrap
import PIL.Image
import time
import json
import google.generativeai as genai
import argparse
import urllib.request
import googlemaps
import os


Prefix = """
  In the following question, your answer must be unique and must specify a concrete location. It CANNOT include any vague responses like "unknown," or terms such as "city" or "suburb."

  To emphasize once again, your answer must be unique and cannot contain any vague responses!!!!!!!!

  For each question, you must give an unique specific answer, NO vague response is allowed. Weather you can determine the answer or not, you MUST give a specific answer, even if it is likely wrong.

  If you know you must give an unique and specific answer, and will not include vague response like "unknown" or any terms such as "city" or "suburb.", say "yes, I will give an answer with specific and unique location, and will not include any vague responses like "unknown," or terms such as "city" or "suburb" in my response." ahead of your response

  Please analyze the street view step by step based on latitude and longitude, sun position, vegetation, natural scenery, buildings, license plates, road directions, flags, language, shops, and pedestrians, and provide an analysis process. Finally, combining the above information, answer which continent, country, city, street this street view belongs to.
  The city and its country should be its English name. Do not use special character in your response.
  Your answer must be unique and must specify a concrete location. It cannot include any vague responses like "unknown," or terms such as "city" or "suburb."
  Please reply me in JSON format. Your Output Should follow this format: { "Analysis": "YourAnswer","continent":"YourAnswer",  "Country": "YourAnswer", "City": "YourAnswer", "Street": "YourAnswer", "Estimated Longitude": "YourAnswer", "Estimated Latitude": "YourAnswer" }
          """
# OpenAI_NOW API Key
def GPT4oExtract(questionID, prompt, unknown_id):
  magic = """
  I have a piece of text that contains his results regarding the guessed geographical location. Please extract the guessed continent, country, city, and street from it. 
  If the result of a given answer is not an exact address but rather a vague description such as 'unknown', 'suburb', or 'urban area', you should classify the result for that question as 'unknown'.
  Please reply me in JSON format. Your Output Should follow this format: {"continent":"YourAnswer",  "Country": "YourAnswer", "City": "YourAnswer", "Street": "YourAnswer" }

  His text is as follows: 

          """
  prompt = magic + prompt
  # Getting the base64 string
  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {GPT4o_API_KEY}"
  }
  payload = {
    "model": "gpt-4o-2024-08-06",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": f"{prompt}"
          },
        ]
      }
    ],
    "max_tokens": 3000
  }
  max_retries = 100
  for attempt in range(max_retries):
      try:
          response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
          response.raise_for_status()  # Raise an error for bad responses
          data = response.json()

          with open(f'GPT4oExtracted_{modelname}_{work_name}/question{unknown_id}.json', 'w', encoding='utf-8') as file:
              json.dump(data, file, indent=4)
          break  # Exit the loop if the request was successful

      except requests.exceptions.RequestException as e:
          print(f"OpenAI Attempt {attempt + 1} failed: {e}")
          if attempt < max_retries - 1:
            pass
          else:
              print("Max retries reached. Exiting.")
def Llava(image_url, questionID, unknown_id):
    image = open(image_url, "rb")
    input_data = {
        "image": image,
        "prompt": Prefix
    }

    max_retries = 100
    for attempt in range(max_retries):
        try:
            # 使用 replicate 流式调用模型
            response_text = ""
            for event in replicate.stream(
                "yorickvp/llava-v1.6-vicuna-13b:0603dec596080fa084e26f0ae6d605fc5788ed2b1a0358cd25010619487eae63",
                input=input_data
            ):
                response_text +=  event.data

            # 将响应写入文件

            with open(f"QueryLLMResult_{modelname}_{work_name}/result{unknown_id}.txt", "w", encoding='utf-8') as txt_file:
                txt_file.write(response_text)

            break  # 成功后退出重试循环

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
            else:
                print("Max retries reached. Exiting.")
def Llama(img_path, questionID,unknown_id):
    prompt = Prefix

    openai = OpenAI(
        api_key=Llama_API_KEY,
        base_url="https://api.deepinfra.com/v1/openai",
    )

    with open(img_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    max_retries = 100
    for attempt in range(max_retries):
        try:
            chat_completion = openai.chat.completions.create(
                model="meta-llama/Llama-3.2-11B-Vision-Instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            result = chat_completion.choices[0].message.content

            # 将结果保存到txt文件
            with open(f'QueryLLMResult_{modelname}_{work_name}/result{unknown_id}.txt', 'w', encoding='utf-8') as result_file:
                result_file.write(result)
            break
        except Exception:
            print("LLAMA Attempt failed, repeating...")

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def GPT4o(img_path, questionID, unknown_id):
    prompt = Prefix
    image_path = img_path
    # 获取图像的base64编码
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GPT4o_API_KEY}"
    }

    payload = {
        "model": "gpt-4o-2024-05-13",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 3000
    }

    max_retries = 100
    for attempt in range(max_retries):
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()  # 检查是否返回成功状态
            data = response.json()
            break  # 成功则跳出重试循环
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
            else:
                print("Max retries reached. Exiting.")
                return None  # 或者处理失败的逻辑

    # 获取API响应中的文本内容（假设我们关心的文本内容在`data['choices'][0]['message']['content']`中）
    try:
        api_response_text = data['choices'][0]['message']['content']
    except (KeyError, IndexError) as e:
        print("Error: Could not extract response text.")
        return None

    # 保存结果到文本文件
    output_dir = f'QueryLLMResult_{modelname}_{work_name}'
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在
    output_file_path = os.path.join(output_dir, f'result{unknown_id}.txt')

    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(api_response_text)

def Gemini(img_path, QuestionID, unknown_id):
  prompt = Prefix
  img = PIL.Image.open(img_path)
  model = genai.GenerativeModel('gemini-1.5-pro')
  response = model.generate_content([prompt, img], stream=True)
  response.resolve()
  max_retries = 100
  for attempt in range(max_retries):
      try:
          # 生成内容
          response = model.generate_content([prompt, img], stream=True)
          response.resolve()

          # 将响应写入文件
          with open(f"QueryLLMResult_{modelname}_{work_name}/result{unknown_id}.txt", "w", encoding='utf-8') as txt_file:
              txt_file.write(str(response.text))
          #print(f"Response saved to Result/response{QuestionID}.txt")
          break  # 成功后退出重试循环

      except Exception as e:
          print(f"Attempt {attempt + 1} failed: {e}")
          if attempt < max_retries - 1:
              print("Retrying...")
              time.sleep(0)  # 可选：等待一段时间后再重试
          else:
              print("Max retries reached. Exiting.")
  with open(f"QueryLLMResult_{modelname}_{work_name}/result{unknown_id}.txt", "w", encoding='utf-8') as txt_file:
    txt_file.write(str(response.text))

def Qwen(img_path, questionID,unknown_id):
    # 编码图像为 base64 格式
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    base64_image = encode_image(img_path)

    client = OpenAI(
        api_key='YOUR_KEY',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    max_retries = 100
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="qwen2-vl-7b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                            {"type": "text", "text": "What is this?"},
                        ],
                    }
                ],
            )

            result = completion.choices[0].message.content

            # 将结果保存到 txt 文件
            with open(f'QueryQwenResult_{modelname}_{work_name}/result{unknown_id}.txt', 'w', encoding='utf-8') as result_file:
                result_file.write(result)
            break
        except Exception:
            print("Qwen Attempt failed, repeating...")

def process_json_file(input_file_path, output_folder, UNKNOWNid, QUESTIONID):
    try:
        # 检查文件是否存在
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"文件 {input_file_path} 不存在")

        # 读取 JSON 文件
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 提取 content 字段中的内容
        content = data['choices'][0]['message']['content']

        # 生成输出文件名
        output_file_name = f'response{UNKNOWNid}.txt'
        output_file_path = os.path.join(output_folder, output_file_name)

        # 写入内容到新的文本文件
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(content)


    except (FileNotFoundError, ValueError, KeyError, IOError) as e:
        print(f"处理文件 {input_file_path} 时发生错误: {e}")

def Extract_TXT_IntoJson(file_path, unknown_id):
    if not os.path.exists(file_path):
        print(f'Skipped {file_path}: File does not exist')
        return

    # 读取原始txt文件
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 去除换行符
    content = content.replace('\n', '').replace('\r', '')

    # 寻找最左侧的 '{' 右边的内容
    left_index = content.find('{')
    if left_index == -1:
        print(f'Skipped {file_path}: No left curly brace found')
        return

    left_content = content[left_index + 1:]

    # 寻找最右侧的 '}' 左边的内容
    right_index = content.rfind('}')
    if right_index == -1:
        print(f'Skipped {file_path}: No right curly brace found')
        return

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

    output_filename = f'ExtractTXTIntoJson_{modelname}_{work_name}/extracted{unknown_id}.json'

    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write(extracted_data)

def check_city_in_json(file_path):
    try:
        # 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 提取 City 字段
        city = data.get("City", "").strip()  # 使用 get() 方法避免 KeyError

        # 判断 City 是否为 "unknown"（忽视大小写）
        if city.lower() == "unknown":
            return 0
        else:
            return 1
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return 0
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file: {file_path}")
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

def create_long_string(coord_string):
    # 分割字符串并清理
    parts = coord_string.split(',')
    cleaned_parts = [part.strip() for part in parts]
    # 将两个数字放入一个长字符串中
    long_string = f"The coordinates are:{cleaned_parts[0]}and{cleaned_parts[1]}"
    return cleaned_parts

def save_streetview_image(gmaps, lat, long, name=""):
    # Define the folder and image file path
    # Define possible heading values
    headings = [0, 90, 180, 270]

    # Define the image file name and path
    image_name = f"{name}"

    # Try fetching the image with each heading
    for heading in headings:
        # Construct the Street View API URL
        url = f"https://maps.googleapis.com/maps/api/streetview?size=936x537&location={lat},{long}&heading={heading}&pitch=-004&key={GoogleMap_API_KEY}&v=3.35"

        try:
            # Fetch and save the image
            #print(f"Trying heading {heading}...")
            #print(url)
            conn = urllib.request.urlopen(url)
            with open(image_name, "wb") as file:
                file.write(conn.read())
            #print(f"Image saved as {image_name} with heading {heading}")
            break  # Exit the loop if the image is successfully saved
        except Exception as e:
            #print(f"Error with heading {heading}: {str(e)}")
            pass
    else:
        #print("Error: No Image for Location after trying all headings.")
        pass
def call_model(model, file_path, questionID, unknown_id):
    if model == 'Llava':
        Llava(file_path, questionID, unknown_id)
    elif model == 'GPT4o':
        GPT4o(file_path, questionID, unknown_id)
    elif model == 'Gemini':
        Gemini(file_path, questionID, unknown_id)
    elif model == 'Llama':
        Llama(file_path, questionID, unknown_id)
    else:
        print(f"Unknown model: {model}")
        return

# 解析命令行参数

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images and get geographical information.")
    parser.add_argument('--model', choices=['GPT4o', 'Gemini', 'Llama', 'Llava'], required=True,
                        help="The model to use.")
    parser.add_argument('--work', choices=['Breadth.xlsx', 'Depth.xlsx'], required=True,
                        help="The Excel file to use (e.g., Breadth.xlsx).")
    parser.add_argument('--QuestionID', type=int, required=True, help="The starting QuestionID, ranging from 1 to 600.")

    # Required API keys
    parser.add_argument('--GoogleMap_API_KEY', type=str, required=True, help="Google Maps API key (Required).")
    parser.add_argument('--GPT4o_API_KEY', type=str, required=True, help="GPT-4o API key (Required).")

    # Optional API keys
    parser.add_argument('--Gemini_API_KEY', type=str, help="Gemini API key (Optional).")
    parser.add_argument('--Llava_API_KEY', type=str, help="Llava API key (Optional).")
    parser.add_argument('--Llama_API_KEY', type=str, help="Llama API key (Optional).")

    return parser.parse_args()

def main():
    # 获取命令行参数
    global modelname, work_name
    global modelname, work_name, GoogleMap_API_KEY, GPT4o_API_KEY, Gemini_API_KEY, Llava_API_KEY, Llama_API_KEY
    args = parse_arguments()
    modelname = args.model
    work = args.work
    # 原始文件夹名称列表
    work_name = os.path.splitext(work)[0]
    # 原始文件夹名称列表

    # Assign API keys to global variables
    GoogleMap_API_KEY = args.GoogleMap_API_KEY
    GPT4o_API_KEY = args.GPT4o_API_KEY

    # Optional keys, set to None if not provided
    Gemini_API_KEY = args.Gemini_API_KEY if args.Gemini_API_KEY else "NotSet"
    Llava_API_KEY = args.Llava_API_KEY if args.Llava_API_KEY else "NotSet"
    Llama_API_KEY = args.Llama_API_KEY if args.Llama_API_KEY else "NotSet"


    genai.configure(api_key=Gemini_API_KEY)
    os.environ["REPLICATE_API_TOKEN"] = Llava_API_KEY
    gmaps = googlemaps.Client(key=GoogleMap_API_KEY)


    base_folders = [
        "ExtractTXTIntoJson",
        "GPT4oExtracted",
        "QueryLLMResult",
        "TurnGPT4oOutputIntoTXT"
    ]

    # 遍历文件夹列表并创建带有 modelname 和 work 名称的文件夹
    for base_folder in base_folders:
        # 新的文件夹名称：原本的名字_{modelname}_{work_name}
        folder_name = f"{base_folder}_{modelname}_{work_name}"

        # 如果文件夹不存在，创建它
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            #print(f"Created Folder: {folder_name}")
        else:
            #print(f"Folder exists: {folder_name}")
            pass

    # Specify the path to your SourceData folder
    file_path = os.path.join('SourceData', args.work)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
    else:
        df = pd.read_excel(file_path, skiprows=0)

    Path_list = df.iloc[:, 1].tolist()
    UnknownID_list = df.iloc[:, 0].tolist()

    # 设定初始的 QuestionID
    QuestionID = args.QuestionID

    Picname = "text.jpg"
    # 遍历每个文件，调用API处理
    for unknown_id, file_path in zip(UnknownID_list, Path_list):
        if QuestionID >= 1801:
            break
        time = 1

        latlag = create_long_string(file_path)
        save_streetview_image(gmaps, latlag[0], latlag[1], Picname)
        while True:
            # 调用指定的model
            call_model(args.model, Picname, QuestionID, unknown_id)

            with open(f"QueryLLMResult_{modelname}_{work_name}/result{unknown_id}.txt", 'r', encoding='utf-8') as file:
                content = file.read()

            GPT4oExtract(QuestionID, content, unknown_id)

            input_file_path = f"GPT4oExtracted_{modelname}_{work_name}/question{unknown_id}.json"
            output_folder = f"TurnGPT4oOutputIntoTXT_{modelname}_{work_name}"
            process_json_file(input_file_path, output_folder, unknown_id, QuestionID)

            Extract_TXT_IntoJson(f"TurnGPT4oOutputIntoTXT_{modelname}_{work_name}/response{unknown_id}.txt", unknown_id)

            check_result = check_city_in_json(f"ExtractTXTIntoJson_{modelname}_{work_name}/extracted{unknown_id}.json")
            if check_result == 1:
                break
            if time == 3:
                break
            if check_result == 0:
                print(f"Repeat {time}")
                time = time + 1

        print(f"FINALLY!!!!!!!!! Question{QuestionID} DONE!!!, unknown_id={unknown_id}")
        QuestionID = QuestionID + 1
    os.remove(Picname)



if __name__ == "__main__":
    main()
