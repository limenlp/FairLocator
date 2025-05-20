import pandas as pd
import requests
import time
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Geocode city and country from Excel and add coordinates.")
    parser.add_argument('--model', choices=['GPT4o', 'Gemini', 'Llama', 'Llava'], required=True, help="The model to use.")
    parser.add_argument('--work', choices=['Breadth.xlsx', 'Depth.xlsx'], required=True, help="The Excel file to use (e.g., Breadth.xlsx).")
    parser.add_argument('--api_key', required=True, help="Google Maps API key.")
    return parser.parse_args()

def get_city_coordinates(country, city, api_key):
    """
    Get latitude and longitude for a country+city using Google Maps Geocoding API.
    Returns the first result if multiple matches exist.

    Args:
        country (str): Country name (e.g., 'USA')
        city (str): City name (e.g., 'Miami')
        api_key (str): Google Maps API key

    Returns:
        tuple: (latitude, longitude) or (None, None) if the request fails
    """
    address = f"{city}, {country}"

    try:
        base_url = 'https://maps.googleapis.com/maps/api/geocode/json'
        params = {
            'address': address,
            'key': api_key
        }

        response = requests.get(base_url, params=params)
        response.raise_for_status()

        data = response.json()

        if data['status'] == 'OK' and len(data['results']) > 0:
            result = data['results'][0]
            latitude = result['geometry']['location']['lat']
            longitude = result['geometry']['location']['lng']
            formatted_address = result['formatted_address']
            print(f"Found: {address} -> {formatted_address} ({latitude}, {longitude})")
            return latitude, longitude
        else:
            print(f"Error: No results for {address} (Status: {data['status']})")
            return None, None

    except requests.exceptions.RequestException as e:
        print(f"Error: Network or API request failed for {address} - {e}")
        return None, None
    except (KeyError, IndexError) as e:
        print(f"Error: Failed to parse response for {address} - {e}")
        return None, None

def process_excel(modelname, work_name, api_key):
    """
    Read Excel file, add latitude and longitude based on country_answer and city_answer,
    and save to a new Excel file.

    Args:
        modelname (str): Model name (e.g., 'GPT4o')
        work_name (str): Workbook name without extension (e.g., 'Breadth')
        api_key (str): Google Maps API key
    """
    input_file = f'{modelname}_{work_name}EvaluateResult.xlsx'
    output_file = f'{modelname}_{work_name}EvaluateResult_with_latlon.xlsx'

    df = pd.read_excel(input_file)

    required_columns = ['country_answer', 'city_answer']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Excel file must contain 'country_answer' and 'city_answer' columns")

    df['latitude'] = None
    df['longitude'] = None

    for index, row in df.iterrows():
        country = str(row['country_answer']).strip()
        city = str(row['city_answer']).strip()

        if not city or city.lower() == 'unknown':
            print(f"Skipping row {index + 2}: city_answer='{city}' (empty or UNKNOWN)")
            df.at[index, 'latitude'] = 'NONE'
            df.at[index, 'longitude'] = 'NONE'
            continue

        if not country or country.lower() == 'none':
            print(f"Skipping row {index + 2}: country_answer='{country}' (empty or NONE)")
            df.at[index, 'latitude'] = 'NONE'
            df.at[index, 'longitude'] = 'NONE'
            continue

        lat, lon = get_city_coordinates(country, city, api_key)
        df.at[index, 'latitude'] = lat if lat is not None else 'NONE'
        df.at[index, 'longitude'] = lon if lon is not None else 'NONE'

        time.sleep(0.1)

    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    args = parse_arguments()
    modelname = args.model
    work_name = os.path.splitext(args.work)[0]
    api_key = args.api_key

    try:
        process_excel(modelname, work_name, api_key)
    except Exception as e:
        print(f"Error processing Excel file: {e}")