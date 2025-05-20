import pandas as pd
import numpy as np
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate distance between two sets of coordinates in Excel.")
    parser.add_argument('--model', choices=['GPT4o', 'Gemini', 'Llama', 'Llava'], required=True, help="The model to use.")
    parser.add_argument('--work', choices=['Breadth.xlsx', 'Depth.xlsx'], required=True, help="The Excel file to use (e.g., Breadth.xlsx).")
    return parser.parse_args()

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points using the Haversine formula.

    Args:
        lat1: Latitude of first point
        lon1: Longitude of first point
        lat2: Latitude of second point
        lon2: Longitude of second point

    Returns:
        float: Distance in kilometers, 20015 if invalid, 0 if identical
    """
    # Check if any coordinate is non-numeric or 'NONE'
    try:
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)
    except (ValueError, TypeError):
        return 20015.0

    # Check if coordinates are identical
    if lat1 == lat2 and lon1 == lon2:
        return 0.0

    # Earth's radius in kilometers
    R = 6371.0

    # Convert to radians and compute Haversine formula
    term = (
        np.cos(np.radians(90 - lat1)) * np.cos(np.radians(90 - lat2)) +
        np.sin(np.radians(90 - lat1)) * np.sin(np.radians(90 - lat2)) * np.cos(np.radians(lon1 - lon2))
    )
    # Clamp the term to [-1, 1] to avoid numerical errors in acos
    term = min(1.0, max(-1.0, term))
    distance = R * np.arccos(term)
    return distance

def process_excel(modelname, work_name):
    """
    Read Excel file, calculate distances based on latitude, longitude, lat, lng,
    and save to a new Excel file with a Distance column.

    Args:
        modelname (str): Model name (e.g., 'GPT4o')
        work_name (str): Workbook name without extension (e.g., 'Breadth')
    """
    input_file = f'{modelname}_{work_name}EvaluateResult_with_latlon.xlsx'
    output_file = f'{modelname}_{work_name}EvaluateResult_with_Distance.xlsx'

    # Read the Excel file
    df = pd.read_excel(input_file)

    # Ensure required columns exist
    required_columns = ['latitude', 'longitude', 'lat', 'lng']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Excel file must contain 'latitude', 'longitude', 'lat', and 'lng' columns")

    # Calculate distance for each row
    df['Distance'] = df.apply(
        lambda row: calculate_distance(
            row['latitude'], row['longitude'], row['lat'], row['lng']
        ), axis=1
    )

    # Save to output Excel file
    df.to_excel(output_file, index=False)
    print(f"Results with Distance column saved to {output_file}")

if __name__ == "__main__":
    args = parse_arguments()
    modelname = args.model
    work_name = os.path.splitext(args.work)[0]

    try:
        process_excel(modelname, work_name)
    except Exception as e:
        print(f"Error processing Excel file: {e}")