import os
import json
import argparse
import pickle

RACE_TRANSLATION = {
    "African": "black",
    "Asian": "asian",
    "Caucasian": "white",
    "Indian": "indian"
}

def save_dict(data: dict, path: str) -> None:
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load_dict(path: str) -> dict:
    with open(path, 'rb') as file:
        return pickle.load(file)

def create_json_files(input_path, output_path):
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                relative_path = os.path.relpath(root, input_path)
                race = relative_path.split(os.sep)[0]

                dominant_race = RACE_TRANSLATION.get(race, "unknown")

                json_content = {
                    "race": {
                        "dominant_race": dominant_race
                    }
                }

                output_dir = os.path.join(output_path, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                json_file_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.json")
                with open(json_file_path, "w") as json_file:
                    print(f'Saving \'{json_file_path}\'')
                    json.dump(json_content, json_file, indent=4)
                
                pkl_file_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.pkl")
                print(f'Saving \'{pkl_file_path}\'')
                save_dict(json_content, pkl_file_path)

                print('--------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSON files for a face dataset.")
    parser.add_argument("--input-path", default='/datasets2/1st_frcsyn_wacv2024/datasets/real/3_BUPT-BalancedFace/race_per_7000_crops_112x112_JUST-PROTOCOL-IMGS', required=True, help="Path to the input dataset directory.")
    parser.add_argument("--output-path", default='', required=False, help="Path to the output directory where JSON files will be saved.")

    args = parser.parse_args()

    if args.output_path == '':
        args.output_path = args.input_path + '_FACE_ATTRIB'

    create_json_files(args.input_path, args.output_path)

    print('\nFinished!\n\n')
