import os, sys
import json
import argparse
import pickle


def save_dict(data: dict, path: str) -> None:
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load_dict(path: str) -> dict:
    with open(path, 'rb') as file:
        return pickle.load(file)


def load_attributes_json(path: str) -> dict:
    with open(path, 'r') as file:
        data = json.load(file)
        return data


def save_attributes_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def copy_face_attribs(source_subj_styles, target_samples, output_path):
    for root_src, dirs_src, files_src in os.walk(source_subj_styles):
        for file_src in files_src:
            if file_src.lower().endswith((".json")):
                subj = root_src.split('/')[-1]
                path_source_file = os.path.join(root_src, file_src)
                print('path_source_file:', path_source_file)
                source_attribs = load_attributes_json(path_source_file)
                # print('source_attribs:', source_attribs)
                # print('source_attribs:', source_attribs)
                
                for root_tgt, dirs_tgt, files_tgt in os.walk(os.path.join(target_samples,subj)):
                    for file_tgt in files_tgt:
                        if file_tgt.lower().endswith((".png", ".jpg", ".jpeg")):
                            path_target_file = os.path.join(root_tgt, file_tgt)
                            # print('path_target_file:', path_target_file)

                            output_subj_path = os.path.join(output_path, subj)
                            os.makedirs(output_subj_path, exist_ok=True)
                            
                            target_attribs = {"race": {}}
                            target_attribs["race"]["race"]          = source_attribs["race"]["race"]
                            target_attribs["race"]["dominant_race"] = source_attribs["race"]["dominant_race"]
                            
                            output_file_name = os.path.basename(path_target_file).split('.')[0] + '.json'
                            output_file_path = os.path.join(output_subj_path, output_file_name)
                            print('output_file_path:', output_file_path)
                            save_attributes_json(target_attribs, output_file_path)

                            output_file_name_bin = os.path.basename(path_target_file).split('.')[0] + '.pkl'
                            output_file_path_bin = os.path.join(output_subj_path, output_file_name_bin)
                            print('output_file_path_bin:', output_file_path_bin)
                            save_dict(target_attribs, output_file_path_bin)

                print('--------------')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSON files for a face dataset.")
    parser.add_argument("--source-subj-styles", default='/datasets2/bjgbiesseck/face_recognition/synthetic/dcface_with_pretrained_models/dcface_original_synthetic_ids/dcface_original_10000_synthetic_ids_FACE_ATTRIB', required=True, help="")
    parser.add_argument("--target-samples",     default='/datasets3/bjgbiesseck/face_recognition/dcface/generated_images/tcdiff_WITH_BFM_e:10_spatial_dim:5_bias:0.0_casia_ir50_09-10_1_EQUALIZED-STYLES_BY-RACE_NCLUSTERS=100_ALLOW-STYLE-REPEAT_WHOLE', required=False, help="")
    parser.add_argument("--output-path",        default='', required=False, help="")

    args = parser.parse_args()

    if args.output_path == '':
        args.output_path = args.target_samples + '_FACE_ATTRIB'

    copy_face_attribs(args.source_subj_styles, args.target_samples, args.output_path)

    print('\nFinished!\n\n')
