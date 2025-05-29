python scripts/01_preprocess_sar.py --cfg cfg/preprocess_sar/preprocess_sar_12_files.yaml --selected cfg/preprocess_sar/stratified_12_files_unseen.json 

python scripts/02_build_coco.py --cfg cfg/preprocess_sar/preprocess_sar_12_files.yaml --selected cfg/preprocess_sar/stratified_12_files_unseen.json --excel data/WESTMEDEddies_Gade_engl_orig.xlsx --coco_json cfg/preprocess_sar/12_unseen_files_annotations.json

python scripts/03_preview_boxes.py --coco cfg/preprocess_sar/12_unseen_files_annotations.json --outdir data/visualization/12_files_unseen