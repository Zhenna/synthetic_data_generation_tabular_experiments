# synthetic_data_generation_tabular_experiments

If you are using SDV for the first time, please refer to the notebook `sdv.ipynb`.

To run the program, please run command below to install dependencies:

'''bash
pip install -r requirements.txt
'''

## 1. Test the program with sdv demo data

Simply run

'''bash
python main.py
'''

## 2. Generate synthetic data for an existing csv file in `real_data`

'''bash
python main.py --folder_path real_data
'''

It is highly recommended to upload a custom metadata and constraints files. Run the command below to generate higher quality synthetic data and save the output as csv file.

'''bash
python main.py --folder_path real_data \
    --path_to_metadata metadata_fake_hotel_guests.json \
    --save_path synthetic_data_sdv.csv \
    --constraint_path custom_constraints.json
'''


