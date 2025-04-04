# %%
import time

# import json
import argparse

from sdv.datasets.demo import download_demo

from utilities import *

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program will generate synthetic data for csv files using SDV."
    )
    parser.add_argument(
        "--folder_path",
        required=False,
        type=str,
        help="Directory to a folder with one csv file. If none is provided, a demo dataset will be used.",
    )
    parser.add_argument(
        "--path_to_metadata",
        required=False,
        type=str,
        help="Path to metadata file. If none is provided, the metadata will be autodetected.",
    )
    parser.add_argument(
        "--num_rows",
        required=False,
        type=int,
        default=500,
        help="Number of rows to generate.",
    )
    parser.add_argument(
        "--save_path",
        required=False,
        type=str,
        help='Path to save the generated synthetic data (example="synthetic_data_sdv.csv"). If none is provided, the data will not be saved.',
    )
    parser.add_argument(
        "--constraint_path",
        required=False,
        type=str,
        help='Path to custom constraints (example="custom_constraints.json"). If none is provided, constraints will not be applied.',
    )

    args = parser.parse_args()

    if args.folder_path:
        data, count = read_csv_data(args.folder_path)
        metadata = autodetect_metadata(data)
        if count > 1:
            raise Exception("Please upload only one table.")
    else:
        data, metadata = download_demo(
            modality="single_table", dataset_name="fake_hotel_guests"
        )
    # %%
    if args.path_to_metadata:
        metadata = Metadata.load_from_json(args.path_to_metadata)
    else:
        metadata = autodetect_metadata(data)

    # %%

    for table_name in data.keys():

        overall_score = 0
        model_results = []

        for model in [
            "CTGAN",
            "CopulaGAN",
            "GaussianCopula",
            "TVAE",
        ]:
            job_start_time = time.time()
            print(f"Running {model} on {table_name}...")

            synthetic_data = create_synthetic_data(
                metadata, data, table_name, model, args.num_rows, args.constraint_path
            )
            print(synthetic_data.head())

            quality_report, shape_score, trend_score = evaluate_synthetic_data(
                synthetic_data, data, metadata, table_name
            )

            avg_score = (shape_score + trend_score) / 2

            model_results.append(
                {
                    "Model": model,
                    "Column Shapes Score": shape_score,
                    "Column Pair Trends Score": trend_score,
                    "Overall Score": avg_score,
                }
            )

            if avg_score > overall_score:
                overall_score = avg_score
                # save csv
                if args.save_path:
                    synthetic_data.to_csv(args.save_path, index=False)
                    print(f"Saving or overwriting synthetic data to {args.save_path}.")

            job_end_time = time.time()
            print(
                f"Program completed. Took {job_end_time - job_start_time:.3f} seconds."
            )

        # save model quality results to csv
        # print(model_results)
        pd.DataFrame(model_results).to_csv(
            f"model_results_{table_name}.csv", index=False
        )

    # %%
