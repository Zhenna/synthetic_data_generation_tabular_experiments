from typing import Literal
import json
import pandas as pd
from sdv.datasets.local import load_csvs
from sdv.metadata import Metadata
from sdv.single_table import (
    GaussianCopulaSynthesizer,
    CTGANSynthesizer,
    TVAESynthesizer,
    CopulaGANSynthesizer,
)
from sdv.evaluation.single_table import evaluate_quality


def read_csv_data(folder: str) -> dict[pd.DataFrame]:
    data = load_csvs(folder_name=folder)
    # print(data.keys())
    count = len(data)
    # print(f"Number of tables: {count}")
    return data, count


def autodetect_metadata(data: pd.DataFrame) -> dict:
    metadata = Metadata.detect_from_dataframes(data)
    return metadata


def select_model(
    metadata: dict,
    model_type: str = Literal["GaussianCopula", "CTGAN", "CopulaGAN", "TVAE"],
) -> callable:

    if model_type == "GaussianCopula":
        return GaussianCopulaSynthesizer(
            metadata, enforce_min_max_values=True, enforce_rounding=True
        )
    elif model_type == "CTGAN":
        return CTGANSynthesizer(
            metadata, enforce_rounding=True, epochs=500, verbose=True
        )
    elif model_type == "CopulaGAN":
        return CopulaGANSynthesizer(
            metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
            epochs=500,
            verbose=True,
        )
    elif model_type == "TVAE":
        return TVAESynthesizer(
            metadata, enforce_min_max_values=True, enforce_rounding=True, epochs=500
        )
    else:
        raise ValueError(
            "Invalid model type. Choose from 'CTGAN', 'GaussianCopula', 'CopulaGAN', or 'TVAE'."
        )


def create_synthetic_data(
    metadata: dict,
    data: dict,
    single_table_name: str,
    model_type: str = Literal["GaussianCopula", "CTGAN", "CopulaGAN", "TVAE"],
    num_rows: int = 500,
    constraints_path: str = None,
):
    synthesizer = select_model(metadata, model_type)
    if constraints_path:
        print(f"Loading constraints from {constraints_path}")
        with open(constraints_path) as f:
            constraints = json.load(f)
        constraints_list = [
            constraints[constraint] for constraint in list(constraints.keys())
        ]
        synthesizer.add_constraints(
            constraints=constraints_list,
        )
    synthesizer.fit(data=data[single_table_name])
    synthetic_data = synthesizer.sample(num_rows)
    return synthetic_data


def evaluate_synthetic_data(
    synthetic_data: pd.DataFrame,
    real_data: dict,
    metadata: dict,
    single_table_name: str,
) -> tuple:
    quality_report = evaluate_quality(
        real_data[single_table_name], synthetic_data, metadata
    )
    return (
        quality_report,
        quality_report.get_details(property_name="Column Shapes")["Score"].mean(),
        quality_report.get_details(property_name="Column Pair Trends")["Score"].mean(),
    )
