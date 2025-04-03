import pandas as pd
from sdv.datasets.local import load_csvs
from sdv.metadata import Metadata
from sdv.multi_table import HMASynthesizer
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer
from sdv.evaluation.single_table import evaluate_quality
# from sdv.tabular import CTGAN  # or GaussianCopula, CopulaGAN, TVAE

def read_csv_data(folder: str) -> dict[pd.DataFrame]:
    data = load_csvs(folder_name=folder)
    print(data.keys())
    count = len(data)
    # print(f"Number of tables: {count}")
    return data, count

def autodetect_metadata(data: pd.DataFrame) -> dict:
    metadata = Metadata.detect_from_dataframes(data)
    return metadata

def select_model(metadata:dict, model_type: str='') -> callable:
    if model_type == 'GaussianCopula':
        return GaussianCopulaSynthesizer(
            metadata,
            enforce_rounding=False,
            epochs=500,
            verbose=True,
        )
    elif model_type == 'CTGAN':
        return CTGANSynthesizer
    elif model_type == 'CopulaGAN':
        return CopulaGANSynthesizer
    elif model_type == 'TVAE':
        return TVAESynthesizer
    else:
        raise ValueError("Invalid model type. Choose from 'CTGAN', 'GaussianCopula', 'CopulaGAN', or 'TVAE'.")

def create_synthetic_data(metadata, data, model_type='GaussianCopulaSynthesizer', num_rows=500):
    # Choose the model
    model_class = {
        'CTGAN': CTGANSynthesizer(metadata),
        'GaussianCopula': GaussianCopulaSynthesizer(metadata),
        'CopulaGAN': CopulaGANSynthesizer(metadata),
        'TVAE': TVAESynthesizer(metadata),
    }.get(model_type, GaussianCopulaSynthesizer(metadata))

    model = model_class()
    model.fit(data)
    synthetic_data = model.sample(num_rows)
    # synthesizer = GaussianCopulaSynthesizer(metadata)
    # synthesizer.fit(data=data)
    # synthetic_data = synthesizer.sample(num_rows)
    return synthetic_data

def evaluate_synthetic_data(synthetic_data, real_data, metadata, single_table_name):
    quality_report = evaluate_quality(
        real_data[single_table_name],
        synthetic_data,
        metadata)
    return quality_report