import json
import torch
from data.test_file_loader import TestFileLoader
from model.evaluation import evaluate_magnetic_fields
from model.model import Model

def get_average_metrics(test_file_loader, model):
    total_metrics = {
        'C_vec': 0.0,
        'C_CS': 0.0,
        "E'_n": 0.0,
        "E'_m": 0.0,
        'epsilon': 0.0,
        'L_div_n': 0.0,
        'sigma_J': 0.0
    }

    num_samples = test_file_loader.lenght
    for file_idx in range(num_samples):
        noaa_ar, B_ref, x, y = test_file_loader.get_label(file_idx)
        input = test_file_loader.get_input(file_idx)
        B_ext = model(input.to(model.device))

        metrics = evaluate_magnetic_fields(B_ref, B_ext)
        print(f"File index: {file_idx}")
        for key in total_metrics:
            total_metrics[key] += metrics[key]

    average_metrics = {key: value / num_samples for key, value in total_metrics.items()}
    return average_metrics

def print_metrics(metrics):
    """
    Prints the evaluation metrics in a formatted manner.

    Args:
        metrics (dict): A dictionary containing the evaluation metrics.
    """
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("config.json") as config_data:
        config = json.load(config_data)

    model = Model(config)
    model.load_model(config['model']['model_path'])

    test_file_loader = TestFileLoader(config)

    metrics_dict = get_average_metrics(test_file_loader=test_file_loader, model=model)
    print_metrics(metrics_dict)
    
    with open("metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)