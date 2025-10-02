import json

from data.isee_nlfff_dataloader import ISEE_NLFFF_Dataloader


if __name__ == "__main__":
    with open("config.json") as config_data:
        config = json.load(config_data)
        
    loader = ISEE_NLFFF_Dataloader()
    loader.save_data(
        nc_files_path=config["nc_files_extractor"]["nc_files_path"],
        save_input_path=config["nc_files_extractor"]["save_input_path"],
        save_label_path=config["nc_files_extractor"]["save_label_path"]
    )
