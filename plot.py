import json
from matplotlib import pyplot as plt
import numpy as np
import torch
from streamtracer import VectorGrid, StreamTracer
from model.model import Model
import astropy.units as u

from plot.sdo_downloader import SDO_Downloader

class Plotter:   
    def __init__(self, jsoc_email):
        self.sdo_downloader = SDO_Downloader(jsoc_email)

    def __get_b_extrapolated(self, input):
        with open("config.json") as config_data:
            config = json.load(config_data)

        model = Model(config)
        model.load_model(config['model']['model_path'])

        B_ext = model(input.to(model.device))

        return B_ext.cpu().detach().numpy()
    
    def __get_tracer_lines(self, B_ext, Bz0, mask_threshold=500):
        active_regions_mask = np.abs(Bz0) > mask_threshold

        row_indices, col_indices = np.where(active_regions_mask)
        z_indices = np.zeros_like(row_indices)

        seed_points = np.stack([row_indices, col_indices, z_indices]).T
        seed_points = seed_points[::15]
        grid_spacing = [1, 1, 1]

        vector_field_for_tracer = B_ext.transpose(1, 2, 3, 0)
        vector_grid = VectorGrid(vector_field_for_tracer.astype(np.float64), grid_spacing)

        max_steps = 20000
        step_size = 0.1
        tracer = StreamTracer(max_steps, step_size)
        tracer.trace(seed_points, vector_grid)
        traced_field_lines = tracer.xs

        return traced_field_lines

    def __plot(self, hmi_map, aia_map, x, y, dx, dy, traced_field_lines):

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10), sharex=True, sharey=True)

        ax = axes[0][0]
        ax.set_aspect('equal')
        ax.pcolormesh(x, y, hmi_map.data, 
                      norm = hmi_map.plot_settings['norm'],
                      cmap = hmi_map.plot_settings['cmap'])
        ax.set_title("Ground truth")

        ax = axes[0][1]
        ax.set_aspect('equal')
        ax.pcolormesh(x, y, aia_map.data, 
                    norm = aia_map.plot_settings['norm'],
                    cmap = aia_map.plot_settings['cmap'])
        ax.set_title("AIA 171")

        ax = axes[1][0]
        ax.set_aspect('equal')
        ax.pcolormesh(x, y, hmi_map.data, 
                      norm = hmi_map.plot_settings['norm'],
                      cmap = hmi_map.plot_settings['cmap'])
        for field_line in traced_field_lines:
            x_coord = field_line[:, 0] * dx + x[0]
            y_coord = field_line[:, 1] * dy + y[0]
            ax.plot(x_coord, y_coord, color='black', linewidth=1.0)
        ax.set_title("FNOs field lines")

        ax = axes[1][1]
        ax.pcolormesh(x, y, aia_map.data, 
                    norm = aia_map.plot_settings['norm'],
                    cmap = aia_map.plot_settings['cmap'],)
        ax.set_aspect('equal')
        for field_line in traced_field_lines:
            x_coord = field_line[:, 0] * dx + x[0]
            y_coord = field_line[:, 1] * dy + y[0]
            ax.plot(x_coord, y_coord, color='black', linewidth=1.0)
        ax.set_title("Aia field lines")

        plt.show()



    def __call__(self, datetime, harp_num, mask_threshold):
        harp_list = self.sdo_downloader.list_harps(datetime)

        print(harp_list)

        hmi_map, hmi_header, hmi_data = self.sdo_downloader.get_hmi_map(datetime, harp_num)
        input, x, y, dx, dy = self.sdo_downloader.get_model_input(hmi_header, hmi_data)
        B_ext = self.__get_b_extrapolated(torch.tensor(input, dtype=torch.float32))
        
        Bz0 = B_ext[2, :, :, 0]

        field_lines = self.__get_tracer_lines(B_ext, Bz0, mask_threshold=mask_threshold)

        aia_map = self.sdo_downloader.get_aia_map(datetime, wavelength=171)
        aia_map_cropped = aia_map.reproject_to(hmi_map.wcs)

        self.__plot(hmi_map, aia_map_cropped, x, y, dx, dy, field_lines)


if __name__ == '__main__':
    jsoc_email = "martinnedev07@gmail.com"

    plotter = Plotter(jsoc_email)
    plotter(datetime="2024.01.15_00:00:00", harp_num=10634, mask_threshold=400) 
    #plotter(datetime="2024.04.15_00:00:00", harp_num=11054, mask_threshold=450) 
    #plotter(datetime="2024.07.19_00:00:00", harp_num=11560, mask_threshold=400) 
    #plotter(datetime="2024.08.14_00:00:00", harp_num=11689, mask_threshold=500) 
    #plotter(datetime="2022.02.17_00:00:00", harp_num=8013, mask_threshold=700) 




