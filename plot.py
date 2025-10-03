import json
from matplotlib import pyplot as plt
import numpy as np
import torch
from streamtracer import VectorGrid, StreamTracer
from model.model import Model
from mpl_toolkits.mplot3d import Axes3D

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
    
    def __get_tracer_lines(self, B_ext, Bz0, mask_threshold=500, is_3D=False):
        active_regions_mask = np.abs(Bz0) > mask_threshold

        row_indices, col_indices = np.where(active_regions_mask)
        z_indices = np.zeros_like(row_indices)

        stride = 55 if is_3D else 20

        seed_points = np.stack([row_indices, col_indices, z_indices]).T
        seed_points = seed_points[::stride]
        grid_spacing = [1, 1, 1]

        vector_field_for_tracer = B_ext.transpose(1, 2, 3, 0)
        vector_grid = VectorGrid(vector_field_for_tracer.astype(np.float64), grid_spacing)

        max_steps = 20000
        step_size = 0.1
        tracer = StreamTracer(max_steps, step_size)
        tracer.trace(seed_points, vector_grid)
        traced_field_lines = tracer.xs

        return traced_field_lines

    def __plot(self, hmi_map, aia_map, x, y, dx, dy, traced_field_lines, save_fig=False, fig_path="test.jpg"):

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
            x_coords = field_line[:, 0] * dx
            y_coords = field_line[:, 1] * dy
            ax.plot(x_coords, y_coords, color='black', linewidth=1.0)
        ax.set_title("FNOs field lines")

        ax = axes[1][1]
        ax.pcolormesh(x, y, aia_map.data, 
                    norm = aia_map.plot_settings['norm'],
                    cmap = aia_map.plot_settings['cmap'],)
        ax.set_aspect('equal')
        for field_line in traced_field_lines:
            x_coords = field_line[:, 0] * dx
            y_coords = field_line[:, 1] * dy
            ax.plot(x_coords, y_coords, color='black', linewidth=1.0)
        ax.set_title("Aia field lines")

        plt.show() if not save_fig else plt.savefig(fig_path, bbox_inches='tight', dpi=1000)

    def __plot_3D(self, hmi_map, x, y, dx, dy, dz, traced_field_lines, save_fig=False, fig_path="test.jpg"):
        fig = plt.figure(figsize=(20, 8))
        ax = fig.add_subplot(projection='3d', computed_zorder=False)

        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        norm = hmi_map.plot_settings['norm']
        cmap = plt.get_cmap(hmi_map.plot_settings['cmap'])
        facecolors = cmap(norm(hmi_map.data))
        ax.plot_surface(X, Y, Z,
                       facecolors=facecolors,
                       rstride=3, cstride=3,
                       linewidth=0, antialiased=False)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        X_max = Y_max = Z_max = 0
        for field_line in traced_field_lines:
            x_coords = field_line[:, 0] * dx
            y_coords = field_line[:, 1] * dy
            z_coords = field_line[:, 2] * dz

            X_max = max(X_max, x_coords.max())
            Y_max = max(Y_max, y_coords.max())
            Z_max = max(Z_max, z_coords.max())
            ax.plot3D(x_coords, y_coords, z_coords, color='red', linewidth=0.4)

        ax.set_box_aspect([X_max, Y_max, Z_max])  # Aspect ratio is 1:1:1

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        plt.show() if not save_fig else plt.savefig(fig_path, bbox_inches='tight', dpi=1000)


    def __call__(self, datetime, harp_num, mask_threshold, is_3D=False, save_fig=False, fig_path="test.jpg"):
        harp_list = self.sdo_downloader.list_harps(datetime)

        print(harp_list)

        hmi_map, hmi_header, hmi_data = self.sdo_downloader.get_hmi_map(datetime, harp_num)
        input, x, y, dx, dy = self.sdo_downloader.get_model_input(hmi_header, hmi_data)
        B_ext = self.__get_b_extrapolated(torch.tensor(input, dtype=torch.float32))
        
        Bz0 = B_ext[2, :, :, 0]

        field_lines = self.__get_tracer_lines(B_ext, Bz0, mask_threshold=mask_threshold, is_3D=is_3D)

        aia_map = self.sdo_downloader.get_aia_map(datetime, wavelength=171)
        aia_map_cropped = aia_map.reproject_to(hmi_map.wcs)

        self.__plot(hmi_map, aia_map_cropped, x, y, dx, dy, field_lines, save_fig=save_fig, fig_path=fig_path) if not is_3D else None
        self.__plot_3D(hmi_map, x, y, dx, dy, dz=dy, traced_field_lines=field_lines, save_fig=save_fig, fig_path=fig_path) if is_3D else None


if __name__ == '__main__':
    jsoc_email = "martinnedev07@gmail.com"

    plotter = Plotter(jsoc_email)
    plotter(datetime="2024.01.15_00:00:00", harp_num=10634, mask_threshold=400, is_3D=True, save_fig=True, fig_path="2024.01.15_10634_3D.jpg") 
    #plotter(datetime="2024.04.15_00:00:00", harp_num=11054, mask_threshold=450) 
    #plotter(datetime="2024.07.19_00:00:00", harp_num=11560, mask_threshold=400) 
    #plotter(datetime="2024.08.14_00:00:00", harp_num=11689, mask_threshold=500) 
    #plotter(datetime="2022.02.17_00:00:00", harp_num=8013, mask_threshold=700) 




