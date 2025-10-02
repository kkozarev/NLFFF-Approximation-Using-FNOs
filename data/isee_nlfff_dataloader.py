import netCDF4
import os
import numpy as np

class ISEE_NLFFF_Dataloader():
    def __init__(self):
        pass

    def __load_file(self, nc_file):
        nc = netCDF4.Dataset(nc_file, 'r')

        x = nc['x'][:]
        y = nc['y'][:]
        z = nc['z'][:]

        bx = np.transpose(nc['Bx'][:])
        by = np.transpose(nc['By'][:])
        bz = np.transpose(nc['Bz'][:])

        b = np.stack([bx, by, bz])

        return x, y, z, b

    def save_data(self, nc_files_path, save_input_path, save_label_path):
        nc_files_list = os.listdir(nc_files_path)

        os.makedirs(save_input_path, exist_ok=True)
        os.makedirs(save_label_path, exist_ok=True)

        print(save_input_path)
        for file in nc_files_list:
            nc_file = os.path.join(nc_files_path, file)
            x,y,z,b = self.__load_file(nc_file)

            print(file)
            b0 = b[:,:,:,0][:,:,:,None]
            
            save_input_file_path = os.path.join(save_input_path, file)
            save_label_file_path = os.path.join(save_label_path, file)

            np.savez(save_input_file_path, b0=b0)
            np.savez(save_label_file_path, b=b, x=x, y=y,z=z)
