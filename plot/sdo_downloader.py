import drms
import numpy as np
from sunpy.map import Map
from astropy.io import fits
from astropy.time import Time
from skimage.transform import resize

class SDO_Downloader:
    def __init__(self, jsoc_email):
        self.client = drms.Client()
        if self.client.check_email(jsoc_email) == False:
                raise ValueError("Invalid JSOC email address.")
        self.jsoc_email = jsoc_email

    def list_harps(self, datetime):
        """
        Lists available HARPs for a given datetime.

        Parameters:
        datetime (str): Datetime in the format 'YYYY.MM.DD_HH:MM:SS_TAI'.

        Returns:
        list: List of available HARPs.
        """
        query = f"hmi.Mharp_720s[][{datetime}]"
        
        try:
            results = self.client.query(query, key=['NOAA_AR', 'HARPNUM'])
            harps = results['HARPNUM'].tolist()
            return harps
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return []
        
    def get_hmi_map(self, datetime, harp_num):
        query = f'hmi.sharp_cea_720s[{harp_num}][{datetime}]'
        hmi_keys, hmi_segments = self.client.query(query, key=drms.JsocInfoConstants.all, seg='Bp, Bt, Br, magnetogram')

        hmi_url = 'http://jsoc.stanford.edu' + hmi_segments.magnetogram[0]
        hmi_image_data = fits.getdata(hmi_url)
        hmi_header = dict(hmi_keys.iloc[0])
        hmi_map = Map(hmi_image_data, hmi_header)

        hmi_Bp_url = 'http://jsoc.stanford.edu' + hmi_segments.Bp[0]
        hmi_Bt_url = 'http://jsoc.stanford.edu' + hmi_segments.Bt[0]
        hmi_Br_url = 'http://jsoc.stanford.edu' + hmi_segments.Br[0]

        Bx = fits.getdata(hmi_Bp_url)
        By = -fits.getdata(hmi_Bt_url)
        Bz = fits.getdata(hmi_Br_url)

        hmi_data = np.stack((Bx, By, Bz)).T
        hmi_data = np.nan_to_num(hmi_data, nan=0.0)

        return hmi_map, hmi_header, hmi_data

    def get_aia_map(self, datetime, wavelength=171):        
        query = f'aia.lev1_euv_12s[{datetime}][{wavelength}]'
        aia_keys, aia_segments = self.client.query(query, key=drms.JsocInfoConstants.all, seg='image')

        aia_url = 'http://jsoc.stanford.edu' + aia_segments.image[0]
        aia_data = fits.getdata(aia_url)
        aia_header = dict(aia_keys.iloc[0])
        aia_map = Map(aia_data, aia_header)

        return aia_map

    def get_model_input(self, hmi_header, hmi_data):
        solar_radius = 696 # in Mm
        hmi_number_of_pixels_x, hmi_number_of_pixels_y, hmi_number_of_pixels_z = hmi_data.shape

        #print(hmi_header['CUNIT1'])
        pixel_size_deg_x, pixel_size_deg_y = hmi_header['CDELT1'], hmi_header['CDELT2']  # degrees per pixel

        pixel_size_radians_x, pixel_size_deg_y = np.deg2rad(pixel_size_deg_x), np.deg2rad(pixel_size_deg_y)

        pixel_size_linear_x = solar_radius * pixel_size_radians_x  # in Mm
        pixel_size_linear_y = solar_radius * pixel_size_deg_y  # in Mm

        x = np.linspace(0, (hmi_number_of_pixels_x-1)*pixel_size_linear_x, hmi_number_of_pixels_x)
        y = np.linspace(0, (hmi_number_of_pixels_y-1)*pixel_size_linear_y, hmi_number_of_pixels_y)

        model_input = hmi_data[:, :, np.newaxis, :]

        model_input = model_input.transpose(3, 0, 1, 2)
        return model_input, x, y, pixel_size_linear_x, pixel_size_linear_y