# The contents of this file are largely copied from analysis.ipynb in https://github.com/grabermtw/Classifying-Satellite-Imagery-with-GNNs

import cv2
import csv
import numpy as np
import datetime
import time
import os
from skimage import io
from PIL import Image as plimg
from segment_image import segment

layers = ["VIIRS_SNPP_CorrectedReflectance_TrueColor", "VIIRS_SNPP_Clear_Sky_Confidence_Day"]
startdate = datetime.date(2022,5,1)
enddate = datetime.date(2022,5,6)
img_extent_step = 5
resolution = 128

for layer in layers:
    print("Downloading {} images...".format(layer))
    layer_outdir = os.path.join(os.getcwd(), "images", layer)
    currentdate = startdate

    while currentdate < enddate:
        outdir = os.path.join(layer_outdir, str(currentdate))
        
         # Create directory if it doesn't exist yet
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
            
        print("Downloading images for {}...".format(currentdate))
        for longitude in range(-180,180,img_extent_step):
            for latitude in range(-90,90,img_extent_step):
                extents = "{0},{1},{2},{3}".format(latitude, longitude,
                                                latitude + img_extent_step,
                                                longitude + img_extent_step)
                outfilepath = os.path.join(outdir,'{0}_{1}_{2}.png'.format(layer, currentdate, extents))
                # Skip any files that have already been downloaded
                # (this enables quick resumption if connection errors are encountered).
                # put this in a while-loop in case there's a connection error and
                # the download for something needs to be retried
                while not os.path.exists(outfilepath) or cv2.imread(outfilepath) is None:
                    # Construct image URL.
                    url = 'https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?\
version=1.3.0&service=WMS&request=GetMap&\
format=image/png&STYLE=default&bbox={0}&CRS=EPSG:4326&\
HEIGHT={3}&WIDTH={3}&TIME={1}&layers={2}'.format(extents, currentdate, layer, resolution)
                    
                    # Occasionally we get an error from a momentary dropout of internet connection or something.
                    # This try-except should 
                    try:
                        # Request and save image
                        img = plimg.fromarray(io.imread(url))
                        img.save(outfilepath)
                    except:
                        print("Error encountered, retrying")
                        time.sleep(5)

        currentdate += datetime.timedelta(1)

# OSM_Land_Water_Map is a static layer, meaning that we don't need to re-download it for every day.
layer = "OSM_Land_Water_Map"
print("Downloading {} images...".format(layer))
outdir = os.path.join(os.getcwd(), "images", "{}".format(layer))

# Create directory if it doesn't exist yet
if not os.path.exists(outdir):
    os.mkdir(outdir)

for longitude in range(-180,180,img_extent_step):
    for latitude in range(-90,90,img_extent_step):
        extents = "{0},{1},{2},{3}".format(latitude, longitude,
                                        latitude + img_extent_step,
                                        longitude + img_extent_step)
        outfilepath = os.path.join(outdir,'{0}_{1}.png'.format(layer, extents))
        # Skip any files that have already been downloaded
        # (this enables quick resumption if connection errors are encountered)
        while not os.path.exists(outfilepath) or cv2.imread(outfilepath) is None:
            # Construct image URL.
            url = 'https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?\
version=1.3.0&service=WMS&request=GetMap&\
format=image/png&STYLE=default&bbox={0}&CRS=EPSG:4326&\
HEIGHT={2}&WIDTH={2}&layers={1}'.format(extents, layer, resolution)
            # Occasionally we get an error from a momentary dropout of internet connection or something.
            # This try-except should 
            try:
                # Request and save image
                img = plimg.fromarray(io.imread(url))
                img.save(outfilepath)
            except:
                print("Error encountered, retrying")
                time.sleep(5)


labeled_data_filename = "labeled_data.csv"

layer_to_label_path = os.path.join("images","VIIRS_SNPP_CorrectedReflectance_TrueColor")
clear_sky_layer_path = os.path.join("images", "VIIRS_SNPP_Clear_Sky_Confidence_Day")
land_water_map_path = os.path.join("images", "OSM_Land_Water_Map")
lw_filelist = os.listdir(land_water_map_path)

resolution = 128
pixel_count = resolution ** 2
# Exclude any images where 40% or more of the image is "no data"
nodata_threshold = pixel_count * 0.6

# dict for memoization of land water map results, since this is a static layer
lw_results = {}

# for segmentation
sigma = 0.1
k = 500
min = 50
segmented_outdir = os.path.join(os.getcwd(), "images", "segmented")
# Create directory if it doesn't exist yet
if not os.path.exists(segmented_outdir):
    os.mkdir(segmented_outdir)


with open(labeled_data_filename, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "segmented_filepath", "weather", "terrain"])
    for date in os.listdir(layer_to_label_path):
        print("Labeling for date {}...".format(date))
        co_re_datepath = os.path.join(layer_to_label_path, date)
        cl_sk_datepath = os.path.join(clear_sky_layer_path, date)
        co_re_filelist = os.listdir(co_re_datepath)
        cl_sk_filelist = os.listdir(cl_sk_datepath)
        for i in range(len(co_re_filelist)):
            # these directories should be ordered the same
            co_re_imgpath = os.path.join(co_re_datepath, co_re_filelist[i])
            cl_sk_imgpath = os.path.join(cl_sk_datepath, cl_sk_filelist[i])
            
            csv_row = [co_re_imgpath]

            # First, check if the corrected reflectance image is in an area of "no data"
            # i.e. it's all or mostly pure black.
            # We want to skip these images.
            co_re_img = cv2.imread(co_re_imgpath, 0) # use 0 flag to read grayscale
            if cv2.countNonZero(co_re_img) < nodata_threshold:
                continue
            
            # Create segmented version of the image
            # save image with pillow
            output = segment(co_re_imgpath, sigma, k, min)
            im = plimg.fromarray(np.uint8(output))
            segmented_filepath = os.path.join("images", "segmented", "segmented" + co_re_filelist[i])
            im.save(segmented_filepath)
            csv_row.append(segmented_filepath)

            # Next, check if the image is mostly cloudy or not cloudy.
            # In this layer, the reddish color (the higher pixel value) corresponds to clear skies,
            # and the whiteish color (the lower pixel value) corresponds to cloudy skies.
            cl_sk_img = cv2.imread(cl_sk_imgpath, 0)
            
            # If there are more dark-colored pixels than light-colored pixels, then it's not cloudy.
            if cv2.countNonZero(cv2.inRange(cl_sk_img, 0, 127)) > cv2.countNonZero(cv2.inRange(cl_sk_img, 128, 255)):
                # clear skies
                csv_row.append("clear")
            else:
                # cloudy skies
                csv_row.append("cloudy")

            # Finally, check if the image is mostly land or water.
            lw_imgpath = os.path.join(land_water_map_path, lw_filelist[i])
            if lw_imgpath in lw_results.keys():
                csv_row.append(lw_results[lw_imgpath])
            else:
                lw_img = cv2.imread(lw_imgpath, 0)
                # If there are more light-colored pixels than dark-colored pixels, then its mostly water
                if cv2.countNonZero(cv2.inRange(lw_img, 128, 128)) > cv2.countNonZero(cv2.inRange(lw_img, 75, 75)):
                    lw_results[lw_imgpath] = "water"
                else:
                    lw_results[lw_imgpath] = "land"
                csv_row.append(lw_results[lw_imgpath])
            
            writer.writerow(csv_row)

print("Labeling complete!")
