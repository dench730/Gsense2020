import ccdproc as ccdp

from ccdproc import ImageFileCollection
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from astropy import units as u
from astropy import constants as c
from astropy.stats import mad_std
from astropy.io import fits
from astropy.visualization import hist


import glob, os
import matplotlib.patches as patches
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
import sep

import warnings
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter("ignore", category=AstropyWarning)

from astropy.nddata import CCDData
import pandas as pd


data_workdir = "/Volumes/Shu-TF_M4/ustc_cmos/20251109Gsense2020"
output_workdir = "/Users/jomic/Downloads/ccd/ustc_cmos"


dark_frames = ImageFileCollection(f"{output_workdir}/dark", glob_include=f"*dark*.gz")
dark_frames.sort(["exptime", "jd"])

exp_list = []
adu_list = []
err_list = []
for exp in np.unique(dark_frames.summary["exptime"]):
    dark_img = fits.getdata(
        dark_frames.files_filtered(exptime=exp, include_path=True)[0]
    )
    adu_list.append(np.mean(dark_img))
    err_list.append(np.std(dark_img) / np.sqrt(dark_img.size))
    exp_list.append(exp)

plt.errorbar(exp_list, adu_list, yerr=err_list, fmt="o")


os.chdir("/Volumes/Shu-TF_M4/ustc_cmos/cmos_flat")
dark_frames = ImageFileCollection(f"./output", glob_include=f"*dark*.gz")
dark_frames.sort(["exptime", "jd"])

exp_list = []
adu_list = []
err_list = []
for exp in np.unique(dark_frames.summary["exptime"]):
    dark_img = fits.getdata(
        dark_frames.files_filtered(exptime=exp, include_path=True)[0]
    )
    adu_list.append(np.mean(dark_img))
    err_list.append(np.std(dark_img) / np.sqrt(dark_img.size))
    exp_list.append(exp)

plt.errorbar(exp_list, adu_list, yerr=err_list, fmt="x")


plt.xlabel("Exposure Time (s)")
plt.ylabel("Dark Current (ADU)")
plt.show()
