"""
create on February 2023,

@author:nannanSun

Function:To extract ms1,ms2 from mzML file

"""

import argparse
from distutils.command.config import config
import pdb
from webbrowser import get
import numpy as np
from pyteomics import mzml, auxiliary
import os
import sys
import time
import torch
import h5py
import pandas as pd
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_dir",
        default="./mzml",
        help="Directory that contains only DIA data files. Centroided .mzXML, .mzML or .raw files from Thermo Fisher equipments are supported. (For Linux systems, `mono` tool has to be installed for the supporting of .raw files. https://www.mono-project.com/download/stable/#download-lin)",
    )

    parser.add_argument(
        "--spectrum_resolution", default=10, type=float, help="Spectrum resolution"
    )
    parser.add_argument("--mz_max", default=1800, type=int, help="Maximum mz")
    parser.add_argument("--mz_min", default=260, type=int, help="minimun mz ")
    parser.add_argument("--mass_H", default=1.0078, type=float, help="H mass")
    parser.add_argument("--save_dir", default="./mzml_parsed")
    return parser


class MS1_Chrom:

    def __init__(self):
        self.rt_list = []
        self.spectra = []
        self.indexs = []
        self.scan_win = []


class MS2_Chrom:

    def __init__(self):
        self.mapping_MS1idx = 0
        self.rt_list = []
        self.spectra = []
        self.indexs = []
        self.precursor_ls = []
        self.precursor_Bound = []


mslevel_string = "ms level"


def MS1_infos(rawdata_reader):
    ms1 = MS1_Chrom()
    for idx, spectrum in enumerate(rawdata_reader):
        if spectrum[mslevel_string] == 1:
            RT = spectrum["scanList"]["scan"][0]["scan start time"]
            ms1.rt_list.append(RT)
            intensity_array = spectrum["intensity array"]
            mz_array = spectrum["m/z array"][intensity_array > 0]
            intensity_array = intensity_array[intensity_array > 0]
            ms1.spectra.append((mz_array, intensity_array))
            ms1.indexs.append(spectrum["index"])
            # ms1.spectra.extend(mz_array)
            # ms1.indexs.append(spectrum["index"])
            # ms1.scan_win.append(spectrum["scanList"]["scan"][0]["scanWindowList"]["scanWindow"][0])
    return ms1


def MS2_infos(rawdata_reader, index_win):
    ms2 = MS2_Chrom()
    for idx, spectrum in enumerate(rawdata_reader):
        if spectrum[mslevel_string] == 1:
            if spectrum["index"] == index_win[0]:
                ms2.mapping_MS1idx = index_win[0]
            else:
                ms2.mapping_MS1idx = 1000001
        elif spectrum[mslevel_string] == 2:
            if spectrum["index"] > index_win[0] and spectrum["index"] < index_win[1]:
                ms2.rt_list.append(
                    60 * spectrum["scanList"]["scan"][0]["scan start time"]
                )
                ms2.indexs.append(spectrum["index"])
                precursor_mz = spectrum["precursorList"]["precursor"][0][
                    "selectedIonList"
                ]["selectedIon"][0]["selected ion m/z"]
                precursor_intensity = spectrum["precursorList"]["precursor"][0][
                    "selectedIonList"
                ]["selectedIon"][0]["peak intensity"]
                precursor_leftBound = spectrum["precursorList"]["precursor"][0][
                    "isolationWindow"
                ]["isolation window lower offset"]
                precursor_rightBound = spectrum["precursorList"]["precursor"][0][
                    "isolationWindow"
                ]["isolation window upper offset"]
                ms2.precursor_ls.append((precursor_mz, precursor_intensity))
                ms2.precursor_Bound.append((precursor_leftBound, precursor_rightBound))
                # import pdb;pdb.set_trace()
                intensity_array = spectrum["intensity array"]
                intensity_array = intensity_array[intensity_array > 0]
                mz_array = spectrum["m/z array"][intensity_array > 0]
                ms2.spectra.append((mz_array, intensity_array))
            else:
                continue
    return ms2


def getData(rawdata_file):
    if rawdata_file.endswith(".mzML"):
        rawdata_reader = mzml.MzML(rawdata_file)
    else:
        pass
    # if not rawdata_reader:
    #     pass
    ms1 = MS1_infos(rawdata_reader)
    # print(ms1.indexs[:10])
    index_win_ls = []
    for idx, index in enumerate(ms1.indexs):
        if idx + 1 == len(ms1.indexs):
            index_win_ls.append([index, index + 100])
        else:
            index_win_ls.append([index, ms1.indexs[idx + 1]])
    # print(index_win_ls)
    ms2_ls = []
    # for idx, index_win in enumerate(index_win_ls):
    #     if idx >0:
    #         continue
    #     #print(rawdata_reader[index_win[0]:index_win[1]])
    #     ms2 = MS2_infos(rawdata_reader[index_win[0]:index_win[1]],index_win)
    #     ms2_ls.append(ms2)
    # import pdb;pdb.set_trace()

    return ms1, ms2_ls, index_win_ls


def z_score(intensity_ls):
    # intensity_ls_log = np.log10(np.array(intensity_ls))
    # z_ls = (np.array(intensity_ls_log) - np.mean(intensity_ls_log)) / np.std(intensity_ls_log)
    z_ls = np.array(intensity_ls) / np.max(intensity_ls)
    return z_ls


def process_spectrum(spectrum_mz_ls, spectrum_intensity_ls, config):

    MZ_SIZE = int((config.mz_max - config.mz_min) * config.spectrum_resolution)
    # neutral mass, location, assuming ion charge z=1
    charge = 1.0
    spectrum_mz_cutted_ls = [
        mz for mz in spectrum_mz_ls if mz >= config.mz_min and mz <= config.mz_max
    ]
    spectrum_mz = np.array(spectrum_mz_cutted_ls, dtype=np.float32)
    neutral_mass = spectrum_mz - charge * config.mass_H
    neutral_mass_location = np.rint(
        (neutral_mass - config.mz_min) * config.spectrum_resolution
    ).astype(np.int32)
    # normalize intensity
    spectrum_intensity_cutted_ls = [
        spectrum_intensity_ls[idx]
        for idx, mz in enumerate(spectrum_mz_ls)
        if mz >= config.mz_min and mz <= config.mz_max
    ]
    spectrum_intensity = np.array(spectrum_intensity_cutted_ls, dtype=np.float32)
    norm_intensity = spectrum_intensity / np.max(spectrum_intensity)
    norm_intensity = z_score(spectrum_intensity)
    # fill spectrum holders
    spectrum_holder = np.zeros(shape=MZ_SIZE, dtype=np.float32)
    for index in range(neutral_mass_location.size):
        # spectrum_holder[neutral_mass_location[index]] += spectrum_intensity[index]
        if spectrum_holder[neutral_mass_location[index]] == 0:
            spectrum_holder[neutral_mass_location[index]] = norm_intensity[index]
        else:
            spectrum_holder[neutral_mass_location[index]] = max(
                spectrum_holder[neutral_mass_location[index]], norm_intensity[index]
            )

    return spectrum_holder


def spectrum_bin(ms1, config):

    ms1_bin = []
    for idx, (spectrum_mz, spectrum_intensity) in enumerate(ms1.spectra):
        if idx >= 0:
            spectrum_holder = process_spectrum(spectrum_mz, spectrum_intensity, config)
            ms1_bin.append(spectrum_holder)

    ms1_bin_array = torch.tensor(np.array(ms1_bin), dtype=torch.float64)

    # print(ms1_bin_array)
    return ms1_bin_array


def main_function(config):
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    infos_dict = {"file_name": [], "label": []}
    count = 0
    for file in tqdm(os.listdir(config.file_dir)):
        if file.endswith(".mzML"):
            if os.path.exists(
                os.path.join(config.save_dir, file.split(".")[0] + ".pt")
            ):
                infos_dict["file_name"].append(file.strip(".mzML"))
                infos_dict["label"].append(int(1))
                print("it had been converted")
                continue
            count += 1
            # start = time.time()
            file_path = os.path.join(config.file_dir, file)
            ms1, ms2_ls, index_win_ls = getData(file_path)
            # import pdb;pdb.set_trace()
            ms1_bin_array = spectrum_bin(ms1, config)
            # print(count,"The ms1_bin shape:{}".format(str(ms1_bin_array.shape)))
            saved_path = os.path.join(config.save_dir, file.strip(".mzML")) + ".pt"
            # np.save(saved_path,ms1_bin_array)
            # print(sys.getsizeof(ms1_bin_array))
            infos_dict["file_name"].append(file.strip(".mzML"))
            infos_dict["label"].append(int(1))
            # tensor_infos = torch.tensor(ms1_bin_array.numpy())
            tensor_infos = ms1_bin_array.to_sparse()
            torch.save(tensor_infos, saved_path)
            # with h5py.File(saved_path,"w")as hf:
            #     hf.create_dataset("data",data=ms1_bin_array.numpy())
            # end = time.time()
            # s = end-start
            # print("Time has been consumed:{}".format(s))
    infos = pd.DataFrame(infos_dict)
    infos.to_excel(f"{config.save_dir}/resolution_10.xlsx")


if __name__ == "__main__":
    parser = get_parser()
    config = parser.parse_known_args()[0]
    main_function(config)
    # ms1,ms2_ls,index_win_ls = getData(config.file_path)
    # spectrum_bin(ms1,config)
