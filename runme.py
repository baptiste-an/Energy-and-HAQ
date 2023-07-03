from health3 import *
from general_functions import *
import zipfile
import scipy
import scipy.io

# # download EXIOBASE and save it in Data/EXIO3, convert heaviest files to .feather
# if not os.path.exists("Data"):
#     os.mkdir("Data")
# if not os.path.exists("Data/EXIO3"):
#     os.mkdir("Data/EXIO3")
# for year in range(1995, 2016, 1):
#     download("https://zenodo.org/record/5589597/files/IOT_" + str(year) + "_pxp.zip", dest_folder="Data/EXIO3")
#     with zipfile.ZipFile("Data/EXIO3/IOT_" + str(year) + "_pxp.zip", "r") as zip_ref:
#         zip_ref.extractall("Data/EXIO3")
#     os.remove("Data/EXIO3/IOT_" + str(year) + "_pxp.zip")

#     path = "Data/EXIO3/IOT_" + str(year) + "_pxp/"
#     feather.write_feather(pd.read_csv(path + "Z.txt", sep="\t", index_col=[0, 1], header=[0, 1]), path + "Z.feather")
#     feather.write_feather(pd.read_csv(path + "Y.txt", sep="\t", index_col=[0, 1], header=[0, 1]), path + "Y.feather")

#     for file in ["Y.txt", "Z.txt", "A.txt"]:
#         os.remove(path + file)

# # download Kbar data and save it in Data/Kbar, convert files to .feather
# if not os.path.exists("Data/Kbar"):
#     os.mkdir("Data/Kbar")
# Z = feather.read_feather("Data/EXIO3/IOT_1995_pxp/Z.feather").fillna(0)
# for year in range(1995, 2016, 1):
#     download(
#         "https://zenodo.org/record/3874309/files/Kbar_exio_v3_6_" + str(year) + "pxp.mat",
#         dest_folder="Data/Kbar",
#     )
#     mat = scipy.io.loadmat("Data/Kbar/Kbar_exio_v3_6_" + str(year) + "pxp")
#     feather.write_feather(
#         pd.DataFrame(mat["KbarCfc"].toarray(), index=Z.index, columns=Z.columns).fillna(0),
#         "Data/Kbar/Kbar_" + str(year) + ".feather",
#     )

# run following functions from ///:
L_and_Lk()
LkYhealth()
SLkYhealth()
imports_dependency()
export_results_with_units()
world_total()
