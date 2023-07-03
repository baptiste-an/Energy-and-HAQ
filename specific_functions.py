import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pymrio
import math
from sklearn.metrics import r2_score
from general_functions import *


pathexio = ""  # add path towards your EXIOBASE data if it is not in current directory
# pathexio = "C:/Users/andrieba/Documents/"

### Population

population = rename_region(
    pd.read_csv(
        "Data/worldbank population/API_SP.POP.TOTL_DS2_en_csv_v2_3852487.csv",
        header=[2],
        index_col=[0],
    ),
    level="Country Name",
).drop("Z - Aggregated categories")[[str(i) for i in range(1995, 2020, 1)]]
population.columns = [int(i) for i in population.columns]
population.loc["Taiwan"] = pd.read_excel("Data/pop Taiwan.xls", header=0, index_col=0)["TW"][
    [i for i in range(1995, 2016, 1)]
]
pop = (
    population.rename(
        dict(
            zip(
                cc.name_shortas("EXIO3")["name_short"].values,
                cc.name_shortas("EXIO3")["EXIO3"].values,
            )
        )
    )
    .groupby(level=0)
    .sum()
) / 1000


### Aggregation of the HAQ index

HAQ = (
    (
        rename_region(
            pd.read_csv(
                "Data/HAQ 2015.csv",
                index_col=["indicator_name", "year_id", "location_name", "location_id"],
            )
            .loc["Healthcare Access and Quality"]["val"]
            .unstack(level="year_id"),
            level="location_name",
        )
        .drop("Z - Aggregated categories")
        .reindex(columns=[1990, 1995, 2000, 2005, 2010, 2015])
        .drop([1990], axis=1)
    )
    .groupby(level=0)
    .sum()
)
HAQ_agg = HAQ.drop("Anguilla") * population.loc[HAQ.drop("Anguilla").index]
HAQ_agg["region"] = cc.convert(names=HAQ_agg.index, to="EXIO3")
HAQ_agg = (
    (HAQ_agg.reset_index().set_index("region").drop("location_name", axis=1).groupby(level="region").sum()) / pop / 1000
)

### Health expenditures

XReuros = pd.read_excel("Data/exchange rates.xlsx").loc[0].drop([2016, 2017, 2018])
# exchange rate for euro zone, euros/dollars


def Y_health_NHA_euros():
    Yhealth = pd.DataFrame()  # Exiobase health sector expenditures
    for i in range(1995, 2016, 1):
        pathIOT = pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/"
        Yhealth[i] = (
            feather.read_feather(pathIOT + "Y.feather")
            .groupby(level="region", axis=1)
            .sum()
            .groupby(level="sector")
            .sum()
            .loc["Health and social work services (85)"]
        )
    # we will compare these expenditures with NHA in order to use the 2000 share for years 19995-1999

    expendituresNHA = (
        pd.read_excel("Data/NHA_health_expenses_2018.xlsx", header=[0], index_col=[0]).groupby(level="region").sum()
        / 1000
    ).drop(
        [2016, 2017, 2018], axis=1
    )  # WHO expenditures in NHA format in kUS$
    expendituresOECD_95_99 = pd.read_excel("Data/OECD_health_expenses.xlsx", header=[0], index_col=[0])
    # OECD expenditures for missing years in WHO in US$
    expendituresNHA = pd.concat([expendituresOECD_95_99, expendituresNHA], axis=1)
    # expenditures in NHA format for WHO and OECD data
    expendituresNHAeuros = expendituresNHA * XReuros
    # from current US$ to current euros

    shareTW = Yhealth.loc["TW"] / (Yhealth.loc["CN"] + Yhealth.loc["TW"])
    # Taiwan is not in the GHED so we assume its share of China is that of Exiobase sector 'Health and social work services (85)'
    expendituresNHAeuros49 = expendituresNHAeuros.T.drop(["CN"], axis=1)
    expendituresNHAeuros49["CN"] = (1 - shareTW).values * expendituresNHAeuros.T["CN"]
    expendituresNHAeuros49["TW"] = (shareTW).values * expendituresNHAeuros.T["CN"]
    expendituresNHAeuros49 = expendituresNHAeuros49.T

    shareExioOverNHA = (Yhealth / expendituresNHAeuros49).interpolate(method="linear", axis=1).T.bfill().T
    # We consider that for the period 1995-2000 the ratio of expenditures in the NHA format and of the exiobase sector (85)
    # stays constant for the countries not in the OECD database

    expendituresNHAeuros49extrapolated = Yhealth / shareExioOverNHA
    # the total health expenditures in the NHA format extrapolated for years 1995-1999

    health_shares_OECD = (pd.read_excel("Data/Health_Shares_OECD.xlsx", index_col=[0, 1], header=[0, 1]) / 100).drop(
        [2016, 2017, 2018], axis=1
    )
    health_shares_OECD = pd.DataFrame(
        health_shares_OECD.stack().T.interpolate(method="linear", axis=0).bfill().stack(level=0).stack(level=0),
        columns=expendituresNHAeuros49.index,
    )

    # For every retailer and every year we want to calculate a mean percentage for all the countries where data is available We multiply the shares by the expenditures In order to have NaN values in the expenditures vector we multiply and divide it by itself

    shares_multiplied_by_expenditures = (
        health_shares_OECD.unstack(level=0) * expendituresNHAeuros49extrapolated.stack()
    ).stack()
    expenditures_with_appropriate_NaNValues = (
        (health_shares_OECD.unstack(level=0) * expendituresNHAeuros49extrapolated.stack())
        / health_shares_OECD.unstack(level=0)
    ).stack()
    mean_shares_OECD = (
        shares_multiplied_by_expenditures.sum(axis=1) / expenditures_with_appropriate_NaNValues.sum(axis=1)
    ).reorder_levels([2, 0, 1])
    for i in health_shares_OECD.index:
        for j in health_shares_OECD.columns:
            if math.isnan(health_shares_OECD.loc[i, j]) == True:
                health_shares_OECD.loc[i, j] = mean_shares_OECD.loc[i]
    health_shares_OECD[
        "WM"
    ] = mean_shares_OECD  # only Israel with atypical "Retailers and other providers of medical goods"
    health_shares_OECD = health_shares_OECD.unstack(level=0).groupby(level="sector").sum()
    health_shares_OECD.loc["Health and social work services (85)"] = 1 - health_shares_OECD.sum()

    i = 2015
    pathIOT = pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/"
    sectors = feather.read_feather(pathIOT + "Y.feather").loc["FR"].index

    Y_health_NHA_euros = health_shares_OECD.T.mul(expendituresNHAeuros49extrapolated.stack(), axis=0)
    Y_health_NHA_euros = pd.DataFrame(Y_health_NHA_euros, columns=sectors).swaplevel()

    return Y_health_NHA_euros


Y_health_NHA_euros = Y_health_NHA_euros()


def ppp_health():
    ppp_all_regions = (
        pd.read_excel("Data/NHA indicators - per capita exp ppp.xlsx", index_col=[0], header=[0])[1:]
        .drop(["Indicators", "Unnamed: 2"], axis=1)
        .stack()
        .replace(dict({":": np.nan}))
        .astype("float")
    )
    pop_all_regions = (
        pd.read_excel("Data/NHA indicators - population.xlsx", index_col=[0], header=[0])[1:]
        .drop(["Indicators", "Unnamed: 2"], axis=1)
        .stack()
        .replace(dict({":": np.nan}))
        .astype("float")
    )
    current_ppp = (ppp_all_regions * pop_all_regions).unstack().drop(["2016", "2017", "2018"], axis=1)
    current_ppp["region"] = cc.convert(names=current_ppp.index, to="EXIO3")
    current_ppp = current_ppp.reset_index()
    current_ppp = current_ppp.set_index("region").groupby(level="region").sum()
    current_ppp.columns = [int(i) for i in current_ppp.columns]

    CPI_US = pd.read_excel("Data/CPI US eurostat.xlsx", index_col=0, header=0).drop([2016, 2017], axis=1)
    CPI_US = CPI_US / CPI_US.loc["US"][2015]
    constant_ppp = current_ppp.drop([2000, 2001], axis=1).div(CPI_US.loc["US"], axis=1)

    return constant_ppp


constant_ppp = ppp_health()


###


def L_and_Lk():
    """Calculates Lk and L and saves it to the Exiobase data files.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    for i in range(1995, 2016, 1):
        pathIOT = "Data/EXIO3/IOT_" + str(i) + "_pxp/"
        Y = feather.read_feather(pathIOT + "Y.feather").groupby(level="region", axis=1).sum()
        Z = feather.read_feather(pathIOT + "Z.feather").fillna(0)
        Kbar = feather.read_feather("Data/Kbar/Kbar_" + str(i) + ".feather").fillna(0)

        Zk = pd.DataFrame(
            Z + Kbar, index=Z.index, columns=Z.columns
        )  # EXTREMELY important to reindex, otherwise pymrio.calc_L doesn't work properly
        x = Z.sum(axis=1) + Y.sum(axis=1)
        Ak = pymrio.calc_A(Zk, x)
        feather.write_feather(pymrio.calc_L(Ak), pathIOT + "Lk.feather")
        A = pymrio.calc_A(Z, x)
        feather.write_feather(pymrio.calc_L(A), pathIOT + "L.feather")

    return None


def LkYhealth():
    if not os.path.exists("Data/LkYhealth"):
        os.mkdir("Data/LkYhealth")
    for i in range(1995, 2016, 1):
        pathIOT = pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/"
        Y = feather.read_feather(pathIOT + "Y.feather").groupby(level="region", axis=1).sum()

        Lk = feather.read_feather(pathIOT + "Lk.feather")
        Lk.columns.names = ["region", "sector"]
        Lk.index.names = ["region", "sector"]

        LkYhealth = pd.DataFrame()

        for j in Y.columns:  # regions
            Y_i_j_200_49 = (
                Y[j]
                .unstack()
                .div(Y[j].unstack().sum(), axis=1)  # reg of prooduction of sector
                .mul(Y_health_NHA_euros.loc[i].loc[j], axis=1)
            ).stack()

            LkYhealth[j] = (
                Lk.mul(Y_i_j_200_49, axis=1)
                .groupby(level="sector", axis=1)
                .sum()[
                    [
                        "Insurance and pension funding services, except compulsory social security services (66)",
                        "Public administration and defence services; compulsory social security services (75)",
                        "Private households with employed persons (95)",
                        "Retail  trade services, except of motor vehicles and motorcycles; repair services of personal and household goods (52)",
                        "Health and social work services (85)",
                    ]
                ]
                .stack()
            )

        feather.write_feather(
            LkYhealth.unstack(),
            pathexio + "Data/LkYhealth/LkYhealth" + str(i) + ".feather",
        )


def LYhealth():
    if not os.path.exists("Data/LYhealth"):
        os.mkdir("Data/LYhealth")
    for i in range(2015, 2016, 1):
        pathIOT = pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/"
        Y = feather.read_feather(pathIOT + "Y.feather").groupby(level="region", axis=1).sum()

        L = feather.read_feather(pathIOT + "L.feather")
        L.columns.names = ["region", "sector"]
        L.index.names = ["region", "sector"]

        LYhealth = pd.DataFrame()

        for j in Y.columns:  # regions
            Y_i_j_200_49 = (
                Y[j]
                .unstack()
                .div(Y[j].unstack().sum(), axis=1)  # reg of prooduction of sector
                .mul(Y_health_NHA_euros.loc[i].loc[j], axis=1)
            ).stack()

            LYhealth[j] = (
                L.mul(Y_i_j_200_49, axis=1)
                .groupby(level="sector", axis=1)
                .sum()[
                    [
                        "Insurance and pension funding services, except compulsory social security services (66)",
                        "Public administration and defence services; compulsory social security services (75)",
                        "Private households with employed persons (95)",
                        "Retail  trade services, except of motor vehicles and motorcycles; repair services of personal and household goods (52)",
                        "Health and social work services (85)",
                    ]
                ]
                .stack()
            )

        feather.write_feather(
            LYhealth.unstack(),
            pathexio + "Data/LYhealth/LYhealth" + str(i) + ".feather",
        )


def SLkYhealth():
    SLkYhealth_imp = pd.DataFrame()
    SLkYhealth_sat = pd.DataFrame()
    for i in range(1995, 2016, 1):
        LkYhealth = feather.read_feather(pathexio + "Data/LkYhealth/LkYhealth" + str(i) + ".feather")
        pathIOT = pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/"

        S = pd.read_csv(pathIOT + "impacts/S.txt", delimiter="\t", header=[0, 1], index_col=[0])
        SLkYhealth_i = pd.DataFrame()
        for j in S.index:
            SLkYhealth_i[j] = LkYhealth.mul(S.loc[j], axis=0).sum()
        SLkYhealth_imp[i] = SLkYhealth_i.stack()

        S = pd.read_csv(pathIOT + "satellite/S.txt", delimiter="\t", header=[0, 1], index_col=[0])
        SLkYhealth_i = pd.DataFrame()
        for j in S.index:
            SLkYhealth_i[j] = LkYhealth.mul(S.loc[j], axis=0).sum()
        SLkYhealth_sat[i] = SLkYhealth_i.stack()

    SLkYhealth_imp.index.names = ["region", "sector", "extension"]
    SLkYhealth_sat.index.names = ["region", "sector", "extension"]

    feather.write_feather(SLkYhealth_imp, "results/SLkYhealth_imp.feather")
    feather.write_feather(SLkYhealth_sat, "results/SLkYhealth_sat.feather")


def imports_dependency():
    imports_imp = pd.DataFrame()
    imports_sat = pd.DataFrame()

    for i in [2013]:
        LkYhealth = feather.read_feather(pathexio + "Data/LkYhealth/LkYhealth" + str(i) + ".feather")
        pathIOT = pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/"

        S = pd.read_csv(pathIOT + "impacts/S.txt", delimiter="\t", header=[0, 1], index_col=[0])
        imports_i = pd.DataFrame()
        for j in S.index:
            imports = LkYhealth.groupby(level=0, axis=1).sum().mul(S.loc[j], axis=0).groupby(level="region").sum()
            imports.index.names = ["region prod"]
            imports.columns.names = ["region cons"]
            imports_i[j] = imports.stack()
        imports_imp[i] = imports_i.stack()

        S = pd.read_csv(pathIOT + "satellite/S.txt", delimiter="\t", header=[0, 1], index_col=[0])
        imports_i = pd.DataFrame()
        for j in S.index:
            imports = LkYhealth.groupby(level=0, axis=1).sum().mul(S.loc[j], axis=0).groupby(level="region").sum()
            imports.index.names = ["region prod"]
            imports.columns.names = ["region cons"]
            imports_i[j] = imports.stack()
        imports_sat[i] = imports_i.stack()

        imports_imp.to_csv("results/imports_imp" + str(i) + ".txt")
        imports_sat.to_csv("results/imports_sat" + str(i) + ".txt")


def export_results_with_units():
    SLkYhealth_imp = feather.read_feather("results/SLkYhealth_imp.feather")
    SLkYhealth_sat = feather.read_feather("results/SLkYhealth_sat.feather")

    SLkYhealth_imp.columns = SLkYhealth_imp.columns.astype(int)
    SLkYhealth_sat.columns = SLkYhealth_sat.columns.astype(int)

    impacts_ext_excel = SLkYhealth_imp.unstack(level="region").groupby(level="extension").sum()
    impacts_ext_excel["unit"] = pd.read_excel("Data/unit impacts.xlsx", header=0, index_col=0)["unit"]
    impacts_ext_excel = impacts_ext_excel.reset_index().set_index(["unit", "extension"])
    impacts_ext_excel = impacts_ext_excel.swaplevel()
    names = pd.read_excel("Data/region names.xlsx", index_col=None)
    dict_names = dict(zip(names["region"], names["full name"]))
    impacts_ext_excel.rename(columns=dict_names).to_excel("results/impacts_ext_all.xlsx")

    satellite_ext_excel = SLkYhealth_sat.unstack(level="region").groupby(level="extension").sum()
    satellite_ext_excel["unit"] = pd.read_excel("Data/unit satellite.xlsx", header=0, index_col=0)["unit"]
    satellite_ext_excel = satellite_ext_excel.reset_index().set_index(["unit", "extension"])
    satellite_ext_excel = satellite_ext_excel.swaplevel()
    names = pd.read_excel("Data/region names.xlsx", index_col=None)
    dict_names = dict(zip(names["region"], names["full name"]))
    satellite_ext_excel.rename(columns=dict_names).to_excel("results/satellite_ext_all.xlsx")


def world_total():
    """Extracts world total impacts from exiobase and saves them in results folder.

    They will be used to calculate the share of healthcare impacts in the total.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    pba_sat = pd.DataFrame()
    for i in range(1995, 2014, 1):
        pba_sat[i] = pd.read_csv(
            pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/satellite/D_pba.txt",
            delimiter="\t",
            header=[0, 1],
            index_col=[0],
        ).sum(axis=1)

    pba_imp = pd.DataFrame()
    for i in range(1995, 2014, 1):
        pba_imp[i] = pd.read_csv(
            pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/impacts/D_pba.txt",
            delimiter="\t",
            header=[0, 1],
            index_col=[0],
        ).sum(axis=1)

    F_sat = pd.DataFrame()
    for i in range(1995, 2014, 1):
        F_sat[i] = pd.read_csv(
            pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/satellite/F_Y.txt",
            delimiter="\t",
            header=[0, 1],
            index_col=[0],
        ).sum(axis=1)

    F_imp = pd.DataFrame()
    for i in range(1995, 2014, 1):
        F_imp[i] = pd.read_csv(
            pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/impacts/F_Y.txt",
            delimiter="\t",
            header=[0, 1],
            index_col=[0],
        ).sum(axis=1)
    (F_sat + pba_sat).to_csv("results/total_sat.csv")
    (F_imp + pba_imp).to_csv("results/total_imp.csv")
