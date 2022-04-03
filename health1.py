import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymrio
import scipy.io
from sklearn.linear_model import LinearRegression
from matplotlib import colors as mcolors
import math
import country_converter as coco

cc = coco.CountryConverter()

pathexio = "C:/Users/andrieba/Documents/"


def plt_rcParams():
    fsize = 10
    tsize = 4
    tdir = "out"
    major = 5.0
    minor = 3.0
    lwidth = 0.8
    lhandle = 2.0
    plt.style.use("default")
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = fsize
    plt.rcParams["legend.fontsize"] = tsize
    plt.rcParams["xtick.direction"] = tdir
    plt.rcParams["ytick.direction"] = tdir
    plt.rcParams["xtick.major.size"] = major
    plt.rcParams["xtick.minor.size"] = minor
    plt.rcParams["ytick.major.size"] = 3.0
    plt.rcParams["ytick.minor.size"] = 1.0
    plt.rcParams["axes.linewidth"] = lwidth
    plt.rcParams["legend.handlelength"] = lhandle
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.axisbelow"] = True
    return None


plt_rcParams()

# mcolors.to_rgba('darkblue')

XR = pd.read_excel("exchange rates.xlsx").loc[0]
# exchange rate for euro zone, euros/dollars
pop = pd.read_excel("pop.xlsx", index_col=[0]).drop([2018, 2019], axis=1)

Yhealth = pd.DataFrame()  # Exiobase health sector expenditures
for i in range(1995, 2019, 1):
    pathIOT = pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/"
    Yhealth[i] = (
        pd.read_csv(pathIOT + "Y.txt", delimiter="\t", header=[0, 1], index_col=[0, 1])
        .groupby(level="region", axis=1)
        .sum()
        .groupby(level="sector")
        .sum()
        .loc["Health and social work services (85)"]
    )
# we will compare these expenditures with NHA in order to use the 2000 share for years 19995-1999

expendituresNHA = (
    pd.read_excel("dépenses santé NHA 2018.xlsx", header=[0], index_col=[0])
    .groupby(level="region")
    .sum()
    / 1000
)  # WHO expenditures in NHA format
expendituresOECD_95_99 = pd.read_excel(
    "dépenses santé OCDE.xlsx", header=[0], index_col=[0]
)
# OECD expenditures for missing years in WHO
expendituresNHA = pd.concat([expendituresOECD_95_99, expendituresNHA], axis=1)
# expenditures in NHA format for WHO and OECD data
expendituresNHAeuros = expendituresNHA * XR
# from current US$ to current euros

shareTW = Yhealth.loc["TW"] / (Yhealth.loc["CN"] + Yhealth.loc["TW"])
# Taiwan is not in the GHED so we assume its share of China is that of Exiobase sector 'Health and social work services (85)'
expendituresNHAeuros49 = expendituresNHAeuros.T.drop(["CN"], axis=1)
expendituresNHAeuros49["CN"] = (1 - shareTW).values * expendituresNHAeuros.T["CN"]
expendituresNHAeuros49["TW"] = (shareTW).values * expendituresNHAeuros.T["CN"]
expendituresNHAeuros49 = expendituresNHAeuros49.T

shareExioOverNHA = (
    (Yhealth / expendituresNHAeuros49).interpolate(method="linear", axis=1).T.bfill().T
)
# We consider that for the period 1995-2000 the ratio of expenditures in the NHA format and of the exiobase sector (85)
# stays constant for the countries not in the OECD database

expendituresNHAeuros49extrapolated = Yhealth / shareExioOverNHA
# the total health expenditures in the NHA format extrapolated for years 1995-1999

health_shares_OECD = (
    pd.read_excel("Health Shares OECD.xlsx", index_col=[0, 1], header=[0, 1]) / 100
)
health_shares_OECD = pd.DataFrame(
    health_shares_OECD.stack()
    .T.interpolate(method="linear", axis=0)
    .bfill()
    .stack(level=0)
    .stack(level=0),
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
    shares_multiplied_by_expenditures.sum(axis=1)
    / expenditures_with_appropriate_NaNValues.sum(axis=1)
).reorder_levels([2, 0, 1])
for i in health_shares_OECD.index:
    for j in health_shares_OECD.columns:
        if math.isnan(health_shares_OECD.loc[i, j]) == True:
            health_shares_OECD.loc[i, j] = mean_shares_OECD.loc[i]
health_shares_OECD[
    "WM"
] = mean_shares_OECD  # only Israel with atypical "Retailers and other providers of medical goods"
health_shares_OECD = health_shares_OECD.unstack(level=0).groupby(level="sector").sum()
health_shares_OECD.loc["Health and social work services (85)"] = (
    1 - health_shares_OECD.sum()
)

i = 2015
pathIOT = pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/"
sectors = (
    pd.read_csv(pathIOT + "Y.txt", delimiter="\t", header=[0, 1], index_col=[0, 1])
    .loc["FR"]
    .index
)

path = pathexio + "GitHub/exiobase-ISTerre/"
XReuros = pd.read_excel(path + "data_CPI/exchange rates euro area.xlsx").loc[0]
# euro area XR read from OECD database
indexCPI2017 = pd.read_excel(
    path + "data_CPI/index CPI 2017, US=1.xlsx",
    header=[0],
    index_col=[0],
)
indexCPI2011 = pd.read_excel(
    path + "data_CPI/index CPI 2011, US=1.xlsx",
    header=[0],
    index_col=[0],
)
indexCPI2005 = pd.read_excel(
    path + "data_CPI/index CPI 2005, US=1.xlsx",
    header=[0],
    index_col=[0],
)
indexCPI1996 = pd.read_excel(
    path + "data_CPI/index CPI 1996, US=1.xlsx",
    header=[0],
    index_col=[0],
)
index = (
    pd.read_csv("index_health.txt", index_col=[0, 1])["0"]
    .unstack()
    .mul(XReuros.drop([1995, 2018]), axis=0)
)


dict_regions = cc.get_correspondence_dict("name_short", "EXIO3")
ppp_all_regions = (
    pd.read_excel(
        "NHA indicators - per capita exp ppp.xlsx", index_col=[0], header=[0]
    )[1:]
    .drop(["Indicators", "Unnamed: 2"], axis=1)
    .stack()
    .replace(dict({":": np.nan}))
    .astype("float")
)
pop_all_regions = (
    pd.read_excel("NHA indicators - population.xlsx", index_col=[0], header=[0])[1:]
    .drop(["Indicators", "Unnamed: 2"], axis=1)
    .stack()
    .replace(dict({":": np.nan}))
    .astype("float")
)
current_ppp = (ppp_all_regions * pop_all_regions).unstack().drop("2018", axis=1)
current_ppp["region"] = cc.convert(names=current_ppp.index, to="EXIO3")
current_ppp = current_ppp.reset_index()
current_ppp = current_ppp.set_index("region").groupby(level="region").sum()
current_ppp.columns = [int(i) for i in current_ppp.columns]

CPI_US = pd.read_excel("CPI US eurostat.xlsx", index_col=0, header=0)
CPI_US = CPI_US / CPI_US.loc["US"][2017]
constant_ppp = current_ppp.drop([2000, 2001], axis=1).div(CPI_US.loc["US"], axis=1)

current_all_regions = (
    pd.read_excel("NHA indicators - per capita exp.xlsx", index_col=[0], header=[0])[1:]
    .drop(["Indicators", "Unnamed: 2"], axis=1)
    .stack()
    .replace(dict({":": np.nan}))
    .astype("float")
)
current = (current_all_regions * pop_all_regions).unstack().drop("2018", axis=1)
current["region"] = cc.convert(names=current.index, to="EXIO3")
current = current.reset_index()
current = current.set_index("region").groupby(level="region").sum()
current.columns = [int(i) for i in current.columns]


Y_health_NHA = health_shares_OECD.T.mul(
    expendituresNHAeuros49extrapolated.stack(), axis=0
)
Y_health_NHA = pd.DataFrame(Y_health_NHA, columns=sectors).swaplevel()
exp_cap = Y_health_NHA.sum(axis=1).drop(2018) / pop.stack()

Y_health_NHA_ppp = (
    Y_health_NHA.sum(axis=1).drop([1995, 2018])
    / pd.DataFrame(index.stack(), index=Y_health_NHA.drop([1995, 2018]).index)[0]
)
Y_health_NHA_ppp_cap = Y_health_NHA_ppp / pop.stack().swaplevel().drop([1995])


def LkYhealth():
    LkY = pd.DataFrame()
    for i in range(1995, 2019, 1):

        pathIOT = pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/"
        Y = (
            pd.read_csv(
                pathIOT + "Y.txt", delimiter="\t", header=[0, 1], index_col=[0, 1]
            )
            .groupby(level="region", axis=1)
            .sum()
        )

        Lk = pd.read_csv(
            pathIOT + "Lk.txt", delimiter=",", header=[0, 1], index_col=[0, 1]
        )
        Lk.columns.names = ["region", "sector"]
        Lk.index.names = ["region", "sector"]

        LkYhealth = pd.DataFrame()

        for j in Y.columns:  # regions

            Y_i_j_200_49 = (
                Y[j]
                .unstack()
                .div(Y[j].unstack().sum(), axis=1)  # reg of prooduction of sector
                .mul(Y_health_NHA.loc[i].loc[j], axis=1)
            ).stack()

            LkYhealth[j] = (
                Lk.mul(Y_i_j_200_49, axis=1)
                .groupby(level="sector", axis=1)
                .sum()[health_shares_OECD.index]
                .stack()
            )

        LkYhealth.to_csv(pathexio + "Data/LkYhealth/LkYhealth" + str(i) + ".txt")
        # columns : ['region cons'], index : ['region prod','sector prod','sector cons']


def SLkYhealth():
    SLkYhealth_imp = pd.DataFrame()
    SLkYhealth_sat = pd.DataFrame()
    for i in range(1995, 2019, 1):
        LkYhealth = pd.read_csv(
            pathexio + "Data/LkYhealth/LkYhealth" + str(i) + ".txt",
            header=[0],
            index_col=[0, 1, 2],
        ).unstack()
        pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/"

        S = pd.read_csv(
            pathIOT + "impacts/S.txt", delimiter="\t", header=[0, 1], index_col=[0]
        )
        SLkYhealth_i = pd.DataFrame()
        for j in S.index:
            SLkYhealth_i[j] = LkYhealth.mul(S.loc[j], axis=0).sum()
        SLkYhealth_imp[i] = SLkYhealth_i.stack()

        S = pd.read_csv(
            pathIOT + "satellite/S.txt", delimiter="\t", header=[0, 1], index_col=[0]
        )
        SLkYhealth_i = pd.DataFrame()
        for j in S.index:
            SLkYhealth_i[j] = LkYhealth.mul(S.loc[j], axis=0).sum()
        SLkYhealth_sat[i] = SLkYhealth_i.stack()
    SLkYhealth_imp.to_csv("Data/SLkYhealth/SLkYhealth_imp.txt")
    SLkYhealth_sat.to_csv("Data/SLkYhealth/SLkYhealth_sat.txt")


impacts_ext = [
    "GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)",
    "Carbon dioxide (CO2) CO2EQ IPCC categories 1 to 4 and 6 to 7 (excl land use, land use change and forestry)",
    "Methane (CH4) CO2EQ IPCC categories 1 to 4 and 6 to 7 (excl land use, land use change and forestry)",
    "Nitrous Oxide (N2O) CO2EQ IPCC categories 1 to 4 and 6 to 7 (excl land use, land use change and forestry)",
    "Carbon dioxide (CO2) IPCC categories 1 to 4 and 6 to 7 (excl land use, land use change and forestry)",
    "Methane (CH4) IPCC categories 1 to 4 and 6 to 7 (excl land use, land use change and forestry)",
    "Nitrous Oxide (N2O) IPCC categories 1 to 4 and 6 to 7 (excl land use, land use change and forestry)",
    "Water Consumption Blue - Total",
    "Water Withdrawal Blue - Total",
    "Nitrogen",
    "Domestic Extraction Used - Forestry and Timber",
    "Domestic Extraction Used - Non-metalic Minerals",
    "Domestic Extraction Used - Iron Ore",
    "Domestic Extraction Used - Non-ferous metal ores",
]
satellite_ext = [
    "HFC - air",
    "PFC - air",
    "SF6 - air",
    "Energy Carrier Net Total",
    "Energy Carrier Net NENE",
    "Energy Carrier Net NTRA",
    "Energy Carrier Net TAVI",
    "Energy Carrier Net TMAR",
    "Energy Carrier Net TOTH",
    "Energy Carrier Net TRAI",
    "Energy Carrier Net TROA",
    "Energy Carrier Net LOSS",
    "Energy Inputs from Nature: Total",
    "Emission Relevant Energy Carrier: Total",
    "Energy Carrier Supply: Total",
    "Energy Carrier Use: Total",
    "Domestic Extraction Used - Fossil Fuel: Total",
]

SLkYhealth_imp = pd.read_csv("Data/SLkYhealthSLkYhealth_imp.txt", index_col=[0, 1, 2])
SLkYhealth_sat = pd.read_csv("Data/SLkYhealthSLkYhealth_sat.txt", index_col=[0, 1, 2])
SLkYhealth_imp.columns = SLkYhealth_imp.columns.astype(int)
SLkYhealth_sat.columns = SLkYhealth_sat.columns.astype(int)

impacts = (
    SLkYhealth_imp.stack().unstack(level=1).sum(axis=1).unstack(level=1)[impacts_ext]
)
impacts_world = impacts.unstack().sum().unstack()

satellite = (
    SLkYhealth_sat.stack().unstack(level=1).sum(axis=1).unstack(level=1)[satellite_ext]
)
satellite_world = satellite.unstack(level=0).sum().unstack(level=0)


satellite_cap = satellite.div(
    pd.DataFrame(pop.stack(), index=satellite.index)[0], axis=0
)
impacts_cap = impacts.div(pd.DataFrame(pop.stack(), index=impacts.index)[0], axis=0)


QualityIndex = pd.read_excel(
    pathexio + "GitHub/health/Health Access and Quality Index.xlsx",
    header=0,
    index_col=0,
)
QualityIndex = (
    pd.DataFrame(QualityIndex, columns=Y_health_NHA.unstack().index)
    .T.interpolate(method="linear")
    .bfill()
    .T.drop([2017, 2018], axis=1)
)


pichler = pd.read_excel("pichler.xlsx", index_col=0)
arup = pd.read_excel("arup.xlsx", index_col=0)
lenzen = pd.read_excel("lenzen.xlsx", index_col=0)

df_lenzen = pd.concat(
    [
        impacts[
            "GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"
        ]
        .loc[lenzen.index]
        .unstack()[2015]
        / 1000
        / pop[2015].loc[lenzen.index]
        / 1000,
        lenzen["ges 2015"] / pop[2015].loc[lenzen.index] * 1000,
    ],
    keys=["calc ges", "lenzen"],
    axis=1,
)
plt.scatter(df_lenzen["lenzen"], df_lenzen["calc ges"])

df_arup = pd.concat(
    [
        impacts[
            "GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"
        ]
        .loc[arup.index]
        .unstack()[2014]
        / 1000
        / pop[2014].loc[arup.index]
        / 1000,
        arup["ges 2014"] / pop[2014].loc[arup.index] * 1000,
    ],
    keys=["calc ges", "arup"],
    axis=1,
)
plt.scatter(df_arup["arup"], df_arup["calc ges"])


for year in [1995, 2000, 2005, 2010, 2016]:
    # [[1995,2000,2005,2010,2016]].stack()
    x = QualityIndex[year]
    y_pri = satellite_cap.loc[QualityIndex.index]["Energy Carrier Net Total"].unstack()[
        year
    ]
    y_loss = satellite_cap.loc[QualityIndex.index]["Energy Carrier Net LOSS"].unstack()[
        year
    ]
    y = np.log(y_pri - y_loss)
    regression = scipy.stats.linregress(x, y)
    plt.scatter(x, y, color="darkblue")
    plt.plot(
        x,
        regression[1] + regression[0] * x,
    )
    plt.scatter(x, y)
    print(str(round(regression[2] ** 2, 2)))

for year in [1]:
    x = QualityIndex[[1995, 2000, 2005, 2010, 2016]].stack()
    y_pri = (
        satellite_cap.loc[QualityIndex.index]["Energy Carrier Net Total"]
        .unstack()[[1995, 2000, 2005, 2010, 2016]]
        .stack()
    )
    y_loss = (
        satellite_cap.loc[QualityIndex.index]["Energy Carrier Net LOSS"]
        .unstack()[[1995, 2000, 2005, 2010, 2016]]
        .stack()
    )
    y = np.log(y_pri - y_loss)
    regression = scipy.stats.linregress(x, y)
    plt.scatter(x, y, color="darkblue")
    plt.plot(
        x,
        regression[1] + regression[0] * x,
    )
    print(str(round(regression[2] ** 2, 2)))

for year in [1995, 2000, 2005, 2010, 2016]:
    # [[1995,2000,2005,2010,2016]].stack()
    x = QualityIndex[year]
    y = exp_cap.unstack()[QualityIndex.index].loc[year]
    y = np.log(y)
    regression = scipy.stats.linregress(x, y)
    plt.scatter(x, y, color="darkblue")
    plt.plot(
        x,
        regression[1] + regression[0] * x,
    )
    plt.scatter(x, y)
    print(str(round(regression[2] ** 2, 2)))

for year in [1]:
    x = QualityIndex[[1995, 2000, 2005, 2010, 2016]].stack()
    y_pri = (
        satellite_cap.loc[QualityIndex.index]["Energy Carrier Net Total"]
        .unstack()[[1995, 2000, 2005, 2010, 2016]]
        .stack()
    )
    y_loss = (
        satellite_cap.loc[QualityIndex.index]["Energy Carrier Net LOSS"]
        .unstack()[[1995, 2000, 2005, 2010, 2016]]
        .stack()
    )
    y = y_pri - y_loss

    plt.scatter(x, y, color="darkblue")


fig, axes = plt.subplots(4, figsize=(3, 10))
k = 0
# R2 = 0.05, 0.23, 0.05, 0.02
for index in [indexCPI1996, indexCPI2005, indexCPI2011, indexCPI2017]:
    year = [1996, 2005, 2011, 2017][k]
    ax = axes[k]

    ppp = (
        Y_health_NHA.sum(axis=1).drop(2018).loc[year]
        / XReuros.loc[year]
        / index["CPI: 06 - Health"]
    )
    MJ_pri = satellite["Energy Carrier Net Total"].unstack()[year]
    MJ_loss = satellite["Energy Carrier Net LOSS"].unstack()[year]
    MJ_fin = MJ_pri - MJ_loss
    y = MJ_fin / ppp
    x = exp_cap.loc[year] / XReuros.loc[year] / index["CPI: 06 - Health"]

    regression = scipy.stats.linregress(x, y)
    ax.scatter(x, y, color="darkblue")
    ax.plot(x, regression[1] + regression[0] * x, color="darkorange")
    ax.set_ylim([0, 3.5])
    print(str(round(regression[2] ** 2, 2)))
    k += 1


(
    (satellite["Energy Carrier Net Total"] - satellite["Energy Carrier Net LOSS"])
    .unstack()
    .drop(1995, axis=1)
    .sum()
    / Y_health_NHA_ppp.swaplevel().sum()
).plot()


(
    (
        (satellite["Energy Carrier Net Total"] - satellite["Energy Carrier Net LOSS"])
        .unstack()
        .drop(1995, axis=1)
        .stack()
    )
    .loc[constant_ppp.unstack().swaplevel().index]
    .unstack()
    .sum()
    / constant_ppp.unstack().swaplevel().unstack().sum()
    * 1000
).plot()
pow(1.484689040614855, 1 / 15)
# + 48%, + 2.5%/an
