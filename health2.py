import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymrio
import scipy.io
from sklearn.linear_model import LinearRegression
from matplotlib import colors as mcolors
import math
import country_converter as coco
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from sklearn.metrics import r2_score
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.io as pio
import pyarrow.feather as feather


pio.orca.config.executable = (
    "C:/Users/andrieba/anaconda3/pkgs/plotly-orca-1.2.1-1/orca_app/orca.exe"
)
pio.orca.config.save()
pio.kaleido.scope.mathjax = None  # ne sert à rien car mathjax ne fonctionne pas
pyo.init_notebook_mode()

cc = coco.CountryConverter()
pathexio = "C:/Users/andrieba/Documents/"


# ........CREATE FUNCTION TO CONVERT ANY REGION NAME FORMAT TO A COMMON FORMAT................

dict_regions = dict()  # create a dict that will be used to rename regions
cc = coco.CountryConverter(
    include_obsolete=True
)  # documentation for coco here: https://github.com/konstantinstadler/country_converter
for i in [
    n for n in cc.valid_class if n != "name_short"
]:  # we convert all the regions in cc to name short and add it to the dict
    dict_regions.update(cc.get_correspondence_dict(i, "name_short"))
name_short = cc.ISO3as("name_short")[
    "name_short"
].values  # array containing all region names in short_name format


def dict_regions_update():
    """Adds to dict the encountered region names that were not in coco.

    If a region is wider than a country (for example "European Union"), it is added to "Z - Aggregated categories" in order to be deleted later.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    dict_regions["Bolivia (Plurinational State of)"] = "Bolivia"
    dict_regions["Czechia"] = "Czech Republic"
    dict_regions["Iran (Islamic Republic of)"] = "Iran"
    dict_regions["China, Taiwan Province of China"] = "Taiwan"
    dict_regions["Congo"] = "Congo Republic"
    dict_regions["Venezuela (Bolivarian Republic of)"] = "Venezuela"
    dict_regions["Dem. People's Republic of Korea"] = "North Korea"
    dict_regions["Bahamas, The"] = "Bahamas"
    dict_regions["Congo, Dem. Rep."] = "DR Congo"
    dict_regions["Congo, Rep."] = "Congo Republic"
    dict_regions["Egypt, Arab Rep."] = "Egypt"
    dict_regions["Faroe Islands"] = "Faeroe Islands"
    dict_regions["Gambia, The"] = "Gambia"
    dict_regions["Hong Kong SAR, China"] = "Hong Kong"
    dict_regions["Iran, Islamic Rep."] = "Iran"
    dict_regions["Korea, Dem. People's Rep."] = "North Korea"
    dict_regions["Korea, Rep."] = "South Korea"
    dict_regions["Lao PDR"] = "Laos"
    dict_regions["Macao SAR, China"] = "Macau"
    dict_regions["North Macedonia"] = "Macedonia"
    dict_regions["Russian Federation"] = "Russia"
    dict_regions["Sint Maarten (Dutch part)"] = "Sint Maarten"
    dict_regions["Slovak Republic"] = "Slovakia"
    dict_regions["St. Martin (French part)"] = "Saint-Martin"
    dict_regions["Syrian Arab Republic"] = "Syria"
    dict_regions["Virgin Islands (U.S.)"] = "United States Virgin Islands"
    dict_regions["West Bank and Gaza"] = "Palestine"
    dict_regions["Yemen, Rep."] = "Yemen"
    dict_regions["Venezuela, RB"] = "Venezuela"
    dict_regions["Brunei"] = "Brunei Darussalam"
    dict_regions["Cape Verde"] = "Cabo Verde"
    dict_regions["Dem. People's Rep. Korea"] = "North Korea"
    dict_regions["Swaziland"] = "Eswatini"
    dict_regions["Taiwan, China"] = "Taiwan"
    dict_regions["Virgin Islands"] = "United States Virgin Islands"
    dict_regions["Yemen, PDR"] = "Yemen"
    dict_regions["Réunion"] = "Reunion"
    dict_regions["Saint Helena"] = "St. Helena"
    dict_regions["China, Hong Kong SAR"] = "Hong Kong"
    dict_regions["China, Macao SAR"] = "Macau"
    dict_regions[
        "Bonaire, Sint Eustatius and Saba"
    ] = "Bonaire, Saint Eustatius and Saba"
    dict_regions["Curaçao"] = "Curacao"
    dict_regions["Saint Barthélemy"] = "St. Barths"
    dict_regions["Saint Martin (French part)"] = "Saint-Martin"
    dict_regions["Micronesia (Fed. States of)"] = "Micronesia, Fed. Sts."
    dict_regions["Micronesia, Federated State=s of"] = "Micronesia, Fed. Sts."
    dict_regions["Bonaire"] = "Bonaire, Saint Eustatius and Saba"
    dict_regions["São Tomé and Principe"] = "Sao Tome and Principe"
    dict_regions["Virgin Islands, British"] = "British Virgin Islands"
    dict_regions["Wallis and Futuna"] = "Wallis and Futuna Islands"
    dict_regions["Micronesia, Federated States of"] = "Micronesia, Fed. Sts."

    dict_regions["VIR"] = "United States Virgin Islands"
    dict_regions["GMB"] = "Gambia"
    dict_regions["NAM"] = "Namibia"
    dict_regions["BHS"] = "Bahamas"
    dict_regions["The Bahamas"] = "Bahamas"
    dict_regions["The Gambia"] = "Gambia"
    dict_regions["Virgin Islands, U.S."] = "United States Virgin Islands"
    dict_regions["Congo, DRC"] = "DR Congo"
    dict_regions["Marshall Is."] = "Marshall Islands"
    dict_regions["Solomon Is."] = "Solomon Islands"
    dict_regions["Timor Leste"] = "Timor-Leste"

    for j in [
        "Africa Eastern and Southern",
        "Africa Western and Central",
        "Arab World",
        "Caribbean small states",
        "Central Europe and the Baltics",
        "Early-demographic dividend",
        "East Asia & Pacific",
        "East Asia & Pacific (excluding high income)",
        "East Asia & Pacific (IDA & IBRD countries)",
        "Euro area",
        "Europe & Central Asia",
        "Europe & Central Asia (excluding high income)",
        "Europe & Central Asia (IDA & IBRD countries)",
        "European Union",
        "Fragile and conflict affected situations",
        "Heavily indebted poor countries (HIPC)",
        "High income",
        "IBRD only",
        "IDA & IBRD total",
        "IDA blend",
        "IDA only",
        "IDA total",
        "Late-demographic dividend",
        "Latin America & Caribbean",
        "Latin America & Caribbean (excluding high income)",
        "Latin America & the Caribbean (IDA & IBRD countries)",
        "Least developed countries: UN classification",
        "Low & middle income",
        "Low income",
        "Lower middle income",
        "Middle East & North Africa",
        "Middle East & North Africa (excluding high income)",
        "Middle East & North Africa (IDA & IBRD countries)",
        "Middle income",
        "North America",
        "Not classified",
        "OECD members",
        "Other small states",
        "Pacific island small states",
        "Post-demographic dividend",
        "Pre-demographic dividend",
        "Small states",
        "South Asia",
        "South Asia (IDA & IBRD)",
        "Sub-Saharan Africa",
        "Sub-Saharan Africa (excluding high income)",
        "Sub-Saharan Africa (IDA & IBRD countries)",
        "Upper middle income",
        "World",
        "Arab League states",
        "China and India",
        "Czechoslovakia",
        "East Asia & Pacific (all income levels)",
        "East Asia & Pacific (IDA & IBRD)",
        "East Asia and the Pacific (IFC classification)",
        "EASTERN EUROPE",
        "Europe & Central Asia (all income levels)",
        "Europe & Central Asia (IDA & IBRD)",
        "Europe and Central Asia (IFC classification)",
        "European Community",
        "High income: nonOECD",
        "High income: OECD",
        "Latin America & Caribbean (all income levels)",
        "Latin America & Caribbean (IDA & IBRD)",
        "Latin America and the Caribbean (IFC classification)",
        "Low income, excluding China and India",
        "Low-income Africa",
        "Middle East & North Africa (all income levels)",
        "Middle East & North Africa (IDA & IBRD)",
        "Middle East (developing only)",
        "Middle East and North Africa (IFC classification)",
        "Other low-income",
        "Serbia and Montenegro",
        "Severely Indebted",
        "South Asia (IFC classification)",
        "Sub-Saharan Africa (all income levels)",
        "SUB-SAHARAN AFRICA (excl. Nigeria)",
        "Sub-Saharan Africa (IDA & IBRD)",
        "Sub-Saharan Africa (IFC classification)",
        "WORLD",
        "UN development groups",
        "More developed regions",
        "Less developed regions",
        "Least developed countries",
        "Less developed regions, excluding least developed countries",
        "Less developed regions, excluding China",
        "Land-locked Developing Countries (LLDC)",
        "Small Island Developing States (SIDS)",
        "World Bank income groups",
        "High-income countries",
        "Middle-income countries",
        "Upper-middle-income countries",
        "Lower-middle-income countries",
        "Low-income countries",
        "No income group available",
        "Geographic regions",
        "Latin America and the Caribbean",
        "Sustainable Development Goal (SDG) regions",
        "SUB-SAHARAN AFRICA",
        "NORTHERN AFRICA AND WESTERN ASIA",
        "CENTRAL AND SOUTHERN ASIA",
        "EASTERN AND SOUTH-EASTERN ASIA",
        "LATIN AMERICA AND THE CARIBBEAN",
        "AUSTRALIA/NEW ZEALAND",
        "OCEANIA (EXCLUDING AUSTRALIA AND NEW ZEALAND)",
        "EUROPE AND NORTHERN AMERICA",
        "EUROPE",
        "Holy See",
        "NORTHERN AMERICA",
        "East Asia & Pacific (ICP)",
        "Europe & Central Asia (ICP)",
        "Latin America & Caribbean (ICP)",
        "Middle East & North Africa (ICP)",
        "North America (ICP)",
        "South Asia (ICP)",
        "Sub-Saharan Africa (ICP)",
        "Andean Latin America",
        "Australasia",
        "Central Latin America",
        "Central Sub-Saharan Africa",
        "East Asia",
        "Eastern Sub-Saharan Africa",
        "Global",
        "High-income",
        "High-income Asia Pacific",
        "High-income North America",
        "Latin America and Caribbean",
        "North Africa and Middle East",
        "Southeast Asia",
        "Southern Latin America",
        "Southern Sub-Saharan Africa",
        "Tropical Latin America",
        "Western Sub-Saharan Africa",
        "Central Europe",
        "Oceania",
        "Central Asia",
        "Western Europe",
        "Eastern Europe",
    ]:
        dict_regions[j] = "Z - Aggregated categories"
    return None


dict_regions_update()
# all the regions that do not correspond to a country are in 'Z - Aggregated categories'
# rename the appropriate level of dataframe using dict_regions
def rename_region(df, level="LOCATION"):
    """Renames the regions of a DataFrame into name_short format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose regions must be renamed
    level : string
        Name of the level containing the region names

    Returns
    df : pd.DataFrame
        DataFrame with regions in name_short format
    -------
    None
    """
    if level in df.index.names:
        axis = 0
    else:
        axis = 1
        df = df.T

    index_names = df.index.names
    df = df.reset_index()
    df = df.set_index(level)
    df = df.rename(index=dict_regions)  # rename index according to dict
    ind = df.index.values
    for i in range(0, len(ind), 1):
        if type(ind[i]) == list:
            # if len(ind[i])==0:
            ind[i] = ind[i][0]
    df = df.reindex(ind)
    df = df.reset_index().set_index(index_names)
    for i in df.index.get_level_values(level).unique():
        if i not in name_short and i != "Z - Aggregated categories":
            print(
                i
                + " is not in dict_regions\nAdd it using\n  >>> dict_regions['"
                + i
                + "'] = 'region' # name_short format\n"
            )
    if axis == 1:
        df = df.T
    return df


# ......


population = pd.read_csv(
    "worldbank population/API_SP.POP.TOTL_DS2_en_csv_v2_3852487.csv",
    header=[2],
    index_col=0,
)
population = rename_region(population, "Country Name")[
    ["1995", "2000", "2005", "2010", "2015"]
].drop("Z - Aggregated categories")
population.loc["Taiwan"] = [21356033, 21966527, 22705713, 23187551, 23618200]
population.columns = [1995, 2000, 2005, 2010, 2015]
pop_agg = population
pop_agg["region"] = cc.convert(names=pop_agg.index, to="EXIO3")
pop_agg = (
    pop_agg.reset_index()
    .set_index("region")
    .drop("Country Name", axis=1)
    .groupby(level="region")
    .sum()
)

HAQ = (
    (
        rename_region(
            pd.read_csv(
                "HAQ 2015.csv",
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
    HAQ_agg.reset_index()
    .set_index("region")
    .drop("location_name", axis=1)
    .groupby(level="region")
    .sum()
) / pop_agg


# ......


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


XReuros = pd.read_excel("exchange rates.xlsx").loc[0].drop([2016, 2017, 2018])
# exchange rate for euro zone, euros/dollars
pop = pd.read_excel("pop.xlsx", index_col=[0]).drop([2016, 2017, 2018, 2019], axis=1)


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
        pd.read_excel("dépenses santé NHA 2018.xlsx", header=[0], index_col=[0])
        .groupby(level="region")
        .sum()
        / 1000
    ).drop(
        [2016, 2017, 2018], axis=1
    )  # WHO expenditures in NHA format in kUS$
    expendituresOECD_95_99 = pd.read_excel(
        "dépenses santé OCDE.xlsx", header=[0], index_col=[0]
    )
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

    shareExioOverNHA = (
        (Yhealth / expendituresNHAeuros49)
        .interpolate(method="linear", axis=1)
        .T.bfill()
        .T
    )
    # We consider that for the period 1995-2000 the ratio of expenditures in the NHA format and of the exiobase sector (85)
    # stays constant for the countries not in the OECD database

    expendituresNHAeuros49extrapolated = Yhealth / shareExioOverNHA
    # the total health expenditures in the NHA format extrapolated for years 1995-1999

    health_shares_OECD = (
        pd.read_excel("Health Shares OECD.xlsx", index_col=[0, 1], header=[0, 1]) / 100
    ).drop([2016, 2017, 2018], axis=1)
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
        (
            health_shares_OECD.unstack(level=0)
            * expendituresNHAeuros49extrapolated.stack()
        )
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
    health_shares_OECD = (
        health_shares_OECD.unstack(level=0).groupby(level="sector").sum()
    )
    health_shares_OECD.loc["Health and social work services (85)"] = (
        1 - health_shares_OECD.sum()
    )

    i = 2015
    pathIOT = pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/"
    sectors = feather.read_feather(pathIOT + "Y.feather").loc["FR"].index

    Y_health_NHA_euros = health_shares_OECD.T.mul(
        expendituresNHAeuros49extrapolated.stack(), axis=0
    )
    Y_health_NHA_euros = pd.DataFrame(Y_health_NHA_euros, columns=sectors).swaplevel()

    return Y_health_NHA_euros


Y_health_NHA_euros = Y_health_NHA_euros()


##################


def ppp_health():

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
    current_ppp = (
        (ppp_all_regions * pop_all_regions)
        .unstack()
        .drop(["2016", "2017", "2018"], axis=1)
    )
    current_ppp["region"] = cc.convert(names=current_ppp.index, to="EXIO3")
    current_ppp = current_ppp.reset_index()
    current_ppp = current_ppp.set_index("region").groupby(level="region").sum()
    current_ppp.columns = [int(i) for i in current_ppp.columns]

    CPI_US = pd.read_excel("CPI US eurostat.xlsx", index_col=0, header=0).drop(
        [2016, 2017], axis=1
    )
    CPI_US = CPI_US / CPI_US.loc["US"][2015]
    constant_ppp = current_ppp.drop([2000, 2001], axis=1).div(CPI_US.loc["US"], axis=1)

    return constant_ppp


constant_ppp = ppp_health()


def LkYhealth():
    LkY = pd.DataFrame()
    for i in range(1995, 2016, 1):

        pathIOT = pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/"
        Y = (
            feather.read_feather(pathIOT + "Y.feather")
            .groupby(level="region", axis=1)
            .sum()
        )

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


def SLkYhealth():
    SLkYhealth_imp = pd.DataFrame()
    SLkYhealth_sat = pd.DataFrame()
    for i in range(1995, 2016, 1):
        LkYhealth = feather.read_feather(
            pathexio + "Data/LkYhealth/LkYhealth" + str(i) + ".feather"
        )
        pathIOT = pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/"

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

    SLkYhealth_imp.index.names = ["region", "sector", "extension"]
    SLkYhealth_sat.index.names = ["region", "sector", "extension"]

    feather.write_feather(SLkYhealth_imp, "SLkYhealth_imp.feather")
    feather.write_feather(SLkYhealth_sat, "SLkYhealth_sat.feather")


def imports_dependency():
    imports_imp = pd.DataFrame()
    imports_sat = pd.DataFrame()

    for i in [2013]:
        LkYhealth = feather.read_feather(
            pathexio + "Data/LkYhealth/LkYhealth" + str(i) + ".feather"
        ).unstack()
        pathIOT = pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/"

        S = pd.read_csv(
            pathIOT + "impacts/S.txt", delimiter="\t", header=[0, 1], index_col=[0]
        )
        SLkYhealth_i = pd.DataFrame()
        imports_i = pd.DataFrame()
        for j in S.index:
            imports = (
                LkYhealth.groupby(level=0, axis=1)
                .sum()
                .mul(S.loc[j], axis=0)
                .groupby(level="region")
                .sum()
            )
            imports.index.names = ["region prod"]
            imports.columns.names = ["region cons"]
            imports_i[j] = imports.stack()
        imports_imp[i] = imports_i.stack()

        S = pd.read_csv(
            pathIOT + "satellite/S.txt", delimiter="\t", header=[0, 1], index_col=[0]
        )
        imports_i = pd.DataFrame()
        for j in S.index:
            imports = (
                LkYhealth.groupby(level=0, axis=1)
                .sum()
                .mul(S.loc[j], axis=0)
                .groupby(level="region")
                .sum()
            )
            imports.index.names = ["region prod"]
            imports.columns.names = ["region cons"]
            imports_i[j] = imports.stack()
        imports_sat[i] = imports_i.stack()

        imports_imp.to_csv("imports_imp" + str(i) + ".txt")
        imports_sat.to_csv("imports_sat" + str(i) + ".txt")


###
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

# SLkYhealth_imp = pd.read_csv(pathexio + "SLkYhealth_imp.txt", index_col=[0, 1, 2])
# SLkYhealth_sat = pd.read_csv(pathexio + "SLkYhealth_sat.txt", index_col=[0, 1, 2])


SLkYhealth_imp = feather.read_feather("SLkYhealth_imp.feather")
SLkYhealth_sat = feather.read_feather("SLkYhealth_sat.feather")

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


def world_total():
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
            pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/satellite/F_hh.txt",
            delimiter="\t",
            header=[0, 1],
            index_col=[0],
        ).sum(axis=1)

    F_imp = pd.DataFrame()
    for i in range(1995, 2014, 1):
        F_imp[i] = pd.read_csv(
            pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/impacts/F_hh.txt",
            delimiter="\t",
            header=[0, 1],
            index_col=[0],
        ).sum(axis=1)
    (F_sat + pba_sat).to_csv("results/total_sat.csv")
    (F_imp + pba_imp).to_csv("results/total_imp.csv")


# world_total()
total_sat = pd.read_csv("results/total_sat.csv", index_col=0).loc[satellite_ext]
total_imp = pd.read_csv("results/total_imp.csv", index_col=0).loc[impacts_ext]

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


def comparison():
    df_pichler = pd.concat(
        [
            impacts[
                "Carbon dioxide (CO2) IPCC categories 1 to 4 and 6 to 7 (excl land use, land use change and forestry)"
            ]
            .loc[pichler.index]
            .unstack()[2014]
            / pop[2014].loc[pichler.index],
            pichler["CO2 2014"] / pop[2014].loc[pichler.index] * 1000,
        ],
        keys=["calc co2", "pichler"],
        axis=1,
    )
    fig, axes = plt.subplots(1, figsize=(5, 5))
    axes.scatter(df_pichler["calc co2"], df_pichler["pichler"])
    regression = scipy.stats.linregress(df_pichler["pichler"], df_pichler["calc co2"])
    axes.plot(df_pichler["calc co2"], df_pichler["calc co2"])
    # print(str(regression[2] ** 2) + 'regression pich')
    print(
        str(r2_score(df_pichler["calc co2"], df_pichler["pichler"])) + "r2 score pich"
    )
    # print(str(r2_score(df_pichler["calc co2"], df_pichler["pichler"]))+'r2 score pich')
    # print(str(r2_score(df_pichler["pichler"], df_pichler["pichler"] * regression[0] + regression[1]))+'r2 score pich')
    # print(str(r2_score(df_pichler["pichler"] * regression[0] + regression[1], df_pichler["pichler"]))+'r2 score pich')
    # print(str(r2_score(df_pichler["calc co2"], df_pichler["pichler"] * regression[0] + regression[1]))+'r2 score pich')
    # print(str(r2_score(df_pichler["pichler"] * regression[0] + regression[1], df_pichler["calc co2"]))+'r2 score pich')
    axes.set_xlabel("This study ($tCO^2/capita$)")
    axes.set_ylabel("Pichler et al. ($tCO^2/capita$)")
    for i in df_pichler["calc co2"].index:
        axes.annotate(
            i,
            (df_pichler["calc co2"].loc[i], df_pichler["pichler"].loc[i]),
        )
    plt.tight_layout()
    plt.savefig("figures/comp Pichler.svg")

    fig, axes = plt.subplots(1, figsize=(5, 5))
    axes.scatter(df_lenzen["calc ges"], df_lenzen["lenzen"])
    regression = scipy.stats.linregress(df_lenzen["lenzen"], df_lenzen["calc ges"])
    axes.plot(df_lenzen["calc ges"], df_lenzen["calc ges"])
    # print(str(regression[2] ** 2) + 'regression lenz')
    print(str(r2_score(df_lenzen["calc ges"], df_lenzen["lenzen"])) + "r2 score lenz")
    # print(str(r2_score(df_lenzen["calc ges"], df_lenzen["lenzen"]))+'r2 score lenz')
    axes.set_xlabel("This study ($tCO^2eq/capita$)")
    axes.set_ylabel("Lenzen et al. ($tCO^2eq/capita$)")
    for i in df_lenzen["calc ges"].index:
        axes.annotate(
            i,
            (df_lenzen["calc ges"].loc[i], df_lenzen["lenzen"].loc[i]),
        )
    plt.tight_layout()
    plt.savefig("figures/comp Lenzen.svg", bbox="tight")

    fig, axes = plt.subplots(1, figsize=(5, 5))
    axes.scatter(df_arup["calc ges"], df_arup["arup"])
    regression = scipy.stats.linregress(df_arup["arup"], df_arup["calc ges"])
    axes.plot(df_arup["calc ges"], df_arup["calc ges"])
    # print(str(regression[2] ** 2) + 'regression arup')
    print(str(r2_score(df_arup["calc ges"], df_arup["arup"])) + "r2 score arup")
    # print(str(r2_score(df_arup["calc ges"], df_arup["arup"]))+'r2 score arup')
    axes.set_xlabel("This study ($tCO^2eq/capita$)")
    axes.set_ylabel("Health Care Without Harm ($tCO^2eq/capita$)")
    for i in df_arup["calc ges"].index:
        axes.annotate(
            i,
            (df_arup["calc ges"].loc[i], df_arup["arup"].loc[i]),
        )
    plt.tight_layout()
    plt.savefig("figures/comp arup.svg")

    print(
        str(
            impacts[
                "Carbon dioxide (CO2) IPCC categories 1 to 4 and 6 to 7 (excl land use, land use change and forestry)"
            ]
            .loc["AT"]
            .loc[2014]
            / 1000
            / 6.8
        )
        + "Weisz Austria 2014"
    )  # Weisz 6.8 MtCO2 2014

    print(
        str(
            impacts[
                "GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"
            ]
            .loc["JP"]
            .loc[2015]
            / 1000000000
            / 72
        )
        + "Nansai Japan 2015"
    )  # Nansai 72.0 MtCO2e

    print(
        str(
            impacts[
                "GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"
            ]
            .loc["US"]
            .loc[2013]
            / 1000000000
            / 655
        )
        + "Eckelman US 2013"
    )  # Eckelman 655 MtCO2e

    print(
        str(
            impacts[
                "GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"
            ]
            .loc["GB"]
            .loc[2015]
            / pop[2015].loc["GB"]
            / 1000
            / 498
        )
        + "Tennison GB 2015"
    )  # Tennison  MtCO2e vs. 498 per capita exio

    print(
        str(
            impacts[
                "GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"
            ]
            .loc["CA"]
            .loc[2015]
            / 1000000000
            / 33.0
        )
        + "Eckelman CA 2015"
    )  # Eckelman 33.0 MtCO2e vs. 26.4 exio

    print(
        str(
            impacts[
                "GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"
            ]
            .loc["AU"]
            .loc[2015]
            / 1000000000
            / 35.8
        )
        + "Malik AU 2015"
    )  # Malik 35.8 MtCO2e vs. 29.0 exio

    print(
        str(
            impacts[
                "GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"
            ]
            .loc["CN"]
            .loc[2012]
            / 1000000000
            / 315
        )
        + "Wu CN 2012"
    )  # Wu 315 MtCO2e vs. 472 exio

    (abs(1 - df_pichler["pichler"] / df_pichler["calc co2"])).mean()
    (abs(1 - df_lenzen["lenzen"] / df_lenzen["calc ges"])).mean()
    (abs(1 - df_arup["arup"] / df_arup["calc ges"])).mean()

    (abs(np.log(df_pichler["pichler"] / df_pichler["calc co2"]))).mean()
    (abs(np.log(df_lenzen["lenzen"] / df_lenzen["calc ges"]))).mean()
    (abs(np.log(df_arup["arup"] / df_arup["calc ges"]))).mean()


comparison()


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
pow(1.830727 / 1.233071, 1 / 15)
# + 2.7%/an

# sns.color_palette("colorblind", as_cmap="True").append('olive')


def sankey():
    i = 2013
    df = pd.read_csv("imports_imp" + str(i) + ".csv", index_col=[2, 0, 1])
    continent = pd.read_excel("continent.xlsx", index_col=[0, 1])
    df = df.rename(index=dict(continent.index))

    ind = (
        df.loc["Domestic Extraction Used - Non-ferous metal ores"]
        .groupby(level=[0, 1])
        .sum()
        .unstack()
        .index
    )
    dict_source = dict(zip(ind.values, [i for i in range(0, len(ind), 1)]))
    dict_target = dict(zip(ind.values, [i for i in range(len(ind), len(ind) * 2, 1)]))

    sankey_metal = (
        df.loc["Domestic Extraction Used - Non-ferous metal ores"]
        .groupby(level=[0, 1])
        .sum()
    )
    sankey_metal = (
        sankey_metal.unstack()
        .rename(index=dict_source, columns=dict_target)
        .stack()
        .reset_index()
    )
    sankey_metal.columns = ["source", "target", "value"]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=ind.append(ind),
                    color="blue",
                ),
                link=dict(
                    source=sankey_metal["source"],
                    target=sankey_metal["target"],
                    value=sankey_metal["value"],
                ),
            )
        ]
    )

    fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
    fig.show()

    i = 2013
    df = pd.read_csv("imports_sat" + str(i) + ".csv", index_col=[2, 0, 1])
    continent = pd.read_excel("continent.xlsx", index_col=[0, 1])
    df = df.rename(index=dict(continent.index))

    ind = (
        df.loc["Domestic Extraction Used - Fossil Fuel: Total"]
        .groupby(level=[0, 1])
        .sum()
        .unstack()
        .index
    )
    dict_source = dict(zip(ind.values, [i for i in range(0, len(ind), 1)]))
    dict_target = dict(zip(ind.values, [i for i in range(len(ind), len(ind) * 2, 1)]))

    sankey_metal = (
        df.loc["Domestic Extraction Used - Fossil Fuel: Total"]
        .groupby(level=[0, 1])
        .sum()
    )
    sankey_metal = (
        sankey_metal.unstack()
        .rename(index=dict_source, columns=dict_target)
        .stack()
        .reset_index()
    )
    sankey_metal.columns = ["source", "target", "value"]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=ind.append(ind),
                    color="blue",
                ),
                link=dict(
                    source=sankey_metal["source"],
                    target=sankey_metal["target"],
                    value=sankey_metal["value"],
                ),
            )
        ]
    )

    fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
    fig.show()


def sankey_non_ferous():
    i = 2013
    df = pd.read_csv("imports_imp" + str(i) + ".csv", index_col=[2, 0, 1])
    continent = pd.read_excel("continent.xlsx", index_col=[0, 1])
    df = df.rename(index=dict(continent.index))
    cmap = sns.color_palette("colorblind", as_cmap="True")

    ind = (
        df.loc["Domestic Extraction Used - Non-ferous metal ores"]
        .groupby(level=[0, 1])
        .sum()
        .unstack()
        .index
    )
    dict_source = dict(zip(ind.values, [i for i in range(0, len(ind), 1)]))
    dict_target = dict(zip(ind.values, [i for i in range(len(ind), len(ind) * 2, 1)]))

    sankey_metal = (
        df.loc["Domestic Extraction Used - Non-ferous metal ores"]
        .groupby(level=[0, 1])
        .sum()
    )
    sankey_metal = (
        sankey_metal.unstack()
        .rename(index=dict_source, columns=dict_target)
        .stack()
        .reset_index()
    )
    sankey_metal.columns = ["source", "target", "value"]

    label = []
    for i in ind:
        label.append(
            i
            + " ("
            + str(
                int(
                    df.loc["Domestic Extraction Used - Non-ferous metal ores"]
                    .groupby(level="region prod")
                    .sum()
                    .loc[i]
                    .values
                    / 1000
                )
            )
            + " Mt)"
        )
    for i in ind:
        label.append(
            i
            + " ("
            + str(
                int(
                    df.loc["Domestic Extraction Used - Non-ferous metal ores"]
                    .groupby(level="region cons")
                    .sum()
                    .loc[i]
                    .values
                    / 1000
                )
            )
            + " Mt)"
        )
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=label,
                    # ind.append(ind),
                    color="lightgray",
                ),
                link=dict(
                    source=sankey_metal["source"],
                    target=sankey_metal["target"],
                    value=sankey_metal["value"],
                    color=[cmap[i] for i in sankey_metal["source"]],
                ),
                valueformat=".0f",
            )
        ]
    )

    fig.show()
    fig.write_image("figures/sankey non ferous.svg", engine="orca")


sankey_non_ferous()


def sankey_iron():
    i = 2013
    df = pd.read_csv("imports_imp" + str(i) + ".csv", index_col=[2, 0, 1])
    continent = pd.read_excel("continent.xlsx", index_col=[0, 1])
    df = df.rename(index=dict(continent.index))
    cmap = sns.color_palette("colorblind", as_cmap="True")

    ind = (
        df.loc["Domestic Extraction Used - Iron Ore"]
        .groupby(level=[0, 1])
        .sum()
        .unstack()
        .index
    )
    dict_source = dict(zip(ind.values, [i for i in range(0, len(ind), 1)]))
    dict_target = dict(zip(ind.values, [i for i in range(len(ind), len(ind) * 2, 1)]))

    sankey_metal = (
        df.loc["Domestic Extraction Used - Iron Ore"].groupby(level=[0, 1]).sum()
    )
    sankey_metal = (
        sankey_metal.unstack()
        .rename(index=dict_source, columns=dict_target)
        .stack()
        .reset_index()
    )
    sankey_metal.columns = ["source", "target", "value"]

    label = []
    for i in ind:
        label.append(
            i
            + " ("
            + str(
                int(
                    df.loc["Domestic Extraction Used - Iron Ore"]
                    .groupby(level="region prod")
                    .sum()
                    .loc[i]
                    .values
                    / 1000
                )
            )
            + " Mt)"
        )
    for i in ind:
        label.append(
            i
            + " ("
            + str(
                int(
                    df.loc["Domestic Extraction Used - Iron Ore"]
                    .groupby(level="region cons")
                    .sum()
                    .loc[i]
                    .values
                    / 1000
                )
            )
            + " Mt)"
        )
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=label,
                    # ind.append(ind),
                    color="lightgray",
                ),
                link=dict(
                    source=sankey_metal["source"],
                    target=sankey_metal["target"],
                    value=sankey_metal["value"],
                    color=[cmap[i] for i in sankey_metal["source"]],
                ),
                valueformat=".0f",
            )
        ]
    )

    fig.show()
    fig.write_image("figures/sankey iron.svg", engine="orca")


sankey_iron()


def sankey_minerals():
    ext = "Domestic Extraction Used - Non-metalic Minerals"
    i = 2013
    df = pd.read_csv("imports_imp" + str(i) + ".csv", index_col=[2, 0, 1])
    continent = pd.read_excel("continent.xlsx", index_col=[0, 1])
    df = df.rename(index=dict(continent.index))
    cmap = sns.color_palette("colorblind", as_cmap="True")

    ind = df.loc[ext].groupby(level=[0, 1]).sum().unstack().index
    dict_source = dict(zip(ind.values, [i for i in range(0, len(ind), 1)]))
    dict_target = dict(zip(ind.values, [i for i in range(len(ind), len(ind) * 2, 1)]))

    sankey_metal = df.loc[ext].groupby(level=[0, 1]).sum()
    sankey_metal = (
        sankey_metal.unstack()
        .rename(index=dict_source, columns=dict_target)
        .stack()
        .reset_index()
    )
    sankey_metal.columns = ["source", "target", "value"]

    label = []
    for i in ind:
        label.append(
            i
            + " ("
            + str(
                int(df.loc[ext].groupby(level="region prod").sum().loc[i].values / 1000)
            )
            + " Mt)"
        )
    for i in ind:
        label.append(
            i
            + " ("
            + str(
                int(df.loc[ext].groupby(level="region cons").sum().loc[i].values / 1000)
            )
            + " Mt)"
        )
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=label,
                    # ind.append(ind),
                    color="lightgray",
                ),
                link=dict(
                    source=sankey_metal["source"],
                    target=sankey_metal["target"],
                    value=sankey_metal["value"],
                    color=[cmap[i] for i in sankey_metal["source"]],
                ),
                valueformat=".0f",
            )
        ]
    )

    fig.show()
    fig.write_image("figures/sankey minerals.svg", engine="orca")


sankey_minerals()


def sankey_fossil():
    ext = "Domestic Extraction Used - Fossil Fuel: Total"
    i = 2013
    df = pd.read_csv("imports_sat" + str(i) + ".csv", index_col=[2, 0, 1])
    continent = pd.read_excel("continent.xlsx", index_col=[0, 1])
    df = df.rename(index=dict(continent.index))
    cmap = sns.color_palette("colorblind", as_cmap="True")

    ind = df.loc[ext].groupby(level=[0, 1]).sum().unstack().index
    dict_source = dict(zip(ind.values, [i for i in range(0, len(ind), 1)]))
    dict_target = dict(zip(ind.values, [i for i in range(len(ind), len(ind) * 2, 1)]))

    sankey_metal = df.loc[ext].groupby(level=[0, 1]).sum()
    sankey_metal = (
        sankey_metal.unstack()
        .rename(index=dict_source, columns=dict_target)
        .stack()
        .reset_index()
    )
    sankey_metal.columns = ["source", "target", "value"]

    label = []
    for i in ind:
        label.append(
            i
            + " ("
            + str(
                int(df.loc[ext].groupby(level="region prod").sum().loc[i].values / 1000)
            )
            + " Mt)"
        )
    for i in ind:
        label.append(
            i
            + " ("
            + str(
                int(df.loc[ext].groupby(level="region cons").sum().loc[i].values / 1000)
            )
            + " Mt)"
        )
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=label,
                    # ind.append(ind),
                    color="lightgray",
                ),
                link=dict(
                    source=sankey_metal["source"],
                    target=sankey_metal["target"],
                    value=sankey_metal["value"],
                    color=[cmap[i] for i in sankey_metal["source"]],
                ),
                valueformat=".0f",
            )
        ]
    )

    fig.show()
    fig.write_image("figures/sankey fossil.svg", engine="orca")


sankey_fossil()


def pie():

    continent = pd.read_excel("continent.xlsx", index_col=[0, 1])
    extensions = [
        "Domestic Extraction Used - Non-metalic Minerals",
        "Domestic Extraction Used - Iron Ore",
        "Domestic Extraction Used - Non-ferous metal ores",
        "Domestic Extraction Used - Fossil Fuel: Total",
    ]
    names = [
        "non-metallic minerals",
        "iron ore",
        "non-ferous metal ores",
        "fossil fuels",
    ]
    startangle = [135, 0, -45, 135]
    cmap = sns.color_palette("colorblind", as_cmap="True")

    df = pd.read_csv("imports_imp2013.csv", index_col=[2, 0, 1])
    df = df.rename(index=dict(continent.index))
    for j in [0, 1, 2]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        ext = extensions[j]
        df_ext = df.loc[ext].div(df.loc[ext].sum())
        df_ext.groupby(level="region prod").sum()["2013"].plot(
            kind="pie",
            title="Production " + names[j],
            ax=axes[0],
            autopct="%1.1f%%",
            startangle=startangle[j],
            colors=cmap,
        )
        df_ext.groupby(level="region cons").sum()["2013"].plot(
            kind="pie",
            title="Consumption " + names[j],
            ax=axes[1],
            autopct="%1.1f%%",
            startangle=startangle[j],
            colors=cmap,
        )
        axes[0].set_ylabel("")
        axes[1].set_ylabel("")

        print(
            df_ext.groupby(level="region prod")
            .sum()["2013"]
            .loc[["Africa", "Rest of Asia", "Middle East", "Latin America", "India"]]
            .sum()
        )
        print(
            df_ext.groupby(level="region cons")
            .sum()["2013"]
            .loc[["Africa", "Rest of Asia", "Middle East", "Latin America", "India"]]
            .sum()
        )
        plt.tight_layout()
        plt.savefig("figures/pie " + names[j] + ".svg")

    dfi = pd.read_csv("imports_sat2013.csv", index_col=[2, 0, 1])
    df = dfi.rename(index=dict(continent.index))
    for j in [3]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        ext = extensions[j]
        df_ext = df.loc[ext].div(df.loc[ext].sum())
        df_ext.groupby(level="region prod").sum()["2013"].plot(
            kind="pie",
            title="Production " + names[j],
            ax=axes[0],
            autopct="%1.1f%%",
            startangle=startangle[j],
            colors=cmap,
        )
        df_ext.groupby(level="region cons").sum()["2013"].plot(
            kind="pie",
            title="Consumption " + names[j],
            ax=axes[1],
            autopct="%1.1f%%",
            startangle=startangle[j],
            colors=cmap,
        )
        axes[0].set_ylabel("")
        axes[1].set_ylabel("")

        print(
            df_ext.groupby(level="region prod")
            .sum()["2013"]
            .loc[["Africa", "Rest of Asia", "Middle East", "Latin America", "India"]]
            .sum()
        )
        print(
            df_ext.groupby(level="region cons")
            .sum()["2013"]
            .loc[["Africa", "Rest of Asia", "Middle East", "Latin America", "India"]]
            .sum()
        )
        plt.tight_layout()
        plt.savefig("figures/pie " + names[j] + ".svg")


# pie()


def graph1():
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))

    cmap = sns.color_palette("colorblind", as_cmap="True")

    df = (
        pd.concat(
            [
                impacts["Domestic Extraction Used - Non-metalic Minerals"],
                impacts[
                    [
                        "Domestic Extraction Used - Iron Ore",
                        "Domestic Extraction Used - Non-ferous metal ores",
                    ]
                ].sum(axis=1),
                satellite["Domestic Extraction Used - Fossil Fuel: Total"],
            ],
            keys=["Non-metalic minerals", "Metal ores", "Fossil fuel"],
            axis=1,
        )
        .swaplevel()
        .swaplevel()
    )
    total = pd.concat(
        [
            total_imp.loc["Domestic Extraction Used - Non-metalic Minerals"],
            total_imp.loc[
                [
                    "Domestic Extraction Used - Iron Ore",
                    "Domestic Extraction Used - Non-ferous metal ores",
                ]
            ].sum(),
            total_sat.loc["Domestic Extraction Used - Fossil Fuel: Total"],
        ],
        keys=["Non-metalic minerals", "Metal ores", "Fossil fuel"],
        axis=1,
    )
    continent = pd.read_excel("continent.xlsx", index_col=[0, 1])
    df.index.names = ["region", "year"]
    df_agg = df.rename(index=dict(continent.index)).groupby(level=df.index.names).sum()
    pop_agg = (
        pop.rename(index=dict(continent.index)).groupby(level="region").sum().stack()
    )
    df_agg_pop = df_agg.div(pop_agg, axis=0)
    j = 0
    share = pd.DataFrame()
    rolled = pd.DataFrame()  # only to have numeric values for paper
    rolled_pop = pd.DataFrame()
    for ext in df.columns:

        df_agg_rolled = (
            df_agg[ext].unstack().T.drop([2014, 2015]).rolling(1, center=True).mean()
        ) / 1000000
        df_agg_rolled["USA and Canada"] = (
            (df_agg[ext].unstack().T.drop([2014, 2015]).rolling(3, center=True).mean())
            / 1000000
        )["USA and Canada"]
        df_agg_rolled.loc[[1995, 2013]] = (
            df_agg[ext].unstack().T.loc[[1995, 2013]] / 1000000
        )
        if ext == "Non-metalic minerals":
            for i in range(1995, 2011, 1):
                df_agg_rolled.loc[i]["India"] = np.nan
        df_agg_rolled.plot.area(ax=axes[j, 0], color=cmap)
        ax2 = axes[j, 0].twinx()
        df_agg_rolled.sum(axis=1).div(total[ext].values / 1000000 / 100).plot(
            ax=ax2, color="black", ls="dashed", label="Share of world\nfootprint"
        )
        ax2.legend(fontsize=7, framealpha=0, loc=1)
        share[ext] = df_agg_rolled.sum(axis=1).div(total[ext].values / 1000000 / 100)
        rolled[ext] = df_agg_rolled.stack()  # for num data
        ax2.set_ylim([0, 8.2])
        ax2.set_ylabel("Share of world footprint (\%)")
        axes[j, 0].set_ylabel(ext + " (Gt)")
        axes[j, 0].legend(fontsize=7, framealpha=0, ncol=2)

        df_agg_pop_rolled = (
            df_agg_pop[ext]
            .unstack()
            .T.drop([2014, 2015])
            .rolling(1, center=True)
            .mean()
        )
        df_agg_pop_rolled["USA and Canada"] = (
            df_agg_pop[ext]
            .unstack()
            .T.drop([2014, 2015])
            .rolling(3, center=True)
            .mean()
        )["USA and Canada"]
        df_agg_pop_rolled.loc[[1995, 2013]] = (
            df_agg_pop[ext].unstack().T.loc[[1995, 2013]]
        )
        # if ext == "Non-metalic minerals":
        #     for i in range(1995, 2011, 1):
        #         df_agg_pop_rolled.loc[i]["India"] = np.nan
        df_agg_pop_rolled.plot(ax=axes[j, 1], color=cmap)
        rolled_pop[ext] = df_agg_pop_rolled.stack()
        axes[j, 1].set_ylabel(ext + " (t/capita)")
        axes[j, 1].legend(fontsize=7, ncol=2, framealpha=0, loc=2)

        j += 1

        axes[0, 0].set_ylim(top=3)
        axes[1, 0].set_ylim(top=0.45)
        axes[2, 0].set_ylim(top=0.75)

        axes[0, 1].set_ylim(top=2.3)
        axes[1, 1].set_ylim(top=0.9)
        axes[2, 1].set_ylim(top=0.82)

        axes[0, 0].set_title("a", loc="left")
        axes[0, 1].set_title("b", loc="left")
        axes[1, 0].set_title("c", loc="left")
        axes[1, 1].set_title("d", loc="left")
        axes[2, 0].set_title("e", loc="left")
        axes[2, 1].set_title("f", loc="left")

        plt.tight_layout()
        plt.savefig("figures/fig1.pdf")
        plt.savefig("figures/fig1.png", bbox="tight")
        plt.savefig("figures/fig1.svg", bbox="tight")

    return rolled, rolled_pop, total, share


rolled, rolled_pop, total, share = graph1()


def num_data_1():
    rolled.loc[2013].sum() / rolled.loc[1995].sum()
    rolled.loc[2013].sum()
    rolled["Non-metalic minerals"].unstack().sum(axis=1)
    rolled["Non-metalic minerals"].unstack()["China"] / rolled[
        "Non-metalic minerals"
    ].unstack().sum(axis=1)
    rolled_pop.swaplevel().loc["Australia"] / rolled_pop.swaplevel().loc["Africa"]
    rolled.loc[2013].sort_values(by="Non-metalic minerals")


def graph2():
    fig, axes = plt.subplots(1, figsize=(8, 8))

    col = pd.read_excel("continent.xlsx", index_col=[0])
    cmap = sns.color_palette("colorblind", as_cmap="True")
    dict_color = dict(
        zip(col["continent"].unique(), cmap[0 : len(col["continent"].unique())])
    )
    col["color"] = col["continent"].replace(dict_color)
    handles = []
    for i in dict_color.keys():
        handles.append(
            mlines.Line2D(
                [],
                [],
                linestyle="None",
                marker=".",
                markersize=10,
                label=i,
                color=dict_color[i],
            )
        )
    handles.append(
        mlines.Line2D([], [], color="black", markersize=80, label="Regression 2015")
    )

    col_line = pd.read_excel("continent.xlsx", index_col=[0])
    sns.color_palette("pastel", as_cmap="True").append("olive")
    cmap = sns.color_palette("pastel", as_cmap="True")
    dict_color = dict(
        zip(
            col_line["continent"].unique(),
            cmap[0 : len(col_line["continent"].unique())],
        )
    )
    col_line["color"] = col_line["continent"].replace(dict_color)

    years = [1995, 2000, 2005, 2010, 2015]
    x = HAQ_agg[years]
    y_pri = satellite_cap["Energy Carrier Net Total"].unstack()
    y_loss = satellite_cap["Energy Carrier Net LOSS"].unstack()
    y = (y_pri - y_loss).rolling(3, center=True, axis=1).mean()[years]
    y[[1995, 2015]] = (y_pri - y_loss)[[1995, 2015]]
    regression = scipy.stats.linregress(x[2015], np.log(y[2015]))
    x_lin = np.linspace(x[2015].min(), x[2015].max(), 100)
    axes.plot(x_lin, np.exp(x_lin * regression[0] + regression[1]), color="black")
    print(str(round(regression[2] ** 2, 2)))

    names = pd.read_excel("region names.xlsx", index_col=None)
    dict_names = dict(zip(names["region"], names["full name"]))

    annotations_fig2 = pd.read_excel("region names.xlsx", index_col=[0])

    for reg in pop[2015].sort_values(ascending=False).index.drop("LU"):
        axes.plot(
            x.loc[reg],
            y.loc[reg],
            label=reg,
            color=col_line.loc[reg].loc["color"],
            zorder=1,
            linewidth=1,
        )
        axes.scatter(
            x.loc[reg].loc[2015],
            y.loc[reg].loc[2015],
            label=reg,
            s=pop.loc[reg].loc[2015] / 1000,
            color=col.loc[reg].loc["color"],
            zorder=2,
            edgecolor="black",
            linewidths=0.5,
        )
        axes.annotate(
            annotations_fig2.loc[reg].loc["full name"],
            (
                x.loc[reg].loc[2015] + annotations_fig2.loc[reg].loc["x axis"],
                y.loc[reg].loc[2015] + annotations_fig2.loc[reg].loc["y axis"],
            ),
        )

    axes.set_xlabel("Healthcare Access and Quality Index")
    axes.set_ylabel("Energy footprint (GJ/capita)")
    axes.set_xlim(right=100.5)
    axes.set_ylim(top=16.5)

    axes.legend(handles=handles, fontsize=10, ncol=2, framealpha=0)

    plt.tight_layout()
    plt.savefig("figures/fig2.pdf")
    plt.savefig("figures/fig2.png", bbox="tight")
    plt.savefig("figures/fig2.svg", bbox="tight")


graph2()


def num_data2():
    (HAQ_agg[2015] / HAQ_agg[1995]).sort_values()


def graph3():
    fig, axes = plt.subplots(1, figsize=(8, 8))

    cmap = sns.color_palette("colorblind", as_cmap="True")

    y_pri = satellite_cap["Energy Carrier Net Total"].unstack()
    y_loss = satellite_cap["Energy Carrier Net LOSS"].unstack()
    y = y_pri - y_loss

    df = pd.concat(
        [y[2015] for i in range(44, 101, 1)],
        keys=[i for i in range(44, 101, 1)],
        axis=1,
    )

    x = HAQ_agg[2015]
    regression = scipy.stats.linregress(x, np.log(y[2015]))
    for ind in range(44, 101, 1):
        for reg in HAQ_agg.index:
            if df.loc[reg].loc[ind] < ind:
                if df[ind].loc[reg] < np.exp(regression[0] * ind + regression[1]):
                    df[ind].loc[reg] = np.exp(regression[0] * ind + regression[1])

    continent = pd.read_excel("continent.xlsx", index_col=[0, 1])
    df_agg = (
        df.mul(pop[2015], axis=0)
        .rename(index=dict(continent.index))
        .groupby(level=0)
        .sum()
        / 1000000
    )
    df_agg.T.loc[[44, 50, 60, 70, 80, 90, 100]].plot.bar(
        stacked="True", width=0.5, ax=axes, color=cmap
    )
    axes.plot(
        [-0.25, 7.25],
        [df_agg[44].sum(), df_agg[44].sum()],
        color="gray",
        linestyle="dashed",
    )

    i = 1
    for ind in [50, 60, 70, 80, 90, 100]:
        axes.annotate(
            "",
            xy=(i + 0.3, df_agg[44].sum()),
            xytext=(i + 0.3, df_agg[ind].sum()),
            arrowprops=dict(arrowstyle="<->", color="black"),
        )
        axes.text(
            x=i - 0.3,
            y=df_agg[ind].sum() + 0.55,
            s="+ "
            + str(round(df_agg[ind].sum() / df_agg[44].sum() * 100 - 100, 2))
            + " \%",
        )
        i += 1

    axes.legend(fontsize=10)
    axes.set_xlabel("World minimum Healthcare Access and Quality Index")
    axes.set_ylabel("Energy required (EJ)")

    plt.tight_layout()
    plt.savefig("figures/fig3.pdf")
    plt.savefig("figures/fig3.png", bbox="tight")
    plt.savefig("figures/fig3.svg")


graph3()
# rajouter des pourcentages sur les barres ou alors la population à côté pour comparer


def num_data3():
    pop_agg.loc[["IN", "WF", "ID", "ZA", "WA"]].sum() / pop_agg.sum()


def graph4():

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    years = [i for i in range(2002, 2016, 1)]

    col = pd.read_excel("continent.xlsx", index_col=[0])
    cmap = sns.color_palette("colorblind", as_cmap="True")
    dict_color = dict(
        zip(col["continent"].unique(), cmap[0 : len(col["continent"].unique())])
    )
    col["color"] = col["continent"].replace(dict_color)

    def adjust_lightness(color, amount=0.5):
        import matplotlib.colors as mc
        import colorsys

        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

    exp = (
        Y_health_NHA_euros.sum(axis=1)
        .loc[constant_ppp.unstack().index]
        .unstack()
        .div(XReuros.loc[constant_ppp.columns], axis=0)
        .unstack()
    )
    y = (
        (satellite["Energy Carrier Net Total"] - satellite["Energy Carrier Net LOSS"])
        .unstack()
        .drop(1995, axis=1)
        .stack()
    ).loc[constant_ppp.unstack().swaplevel().index] / exp.swaplevel()
    x = (
        exp
        / pop.stack().loc[constant_ppp.unstack().swaplevel().index].swaplevel()
        * 1000
    )
    for reg in y.unstack().index:
        for year in range(2002, 2016, 1):
            axes[0, 0].scatter(
                np.log(x.loc[reg].loc[year]),
                y.loc[reg].loc[year],
                s=pop.loc[reg].loc[year] / 6000,
                # color=str(1 - (year - 2002) / 14),
                color=adjust_lightness(
                    col.loc[reg].loc["color"], 1.5 - (year - 2002) / 14
                ),
            )
            axes[0, 1].scatter(
                year,
                y.loc[reg].loc[year],
                s=pop.loc[reg].loc[year] / 6000,
                # color=str(1 - (year - 2002) / 14),
                color=adjust_lightness(
                    col.loc[reg].loc["color"], 1.5 - (year - 2002) / 14
                ),
            )

    y = (
        (
            (
                satellite["Energy Carrier Net Total"]
                - satellite["Energy Carrier Net LOSS"]
            )
            .unstack()
            .drop(1995, axis=1)
            .stack()
        ).loc[constant_ppp.unstack().swaplevel().index]
        / constant_ppp.unstack().swaplevel()
        * 1000
    )
    x = (
        constant_ppp.unstack().swaplevel()
        / pop.stack().loc[constant_ppp.unstack().swaplevel().index]
    )
    k = 0

    for reg in y.unstack().index:
        for year in years:
            axes[1, 0].scatter(
                np.log(x.loc[reg].loc[year]),
                y.loc[reg].loc[year],
                s=pop.loc[reg].loc[year] / 6000,
                # color=str(1 - (year - 2002) / 14),
                color=adjust_lightness(
                    col.loc[reg].loc["color"], 1.5 - (year - 2002) / 14
                ),
            )
            axes[1, 1].scatter(
                year,
                y.loc[reg].loc[year],
                s=pop.loc[reg].loc[year] / 6000,
                # color=str(k / 49),
                # color=str(1 - (year - 2002) / 14),
                color=adjust_lightness(
                    col.loc[reg].loc["color"], 1.5 - (year - 2002) / 14
                ),
            )
        k += 1

    y_world_0 = (
        (satellite["Energy Carrier Net Total"] - satellite["Energy Carrier Net LOSS"])
        .unstack()
        .drop(1995, axis=1)
    )[years].sum() / exp.unstack().sum()
    x_world_0 = np.log(
        exp.unstack().sum() / pop.sum().loc[exp.unstack().sum().index] * 1000
    )
    axes[0, 0].plot(x_world_0, y_world_0, color="black", zorder=2)
    axes[0, 1].plot(y_world_0.index, y_world_0, color="black", zorder=2)

    y_world_1 = (
        (
            (
                satellite["Energy Carrier Net Total"]
                - satellite["Energy Carrier Net LOSS"]
            )
            .unstack()
            .drop(1995, axis=1)
        )[years].sum()
        / constant_ppp.sum()
        * 1000
    )
    x_world_1 = np.log(constant_ppp.sum() / pop.sum().loc[constant_ppp.sum().index])
    axes[1, 0].plot(x_world_1, y_world_1, color="black", zorder=2)
    axes[1, 1].plot(y_world_1.index, y_world_1, color="black", zorder=2)

    axes[0, 0].set_ylim(top=16)
    axes[1, 0].set_ylim(top=16)
    axes[0, 1].set_ylim(top=16)
    axes[1, 1].set_ylim(top=16)

    axes[1, 0].set_xlim([2.8, 9.5])
    axes[0, 0].set_xlim([2.8, 9.5])

    # axes[1, 1].set_xticks([2002, 2005, 2008, 2010, 2012, 2014, 2015])

    axes[0, 0].set_ylabel("Energy intensity (MJ/USdol)")  # $, essayer en svg
    axes[0, 1].set_ylabel("Energy intensity (MJ/USdol)")
    axes[1, 0].set_ylabel("Energy intensity (MJ/USdolppp2015)")
    axes[1, 1].set_ylabel("Energy intensity (MJ/USdolppp2015)")
    axes[0, 0].set_xlabel("Health expenditures (USdol)")
    axes[1, 0].set_xlabel("Health expenditures (USdolppp2015)")
    axes[0, 1].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")

    handles = []
    for i in dict_color.keys():
        handles.append(
            mlines.Line2D(
                [],
                [],
                linestyle="None",
                marker=".",
                markersize=10,
                label=i,
                color=dict_color[i],
            )
        )
    handles.append(
        mlines.Line2D([], [], color="black", markersize=80, label="World mean")
    )

    axes[1, 0].legend(handles=handles, fontsize=10, ncol=2, framealpha=0, loc=2)
    axes[1, 1].legend(handles=handles, fontsize=10, ncol=2, framealpha=0, loc=2)

    axes[0, 0].set_title("a", loc="left")
    axes[0, 1].set_title("b", loc="left")
    axes[1, 0].set_title("c", loc="left")
    axes[1, 1].set_title("d", loc="left")

    plt.tight_layout()
    plt.savefig("figures/fig4.pdf")
    plt.savefig("figures/fig4.png", bbox="tight")
    plt.savefig("figures/fig4.svg")


graph4()


#####################


# Numeric data in paper
total = pd.concat(
    [
        total_imp.loc["Domestic Extraction Used - Non-metalic Minerals"],
        total_imp.loc[
            [
                "Domestic Extraction Used - Iron Ore",
                "Domestic Extraction Used - Non-ferous metal ores",
            ]
        ].sum(),
        total_sat.loc["Domestic Extraction Used - Fossil Fuel: Total"],
    ],
    keys=["Non-metalic minerals", "Metal ores", "Fossil fuel"],
    axis=1,
)

df = pd.concat(
    [
        impacts["Domestic Extraction Used - Non-metalic Minerals"],
        impacts[
            [
                "Domestic Extraction Used - Iron Ore",
                "Domestic Extraction Used - Non-ferous metal ores",
            ]
        ].sum(axis=1),
        satellite["Domestic Extraction Used - Fossil Fuel: Total"],
    ],
    keys=["Non-metalic minerals", "Metal ores", "Fossil fuel"],
    axis=1,
)

world_share = pd.DataFrame(
    df.groupby(level=1).sum().drop([2014, 2015]).stack().values / total.stack().values,
    index=df.groupby(level=1).sum().drop([2014, 2015]).stack().index,
)[0].unstack()


# .....SI


# ...................VALIDATION....................

# check if NHA ppp is indeed current ppp
# (expendituresNHA * 1000 / current_ppp).drop(list(range(1995, 2000, 1)), axis=1).loc[
#     "US"
# ]

# utiliser ICP surestime les différences, current PPP = 70% de ce qui est calculé avec PPP 2017
constant_ppp.mul(XReuros.drop([1995, 1996, 1997, 1998, 1999, 2018]), axis=1)[
    2017
].sum() / 1000 / Y_health_NHA_ppp17.loc[2017].sum()
# ici on compare directement les impacts et on confirme ça:
index.loc[2017] / (expendituresNHA[2017] * 1000 / current_ppp[2017])
