from specific_functions import *
import plotly.offline as pyo
import plotly.io as pio
import seaborn as sns
import plotly.graph_objs as go
import matplotlib.lines as mlines


###
pio.orca.config.executable = "C:/Users/andrieba/anaconda3/pkgs/plotly-orca-1.2.1-1/orca_app/orca.exe"
pio.orca.config.save()
# pio.kaleido.scope.mathjax = None  # ne sert à rien car mathjax ne fonctionne pas
pyo.init_notebook_mode()

### Data needed for the graphs

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

SLkYhealth_imp = feather.read_feather("results/SLkYhealth_imp.feather")
SLkYhealth_sat = feather.read_feather("results/SLkYhealth_sat.feather")

SLkYhealth_imp.columns = SLkYhealth_imp.columns.astype(int)
SLkYhealth_sat.columns = SLkYhealth_sat.columns.astype(int)

impacts = SLkYhealth_imp.stack().unstack(level=1).sum(axis=1).unstack(level=1)[impacts_ext]
impacts_world = impacts.unstack().sum().unstack()

satellite = SLkYhealth_sat.stack().unstack(level=1).sum(axis=1).unstack(level=1)[satellite_ext]
satellite_world = satellite.unstack(level=0).sum().unstack(level=0)

satellite_cap = satellite.div(pd.DataFrame(pop.stack(), index=satellite.index)[0], axis=0)
impacts_cap = impacts.div(pd.DataFrame(pop.stack(), index=impacts.index)[0], axis=0)

total_sat = pd.read_csv("results/total_sat.csv", index_col=0).loc[satellite_ext]
total_imp = pd.read_csv("results/total_imp.csv", index_col=0).loc[impacts_ext]


### data for panel data analysis
y_pri = satellite_cap["Energy Carrier Net Total"].unstack()
y_loss = satellite_cap["Energy Carrier Net LOSS"].unstack()
years = [1995, 2000, 2005, 2010, 2015]
x = HAQ_agg[years]
panel = pd.concat([x.stack(), (y_pri - y_loss).stack().loc[x.stack().index]], keys=["x", "y"], axis=1)
panel = panel.reset_index()
panel.to_excel("results/data_for_panel.xlsx")


### Graphs in the paper


def fig1():
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
    continent = pd.read_excel("Data/continent.xlsx", index_col=[0, 1])
    df.index.names = ["region", "year"]
    df_agg = df.rename(index=dict(continent.index)).groupby(level=df.index.names).sum()
    pop_agg = pop.rename(index=dict(continent.index)).groupby(level="Country Name").sum().stack()
    pop_agg.index.names = ["region", "year"]
    df_agg_pop = df_agg.div(pop_agg, axis=0)
    j = 0
    share = pd.DataFrame()
    rolled = pd.DataFrame()  # only to have numeric values for paper
    rolled_pop = pd.DataFrame()
    yaxisnames = [
        "Share of global non-metallic mineral use (\%)",
        "Share of global metal ore use (\%)",
        "Share of global fossil fuel use (\%)",
    ]
    for ext in df.columns:
        df_agg_rolled = (df_agg[ext].unstack().T.drop([2014, 2015]).rolling(1, center=True).mean()) / 1000000
        df_agg_rolled["USA and Canada"] = (
            (df_agg[ext].unstack().T.drop([2014, 2015]).rolling(3, center=True).mean()) / 1000000
        )["USA and Canada"]
        df_agg_rolled.loc[[1995, 2013]] = df_agg[ext].unstack().T.loc[[1995, 2013]] / 1000000

        df_agg_rolled.plot.area(ax=axes[j, 0], color=cmap)
        ax2 = axes[j, 0].twinx()
        df_agg_rolled.sum(axis=1).div(total[ext].values / 1000000 / 100).plot(
            ax=ax2, color="black", ls="dashed", label="Share of world\nfootprint"
        )
        ax2.legend(fontsize=7, framealpha=0, loc=1)
        share[ext] = df_agg_rolled.sum(axis=1).div(total[ext].values / 1000000 / 100)
        rolled[ext] = df_agg_rolled.stack()  # for num data
        ax2.set_ylim([0, 8.3])

        axes[j, 0].set_ylabel(ext + " (Gt)")
        axes[j, 0].legend(fontsize=7, framealpha=0, ncol=2)

        df_agg_pop_rolled = df_agg_pop[ext].unstack().T.drop([2014, 2015]).rolling(1, center=True).mean()
        df_agg_pop_rolled["USA and Canada"] = (
            df_agg_pop[ext].unstack().T.drop([2014, 2015]).rolling(3, center=True).mean()
        )["USA and Canada"]
        df_agg_pop_rolled.loc[[1995, 2013]] = df_agg_pop[ext].unstack().T.loc[[1995, 2013]]

        df_agg_pop_rolled.plot(ax=axes[j, 1], color=cmap)
        rolled_pop[ext] = df_agg_pop_rolled.stack()
        axes[j, 1].set_ylabel(ext + " (t/capita)")
        axes[j, 1].legend(fontsize=7, ncol=2, framealpha=0, loc=2)

        ax2.set_ylabel(yaxisnames[j])

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
    rolled_pop.unstack(level=0).stack(level=0).swaplevel().sort_index().to_excel("figures/fig1_cap.xlsx")
    return rolled, rolled_pop, total, share


def fig3():
    fig, axes = plt.subplots(1, figsize=(8, 8))

    col = pd.read_excel("Data/continent.xlsx", index_col=[0])
    cmap = sns.color_palette("colorblind", as_cmap="True")
    dict_color = dict(zip(col["continent"].unique(), cmap[0 : len(col["continent"].unique())]))
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
    handles.append(mlines.Line2D([], [], color="black", markersize=80, label="Region FE regression"))

    col_line = pd.read_excel("Data/continent.xlsx", index_col=[0])
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

    y = y_pri - y_loss
    y.loc[["CA", "CH", "FI", "MT", "NO", "NL", "SI", "SK", "US", "TW"]] = (
        (y_pri - y_loss)
        .rolling(3, center=True, axis=1)
        .mean()
        .loc[["CA", "CH", "FI", "MT", "NO", "NL", "SI", "SK", "US", "TW"]]
    )
    y = y[[2000, 2005, 2010]]
    y[[1995, 2015]] = (y_pri - y_loss)[[1995, 2015]]
    y = y.sort_index(axis=1)

    x_lin = np.linspace(x.stack().min(), x.stack().max(), 100)
    axes.plot(x_lin, np.exp(-2.7350 + 0.0513 * x_lin), color="black")

    names = pd.read_excel("Data/region names.xlsx", index_col=None)
    dict_names = dict(zip(names["region"], names["full name"]))

    annotations_fig2 = pd.read_excel("Data/region names.xlsx", index_col=[0])

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
    plt.savefig("figures/fig3.pdf")
    plt.savefig("figures/fig3.png", bbox="tight")
    plt.savefig("figures/fig3.svg", bbox="tight")


def fig4():
    i = 2015
    LkYhealth = feather.read_feather(pathexio + "Data/LkYhealth/LkYhealth" + str(i) + ".feather")
    pathIOT = pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/"

    S = pd.read_csv(pathIOT + "satellite/S.txt", delimiter="\t", header=[0, 1], index_col=[0])
    S = S.loc["Energy Carrier Net Total"] - S.loc["Energy Carrier Net LOSS"]
    SLkYhealth_i = LkYhealth.mul(S, axis=0).groupby(level="sector").sum()

    cmap = sns.color_palette("colorblind", as_cmap="True")
    conc = pd.read_excel("Data/concordance_products2.xlsx", index_col=[0, 1])
    regions = (
        (constant_ppp / pop)[2015].loc[HAQ_agg[2015].sort_values(ascending=False).head(10).index].sort_values().index
    )
    regions = regions.drop("LU")
    df = agg(SLkYhealth_i, conc, axis=0).groupby(level=0, axis=1).sum().div(pop[2015], axis=1)[regions].T
    df = df[df.loc["ES"].sort_values(ascending=False).index]

    df2 = agg(SLkYhealth_i, conc, axis=0).groupby(level=0, axis=1).sum().div(pop[2015], axis=1).T
    df2.to_excel("results/scopes.xlsx")
    (df2.div(df2.sum(axis=1), axis=0) * 100).round(1).to_excel("results/scopes_shares.xlsx")

    fig, axes = plt.subplots(2, figsize=(10, 10))
    ax1 = axes[0]
    ax2 = ax1.twinx()
    (constant_ppp / pop)[2015].loc[regions].reset_index().plot.scatter(
        x="region", y=2015, ax=ax2, color="black", marker="X", s=40
    )
    ax2.set_ylim([0, 7650])
    df.plot.bar(stacked=True, ax=ax1, color=cmap)
    ax1.legend(fontsize=12)

    k = 0
    for reg in regions:
        ax1.text(
            x=k - 0.28,
            y=df.sum(axis=1).loc[reg] + 0.2,
            s="HAQ=" + str(round(HAQ_agg[2015].loc[reg], 1)),
            fontsize=12,
            weight="bold",
        )
        k += 1

    df = df.div(df.sum(axis=1), axis=0) * 100
    df.plot.bar(stacked=True, ax=axes[1], color=cmap)
    axes[1].get_legend().remove()

    y_offset = -3.5
    for bar in axes[1].patches:
        if bar.get_height() > 3:
            axes[1].text(
                s=str(round(bar.get_height(), 1)) + " \%",
                x=bar.get_x() + bar.get_width() / 2,
                y=bar.get_height() + bar.get_y() + y_offset,
                ha="center",
                color="black",
                weight="bold",
                size=12,
            )
    # ax1.set_xlabel('')
    ax1.set_ylabel("Energy footprint (GJ/capita)", fontsize=12)
    ax2.set_ylabel("Health expenditures (USdol 2015 ppp/capita)", fontsize=12)
    # axes[1].set_xlabel('')
    axes[1].set_ylabel("Share of the energy footprint (\%)", fontsize=12)

    axes[0].set_title("a", loc="left", fontsize=15, weight="bold")
    axes[1].set_title("b", loc="left", fontsize=15, weight="bold")

    plt.tight_layout()
    plt.savefig("figures/fig4.pdf")
    plt.savefig("figures/fig4.png", bbox="tight")
    plt.savefig("figures/fig4.svg", bbox="tight")


def fig5():
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    years = [i for i in range(2002, 2016, 1)]

    col = pd.read_excel("Data/continent.xlsx", index_col=[0])
    cmap = sns.color_palette("colorblind", as_cmap="True")
    dict_color = dict(zip(col["continent"].unique(), cmap[0 : len(col["continent"].unique())]))
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
    x = exp / pop.stack().loc[exp.index] * 1000
    for reg in y.unstack().index:
        for year in range(2002, 2016, 1):
            axes[0, 0].scatter(
                x.loc[reg].loc[year],
                y.loc[reg].loc[year],
                s=pop.loc[reg].loc[year] / 6000,
                # color=str(1 - (year - 2002) / 14),
                color=adjust_lightness(col.loc[reg].loc["color"], 1.5 - (year - 2002) / 14),
            )
            axes[0, 0].set_xscale("log")
            axes[0, 1].scatter(
                year,
                y.loc[reg].loc[year],
                s=pop.loc[reg].loc[year] / 6000,
                # color=str(1 - (year - 2002) / 14),
                color=adjust_lightness(col.loc[reg].loc["color"], 1.5 - (year - 2002) / 14),
            )

    y = (
        (
            (satellite["Energy Carrier Net Total"] - satellite["Energy Carrier Net LOSS"])
            .unstack()
            .drop(1995, axis=1)
            .stack()
        ).loc[constant_ppp.unstack().swaplevel().index]
        / constant_ppp.unstack().swaplevel()
        * 1000
    )
    x = constant_ppp.unstack().swaplevel() / pop.stack().loc[constant_ppp.unstack().swaplevel().index]
    k = 0

    for reg in y.unstack().index:
        for year in years:
            axes[1, 0].scatter(
                x.loc[reg].loc[year],
                y.loc[reg].loc[year],
                s=pop.loc[reg].loc[year] / 6000,
                # color=str(1 - (year - 2002) / 14),
                color=adjust_lightness(col.loc[reg].loc["color"], 1.5 - (year - 2002) / 14),
            )
            axes[1, 0].set_xscale("log")
            axes[1, 1].scatter(
                year,
                y.loc[reg].loc[year],
                s=pop.loc[reg].loc[year] / 6000,
                # color=str(k / 49),
                # color=str(1 - (year - 2002) / 14),
                color=adjust_lightness(col.loc[reg].loc["color"], 1.5 - (year - 2002) / 14),
            )
        k += 1

    y_world_0 = (
        (satellite["Energy Carrier Net Total"] - satellite["Energy Carrier Net LOSS"]).unstack().drop(1995, axis=1)
    )[years].sum() / exp.unstack().sum()
    x_world_0 = exp.unstack().sum() / pop.sum().loc[exp.unstack().sum().index] * 1000
    axes[0, 0].plot(x_world_0, y_world_0, color="black", zorder=2)
    axes[0, 1].plot(y_world_0.index, y_world_0, color="black", zorder=2)

    y_world_1 = (
        ((satellite["Energy Carrier Net Total"] - satellite["Energy Carrier Net LOSS"]).unstack().drop(1995, axis=1))[
            years
        ].sum()
        / constant_ppp.sum()
        * 1000
    )
    x_world_1 = constant_ppp.sum() / pop.sum().loc[constant_ppp.sum().index]
    axes[1, 0].plot(x_world_1, y_world_1, color="black", zorder=2)
    axes[1, 1].plot(y_world_1.index, y_world_1, color="black", zorder=2)

    axes[0, 0].set_ylim(top=16)
    axes[1, 0].set_ylim(top=16)
    axes[0, 1].set_ylim(top=16)
    axes[1, 1].set_ylim(top=16)

    axes[1, 0].set_xlim(np.exp([2.8, 9.5]))
    axes[0, 0].set_xlim(np.exp([2.8, 9.5]))

    # axes[1, 1].set_xticks([2002, 2005, 2008, 2010, 2012, 2014, 2015])

    axes[0, 0].set_ylabel("Energy intensity (MJ/US\$)")  # $, essayer en svg
    axes[0, 1].set_ylabel("Energy intensity (MJ/US\$)")
    axes[1, 0].set_ylabel("Energy intensity (MJ/US\$ppp2015)")
    axes[1, 1].set_ylabel("Energy intensity (MJ/US\$ppp2015)")
    axes[0, 0].set_xlabel("Health expenditures (US\$)")
    axes[1, 0].set_xlabel("Health expenditures (US\$ppp2015)")
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
    handles.append(mlines.Line2D([], [], color="black", markersize=80, label="World mean"))

    axes[1, 0].legend(handles=handles, fontsize=10, ncol=2, framealpha=0, loc=2)
    axes[1, 1].legend(handles=handles, fontsize=10, ncol=2, framealpha=0, loc=2)

    axes[0, 0].set_title("a", loc="left")
    axes[0, 1].set_title("b", loc="left")
    axes[1, 0].set_title("c", loc="left")
    axes[1, 1].set_title("d", loc="left")

    plt.tight_layout()
    plt.savefig("figures/fig5.pdf")
    plt.savefig("figures/fig5.png", bbox="tight")
    plt.savefig("figures/fig5.svg")


# .....SI


def sankey_non_ferous():
    i = 2013
    df = pd.read_csv("results/imports_imp" + str(i) + ".txt", index_col=[2, 0, 1])
    continent = pd.read_excel("Data/continent.xlsx", index_col=[0, 1])
    df = df.rename(index=dict(continent.index))
    cmap = sns.color_palette("colorblind", as_cmap="True")

    ind = df.loc["Domestic Extraction Used - Non-ferous metal ores"].groupby(level=[0, 1]).sum().unstack().index
    dict_source = dict(zip(ind.values, [i for i in range(0, len(ind), 1)]))
    dict_target = dict(zip(ind.values, [i for i in range(len(ind), len(ind) * 2, 1)]))

    sankey_metal = df.loc["Domestic Extraction Used - Non-ferous metal ores"].groupby(level=[0, 1]).sum()
    sankey_metal = sankey_metal.unstack().rename(index=dict_source, columns=dict_target).stack().reset_index()
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


def sankey_iron():
    i = 2013
    df = pd.read_csv("results/imports_imp" + str(i) + ".txt", index_col=[2, 0, 1])
    continent = pd.read_excel("Data/continent.xlsx", index_col=[0, 1])
    df = df.rename(index=dict(continent.index))
    cmap = sns.color_palette("colorblind", as_cmap="True")

    ind = df.loc["Domestic Extraction Used - Iron Ore"].groupby(level=[0, 1]).sum().unstack().index
    dict_source = dict(zip(ind.values, [i for i in range(0, len(ind), 1)]))
    dict_target = dict(zip(ind.values, [i for i in range(len(ind), len(ind) * 2, 1)]))

    sankey_metal = df.loc["Domestic Extraction Used - Iron Ore"].groupby(level=[0, 1]).sum()
    sankey_metal = sankey_metal.unstack().rename(index=dict_source, columns=dict_target).stack().reset_index()
    sankey_metal.columns = ["source", "target", "value"]

    label = []
    for i in ind:
        label.append(
            i
            + " ("
            + str(
                int(
                    df.loc["Domestic Extraction Used - Iron Ore"].groupby(level="region prod").sum().loc[i].values
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
                    df.loc["Domestic Extraction Used - Iron Ore"].groupby(level="region cons").sum().loc[i].values
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


def sankey_minerals():
    ext = "Domestic Extraction Used - Non-metalic Minerals"
    i = 2013
    df = pd.read_csv("results/imports_imp" + str(i) + ".txt", index_col=[2, 0, 1])
    continent = pd.read_excel("Data/continent.xlsx", index_col=[0, 1])
    df = df.rename(index=dict(continent.index))
    cmap = sns.color_palette("colorblind", as_cmap="True")

    ind = df.loc[ext].groupby(level=[0, 1]).sum().unstack().index
    dict_source = dict(zip(ind.values, [i for i in range(0, len(ind), 1)]))
    dict_target = dict(zip(ind.values, [i for i in range(len(ind), len(ind) * 2, 1)]))

    sankey_metal = df.loc[ext].groupby(level=[0, 1]).sum()
    sankey_metal = sankey_metal.unstack().rename(index=dict_source, columns=dict_target).stack().reset_index()
    sankey_metal.columns = ["source", "target", "value"]

    label = []
    for i in ind:
        label.append(i + " (" + str(int(df.loc[ext].groupby(level="region prod").sum().loc[i].values / 1000)) + " Mt)")
    for i in ind:
        label.append(i + " (" + str(int(df.loc[ext].groupby(level="region cons").sum().loc[i].values / 1000)) + " Mt)")
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


def sankey_fossil():
    ext = "Domestic Extraction Used - Fossil Fuel: Total"
    i = 2013
    df = pd.read_csv("results/imports_sat" + str(i) + ".txt", index_col=[2, 0, 1])
    continent = pd.read_excel("Data/continent.xlsx", index_col=[0, 1])
    df = df.rename(index=dict(continent.index))
    cmap = sns.color_palette("colorblind", as_cmap="True")

    ind = df.loc[ext].groupby(level=[0, 1]).sum().unstack().index
    dict_source = dict(zip(ind.values, [i for i in range(0, len(ind), 1)]))
    dict_target = dict(zip(ind.values, [i for i in range(len(ind), len(ind) * 2, 1)]))

    sankey_metal = df.loc[ext].groupby(level=[0, 1]).sum()
    sankey_metal = sankey_metal.unstack().rename(index=dict_source, columns=dict_target).stack().reset_index()
    sankey_metal.columns = ["source", "target", "value"]

    label = []
    for i in ind:
        label.append(i + " (" + str(int(df.loc[ext].groupby(level="region prod").sum().loc[i].values / 1000)) + " Mt)")
    for i in ind:
        label.append(i + " (" + str(int(df.loc[ext].groupby(level="region cons").sum().loc[i].values / 1000)) + " Mt)")
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


def pie():
    continent = pd.read_excel("Data/continent.xlsx", index_col=[0, 1])
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

    df = pd.read_csv("results/imports_imp2013.txt", index_col=[2, 0, 1])
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
        print(ext)
        print("region prod China, then GS")
        print(df_ext.groupby(level="region prod").sum()["2013"].loc["China"].sum())
        print(
            df_ext.groupby(level="region prod")
            .sum()["2013"]
            .loc[["Africa", "Rest of Asia", "Middle East", "Latin America", "India"]]
            .sum()
        )
        print("region cons China, then GS")
        print(df_ext.groupby(level="region cons").sum()["2013"].loc["China"].sum())
        print(
            df_ext.groupby(level="region cons")
            .sum()["2013"]
            .loc[["Africa", "Rest of Asia", "Middle East", "Latin America", "India"]]
            .sum()
        )
        plt.tight_layout()
        plt.savefig("figures/pie " + names[j] + ".svg")

    dfi = pd.read_csv("results/imports_sat2013.txt", index_col=[2, 0, 1])
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

        print(ext)
        print("region prod China, then GS")
        print(df_ext.groupby(level="region prod").sum()["2013"].loc["China"].sum())
        print(
            df_ext.groupby(level="region prod")
            .sum()["2013"]
            .loc[["Africa", "Rest of Asia", "Middle East", "Latin America", "India"]]
            .sum()
        )
        print("region cons China, then GS")
        print(df_ext.groupby(level="region cons").sum()["2013"].loc["China"].sum())
        print(
            df_ext.groupby(level="region cons")
            .sum()["2013"]
            .loc[["Africa", "Rest of Asia", "Middle East", "Latin America", "India"]]
            .sum()
        )
        plt.tight_layout()
        plt.savefig("figures/pie " + names[j] + ".svg")


def fig3_LUX():
    fig, axes = plt.subplots(1, figsize=(8, 8))

    col = pd.read_excel("Data/continent.xlsx", index_col=[0])
    cmap = sns.color_palette("colorblind", as_cmap="True")
    dict_color = dict(zip(col["continent"].unique(), cmap[0 : len(col["continent"].unique())]))
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
    handles.append(mlines.Line2D([], [], color="black", markersize=80, label="Region FE regression"))

    col_line = pd.read_excel("Data/continent.xlsx", index_col=[0])
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

    y = y_pri - y_loss
    y.loc[["CA", "CH", "FI", "MT", "NO", "NL", "SI", "SK", "US", "TW"]] = (
        (y_pri - y_loss)
        .rolling(3, center=True, axis=1)
        .mean()
        .loc[["CA", "CH", "FI", "MT", "NO", "NL", "SI", "SK", "US", "TW"]]
    )
    y = y[[2000, 2005, 2010]]
    y[[1995, 2015]] = (y_pri - y_loss)[[1995, 2015]]
    y = y.sort_index(axis=1)

    x_lin = np.linspace(x.stack().min(), x.stack().max(), 100)
    axes.plot(x_lin, np.exp(-2.7350 + 0.0513 * x_lin), color="black")

    names = pd.read_excel("Data/region names.xlsx", index_col=None)
    dict_names = dict(zip(names["region"], names["full name"]))

    annotations_fig2 = pd.read_excel("Data/region names.xlsx", index_col=[0])

    for reg in pop[2015].sort_values(ascending=False).index:
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
    # axes.set_ylim(top=16.5)

    axes.legend(handles=handles, fontsize=10, ncol=2, framealpha=0)

    plt.tight_layout()
    plt.savefig("figures/fig3_LUX.pdf")
    plt.savefig("figures/fig3_LUX.png", bbox="tight")
    plt.savefig("figures/fig3_LUX.svg", bbox="tight")


def comparison():
    pichler = pd.read_excel("Data/pichler.xlsx", index_col=0)
    arup = pd.read_excel("Data/arup.xlsx", index_col=0)
    lenzen = pd.read_excel("Data/lenzen.xlsx", index_col=0)
    df_lenzen = pd.concat(
        [
            impacts["GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"]
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
    df_arup = pd.concat(
        [
            impacts["GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"]
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
    print(str(r2_score(df_pichler["calc co2"], df_pichler["pichler"])) + "r2 score pich")
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
            impacts["GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"]
            .loc["JP"]
            .loc[2015]
            / 1000000000
            / 72
        )
        + "Nansai Japan 2015"
    )  # Nansai 72.0 MtCO2e

    print(
        str(
            impacts["GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"]
            .loc["US"]
            .loc[2013]
            / 1000000000
            / 655
        )
        + "Eckelman US 2013"
    )  # Eckelman 655 MtCO2e

    print(
        str(
            impacts["GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"]
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
            impacts["GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"]
            .loc["CA"]
            .loc[2015]
            / 1000000000
            / 33.0
        )
        + "Eckelman CA 2015"
    )  # Eckelman 33.0 MtCO2e vs. 26.4 exio

    print(
        str(
            impacts["GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"]
            .loc["AU"]
            .loc[2015]
            / 1000000000
            / 35.8
        )
        + "Malik AU 2015"
    )  # Malik 35.8 MtCO2e vs. 29.0 exio

    print(
        str(
            impacts["GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"]
            .loc["CN"]
            .loc[2012]
            / 1000000000
            / 315
        )
        + "Wu CN 2012"
    )  # Wu 315 MtCO2e vs. 472 exio

    print(
        str(
            impacts["GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"]
            .loc["NL"]
            .loc[2015]
            / 1000000000
            / 17.575
        )
        + "Steenmeijer Netherlands 2016"
    )

    (abs(1 - df_pichler["pichler"] / df_pichler["calc co2"])).mean()
    (abs(1 - df_lenzen["lenzen"] / df_lenzen["calc ges"])).mean()
    (abs(1 - df_arup["arup"] / df_arup["calc ges"])).mean()

    (abs(np.log(df_pichler["pichler"] / df_pichler["calc co2"]))).mean()
    (abs(np.log(df_lenzen["lenzen"] / df_lenzen["calc ges"]))).mean()
    (abs(np.log(df_arup["arup"] / df_arup["calc ges"]))).mean()


def employment():
    D_pba = pd.DataFrame()
    for i in range(1995, 2016, 1):
        pathIOT = "Data/EXIO3/IOT_" + str(i) + "_pxp/"
        D_pba[i] = pd.read_csv(pathIOT + "satellite/D_pba.txt", sep="\t", header=[0, 1], index_col=0).head(9).unstack()

    D_pba_imp = pd.DataFrame()
    for i in range(1995, 2016, 1):
        pathIOT = "Data/EXIO3/IOT_" + str(i) + "_pxp/"
        D_pba_imp[i] = pd.read_csv(pathIOT + "impacts/D_pba.txt", sep="\t", header=[0, 1], index_col=0).loc[
            "Value Added"
        ]

    IC = pd.DataFrame()  # intermediate consumption
    for i in range(1995, 2016, 1):
        pathIOT = "Data/EXIO3/IOT_" + str(i) + "_pxp/"
        IC[i] = (
            feather.read_feather(pathIOT + "Z.feather").sum().swaplevel().loc["Health and social work services (85)"]
        )

    AV = D_pba_imp.swaplevel().loc["Health and social work services (85)"]

    (AV / (IC + AV)).T[["CN", "FR", "LU", "ES"]].loc[[2000, 2012]]
    (AV / (IC + AV)).to_excel("results/value_added_in_input.xlsx")

    CoE = (
        D_pba.unstack(level=0).loc["Health and social work services (85)"][2:5].sum().unstack(level=0)
    )  # compensation of employees

    (CoE / (CoE + D_pba_imp.swaplevel().loc["Health and social work services (85)"])).T[["CN", "FR", "LU", "ES"]].loc[
        [2002, 2012]
    ]

    (CoE / (IC + AV)).to_excel("results/compensation_of_employees_in_input.xlsx")
    EI = (
        (
            (satellite["Energy Carrier Net Total"] - satellite["Energy Carrier Net LOSS"])
            .unstack()
            .drop(1995, axis=1)
            .stack()
        ).loc[constant_ppp.unstack().swaplevel().index]
        / constant_ppp.unstack().swaplevel()
        * 1000
    ).unstack()
    EI.to_excel("results/energy_intensity.xlsx")

    fig, axes = plt.subplots(1, figsize=(6, 6))
    x = (CoE / (IC + AV)).drop(["TW", "IN"])
    y = EI.drop("IN")
    x = (x[2015] / x[2002] - 1) * 100
    y = (y[2015] / y[2002] - 1) * 100
    axes.scatter(x, y)
    axes.axvline(color="black")
    axes.axhline(color="black")
    regression = scipy.stats.linregress(x, y)
    x_lin = np.linspace(x.min(), x.max(), 100)
    axes.plot(x_lin, x_lin * regression[0] + regression[1])
    axes.set_ylabel("Variation in Energy Intensity 2002-2015 (\%)")
    axes.set_xlabel("Variation in employment cost in input 2002-2015 (\%)")
    axes.annotate("R2 = " + str(round(regression[2] ** 2, 2)), (-70, 75), fontsize=13, weight="bold")
    plt.savefig("figures/employment_cost.svg")


# Numeric data in paper


def num_data_1():
    rolled.loc[2013].sum() / rolled.loc[1995].sum()
    rolled.loc[2013].sum()
    rolled["Non-metalic minerals"].unstack().sum(axis=1)
    rolled["Non-metalic minerals"].unstack()["China"] / rolled["Non-metalic minerals"].unstack().sum(axis=1)
    rolled_pop.swaplevel().loc["Australia"] / rolled_pop.swaplevel().loc["Africa"]
    rolled.loc[2013].sort_values(by="Non-metalic minerals")


def num_data2():
    (HAQ_agg[2015] / HAQ_agg[1995]).sort_values()


def num_data3():
    pop.loc[["IN", "WF", "ID", "ZA", "WA"]].sum() / pop.sum()


def num_data4():
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


####


def figs():
    rolled, rolled_pop, total, share = fig1()
    fig3()
    fig4()
    fig5()
    comparison()
    fig3_LUX()
    sankey_non_ferous()
    sankey_iron()
    sankey_minerals()
    sankey_fossil()
    pie()
    employment()


def capital_share():
    i = 2013
    LkYhealth = feather.read_feather(pathexio + "Data/LkYhealth/LkYhealth" + str(i) + ".feather")
    LYhealth = feather.read_feather(pathexio + "Data/LYhealth/LYhealth" + str(i) + ".feather")
    pathIOT = pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/"
    S_sat = pd.read_csv(pathIOT + "satellite/S.txt", delimiter="\t", header=[0, 1], index_col=[0])
    S_imp = pd.read_csv(pathIOT + "impacts/S.txt", delimiter="\t", header=[0, 1], index_col=[0])
    SLkY = pd.DataFrame()
    SLY = pd.DataFrame()
    for i in impacts_ext:
        SLkY[i] = LkYhealth.mul(S_imp.loc[i], axis=0).groupby(level="sector").sum().sum()
        SLY[i] = LYhealth.mul(S_imp.loc[i], axis=0).groupby(level="sector").sum().sum()
    for i in satellite_ext:
        SLkY[i] = LkYhealth.mul(S_sat.loc[i], axis=0).groupby(level="sector").sum().sum()
        SLY[i] = LYhealth.mul(S_sat.loc[i], axis=0).groupby(level="sector").sum().sum()

    SLY_2 = (
        (
            pd.concat(
                [
                    SLY["Domestic Extraction Used - Non-metalic Minerals"],
                    SLY[
                        [
                            "Domestic Extraction Used - Iron Ore",
                            "Domestic Extraction Used - Non-ferous metal ores",
                        ]
                    ].sum(axis=1),
                    SLY["Domestic Extraction Used - Fossil Fuel: Total"],
                    SLY["Energy Carrier Net Total"] - SLY["Energy Carrier Net LOSS"],
                ],
                keys=["Non-metalic minerals", "Metal ores", "Fossil fuel", "Final energy"],
                axis=1,
            )
            .swaplevel()
            .swaplevel()
        )
        .groupby(level=0)
        .sum()
    )
    SLY_2.loc["World"] = SLY_2.sum()
    SLkY_2 = (
        (
            pd.concat(
                [
                    SLkY["Domestic Extraction Used - Non-metalic Minerals"],
                    SLkY[
                        [
                            "Domestic Extraction Used - Iron Ore",
                            "Domestic Extraction Used - Non-ferous metal ores",
                        ]
                    ].sum(axis=1),
                    SLkY["Domestic Extraction Used - Fossil Fuel: Total"],
                    SLkY["Energy Carrier Net Total"] - SLkY["Energy Carrier Net LOSS"],
                ],
                keys=["Non-metalic minerals", "Metal ores", "Fossil fuel", "Final energy"],
                axis=1,
            )
            .swaplevel()
            .swaplevel()
        )
        .groupby(level=0)
        .sum()
    )
    SLkY_2.loc["World"] = SLkY_2.sum()
    ((SLkY_2 - SLY_2) / SLkY_2 * 100).to_excel("results/capital_share.xlsx")


def fig4_other():
    i = 2013
    LkYhealth = feather.read_feather(pathexio + "Data/LkYhealth/LkYhealth" + str(i) + ".feather")
    LYhealth = feather.read_feather(pathexio + "Data/LYhealth/LYhealth" + str(i) + ".feather")
    pathIOT = pathexio + "Data/EXIO3/IOT_" + str(i) + "_pxp/"
    S_sat = pd.read_csv(pathIOT + "satellite/S.txt", delimiter="\t", header=[0, 1], index_col=[0])
    S_imp = pd.read_csv(pathIOT + "impacts/S.txt", delimiter="\t", header=[0, 1], index_col=[0])
    SLkY = pd.DataFrame()
    SLY = pd.DataFrame()
    for i in impacts_ext:
        SLkY[i] = LkYhealth.mul(S_imp.loc[i], axis=0).groupby(level="sector").sum().unstack()
        SLY[i] = LYhealth.mul(S_imp.loc[i], axis=0).groupby(level="sector").sum().unstack()
    for i in satellite_ext:
        SLkY[i] = LkYhealth.mul(S_sat.loc[i], axis=0).groupby(level="sector").sum().unstack()
        SLY[i] = LYhealth.mul(S_sat.loc[i], axis=0).groupby(level="sector").sum().unstack()

    conc = pd.read_excel("Data/concordance_products2.xlsx", index_col=[0, 1])
    df = agg(SLkY.groupby(level=[0, 2]).sum().unstack(level=0), conc, axis=0)
    df = pd.concat(
        [
            df["Domestic Extraction Used - Non-metalic Minerals"],
            df["Domestic Extraction Used - Iron Ore"] + df["Domestic Extraction Used - Non-ferous metal ores"],
            df["Domestic Extraction Used - Fossil Fuel: Total"],
            df["Energy Carrier Net Total"] - df["Energy Carrier Net LOSS"],
        ],
        keys=["Non-metalic minerals", "Metal ores", "Fossil fuel", "Final energy"],
        axis=1,
    )
    df2 = df.div(df.sum(), axis=1) * 100
    df2.swaplevel(axis=1)["CN"]
