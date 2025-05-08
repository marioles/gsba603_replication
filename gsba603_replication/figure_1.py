import matplotlib.pyplot as plt
import pandas as pd
import re
import scipy as sp
import statsmodels.formula.api as smf

from . import utils


def filter_first_half_shock(df):
    treatment_filter_ss = df["treatment_subsample"] == 1
    treatment_filter_ss = treatment_filter_ss | (df["placebo_subsample"] == 2)
    treatment_filter_df = df.loc[treatment_filter_ss].copy()
    return treatment_filter_df


def recode_tau(df):
    recode_df = df.copy()

    bin_ls = [-25, -18, -12, -6, 0, 6, 12, 18, 24]
    label_ls = [-4, -3, -2, -1, 0, 1, 2, 3]

    tau_ss = recode_df["tau"].copy()
    recode_df["ttt"] = tau_ss
    recode_df["tau"] = pd.cut(tau_ss, bins=bin_ls, labels=label_ls, right=False)
    recode_df["tau"] = recode_df["tau"].astype(int)
    return recode_df


def pre_process_data(df):
    # Two-year analysis window
    window_filter_df = utils.filter_window(df=df, months=24)

    # Observations hit during first half but not yet treated
    treatment_filter_df = filter_first_half_shock(df=window_filter_df)

    # Recode tau
    recode_df = recode_tau(df=treatment_filter_df)

    # No-attrition filter
    filter_df = utils.filter_attrition(df=recode_df)

    # Calculations
    filter_df = utils.calculate_outcomes(df=filter_df)

    filter_df["a_symptom"] = filter_df["n_symptom"].apply(lambda x: 1 if x > 0 else 0)
    return filter_df


def regress_diff_in_diff(df, dv, tau, treatment, fe=None, control=None, clustvar=None):
    copy_df = df.copy()

    if treatment == "Treatment":
        rename_dd = {"Treatment": "treatment"}
        copy_df = copy_df.rename(columns=rename_dd)
        treatment = "treatment"

    iter_ls = [
        dv,
        tau,
        treatment,
        clustvar,
        fe,
        control,
    ]
    keep_ls = list()
    for i in iter_ls:
        if isinstance(i, list):
            keep_ls += i
        else:
            keep_ls.append(i)
    keep_ls = list(set(keep_ls))
    copy_df = copy_df.loc[:, keep_ls].copy()
    copy_df = copy_df.dropna()

    # Handle categorical variable
    tau_ss = copy_df[tau].copy()
    category_ls = sorted(tau_ss.dropna().unique())
    tau_cat_ss = pd.Categorical(tau_ss, categories=category_ls, ordered=True)
    tau_cat = f"{tau}_cat"
    copy_df[tau_cat] = tau_cat_ss

    if clustvar is not None:
        copy_df[clustvar] = pd.Categorical(copy_df[clustvar], ordered=True)

    # Create formula
    formula = f"{dv} ~ C({tau_cat}, Treatment(reference=-1)) + {treatment} + C({tau_cat}, Treatment(reference=-1)):{treatment}"
    if isinstance(fe, list):
        fe_formula_ls = [f"C({i})" for i in fe]
        fe_formula = " + ".join(fe_formula_ls)
    else:
        fe_formula = f"C({fe})"
    formula = f"{formula} + {fe_formula}"

    if control is not None:
        if isinstance(control, list):
            control_formula = " + ".join(control)
        else:
            control_formula = control
        formula = f"{formula} + {control_formula}"

    model = smf.ols(formula, data=copy_df)
    if clustvar is not None:
        cluster_ss = copy_df[clustvar].copy()
        cluster_dd = {"groups": cluster_ss}
        results = model.fit(cov_type="cluster", cov_kwds=cluster_dd)
    else:
        results = model.fit()

    return results


def get_append_series():
    ss = pd.Series(0, index=[-1], dtype=float)
    return ss


def get_regex():
    regex_str = r"^C\(tau_cat, Treatment\(reference=-1\)\)\[T\.(.*?)\]:treatment$"
    return regex_str


def extract_revelant_values(ss, regex_str):
    filter_ss = ss.filter(regex=regex_str, axis=0)
    index_ls = list(filter_ss.index)
    reindex_ls = [re.match(regex_str, i).group(1) for i in index_ls]
    reindex_ls = [int(i) for i in reindex_ls]
    filter_ss.index = reindex_ls
    return filter_ss


def append_baseline(ss):
    append_ss = get_append_series()
    concat_ls = [append_ss, ss]
    concat_ss = pd.concat(concat_ls, axis=0)
    sort_ss = concat_ss.sort_index()
    return sort_ss


def get_title(dv):
    title_dd = {
        "a_symptom": "Panel A. Prob. of reporting any symptoms",
        "tot_hhspend": "Panel B. Health spending",
        "tot_exp_w": "Panel C. Non health consumption spending",
        "hours_hired": "Panel D. Hired labor (hours/month)",
        "costs_nw_w": "Panel E. Business spending",
        "REVnw_w": "Panel F. Revenues",
    }
    title = title_dd[dv]
    return title


def get_x_title(dv):
    x_title = "Time to event (half year)"
    return x_title


def get_y_title(dv):
    y_title_dd = {
        "a_symptom": "Perc. points",
        "tot_hhspend": "THB",
        "tot_exp_w": "THB",
        "hours_hired": "Hours",
        "costs_nw_w": "THB",
        "REVnw_w": "THB",
    }
    y_title = y_title_dd[dv]
    return y_title


def get_confidence_color(confidence):
    confidence_color_dd = {
        0.90: "black",
        0.95: "gray",
    }
    color = confidence_color_dd[confidence]
    return color


def plot_from_data(ax, plot_df, dv, confidence_ls):
    title = get_title(dv=dv)
    x_title = get_x_title(dv=dv)
    y_title = get_y_title(dv=dv)

    x_ss = plot_df.index
    y_ss = plot_df["coefficient"]

    ax.plot(x_ss, y_ss, marker="o", linewidth=2, label="Coef")

    for confidence in confidence_ls:
        lower_label = f"lower_{int(100 * confidence)}"
        upper_label = f"upper_{int(100 * confidence)}"
        lower_ss = plot_df[lower_label]
        upper_ss = plot_df[upper_label]
        plot_label = f"{int(100 * confidence)}% CI"
        plot_color = get_confidence_color(confidence)
        ax.fill_between(x_ss, lower_ss, upper_ss, color=plot_color, alpha=0.25, label=plot_label)

    ax.axhline(0, color="maroon", linestyle="--")
    ax.axvline(-1, color="maroon", linestyle="--")

    ax.set_title(title)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.legend()
    ax.grid(True)


def get_confidence_list():
    confidence_ls = [0.95, 0.9]
    return confidence_ls


def generate_sub_plot(ax, dv, result):
    param_ss = result.params
    regex_str = get_regex()

    raw_dd = {
        "coefficient": result.params,
        "std_error": result.bse,
    }

    extract_dd = {k: extract_revelant_values(ss=ss, regex_str=regex_str) for k, ss in raw_dd.items()}
    concat_dd = {k: append_baseline(ss=ss) for k, ss in extract_dd.items()}
    plot_df = pd.DataFrame(concat_dd)

    confidence_ls = get_confidence_list()
    for confidence in confidence_ls:
        interval = (1 - confidence) / 2 + confidence
        critical_value = sp.stats.norm.ppf(interval)
        lower_label = f"lower_{int(100 * confidence)}"
        upper_label = f"upper_{int(100 * confidence)}"
        plot_df[lower_label] = plot_df["coefficient"] - critical_value * plot_df["std_error"]
        plot_df[upper_label] = plot_df["coefficient"] + critical_value * plot_df["std_error"]

    plot_from_data(ax=ax, plot_df=plot_df, dv=dv, confidence_ls=confidence_ls)


def generate_plot(dependent_ls, result_dd):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    iterate_ls = zip(dependent_ls, axes)
    for dv, ax in iterate_ls:
        result = result_dd[dv]
        generate_sub_plot(ax=ax, dv=dv, result=result)

    plt.tight_layout()
    return fig


def export_plot(name, panel_plot):
    export_path = utils._get_export_path()
    path = f"{export_path}/{name}"
    panel_plot.savefig(path)
