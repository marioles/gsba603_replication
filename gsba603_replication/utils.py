import os
import pandas as pd
import re

from statsmodels.stats.sandwich_covariance import cov_cluster


def calculate_outcomes(df):
    df["exp_nf_w"] = df["exp_nf"].sub(df["tot_hhspend"], fill_value=0)

    df["tot_exp_w"] = df["exp_nf_w"].add(df["food_w"], fill_value=0)

    df["totcons_w"] = df["exp_nf_w"].add(df["tot_hhspend"], fill_value=0)
    df["totcons_w"] = df["totcons_w"].add(df["food_w"], fill_value=0)

    df["costs_nw_w"] = df["costs_ag_w"].add(df["costs_livs_w"], fill_value=0)
    df["costs_nw_w"] = df["costs_nw_w"].add(df["costs_fs_w"], fill_value=0)
    df["costs_nw_w"] = df["costs_nw_w"].add(df["costs_nfbiz_w"], fill_value=0)

    df["REVnw_w"] = df["AgREV_w"].add(df["LREV_w"], fill_value=0)
    df["REVnw_w"] = df["REVnw_w"].add(df["FSREV_w"], fill_value=0)
    df["REVnw_w"] = df["REVnw_w"].add(df["BREV_w"], fill_value=0)

    df["a_symptom"] = df["n_symptom"].apply(lambda x: 1 if x > 0 else 0)
    return df


def filter_attrition(df):
    attrition_filter_ss = df["no_attrition_food"] == 1
    filter_df = df.loc[attrition_filter_ss].copy()
    return filter_df


def filter_window(df, months=24):
    window_filter_ss = df["tau"] >= -months
    window_filter_ss = window_filter_ss & (df["tau"] < months)
    window_filter_df = df.loc[window_filter_ss].copy()
    return window_filter_df


def _get_base_path():
    base_path = os.getenv("BASE_PATH")
    return base_path


def _get_clean_data_path():
    clean_path = os.getenv("CLEAN_PATH")
    return clean_path


def _get_file_path():
    file_path = os.getenv("FILE_PATH")
    return file_path


def _get_dyads_file_path():
    file_path = os.getenv("DYADS_FILE_PATH")
    return file_path


def get_export_path():
    export_path = os.getenv("EXPORT_PATH")
    return export_path


def get_treat_file_path():
    base_path = _get_base_path()
    clean_path = _get_clean_data_path()
    file_path = _get_file_path()
    path = f"{base_path}/{clean_path}/{file_path}"
    return path


def get_dyads_file_path():
    base_path = _get_base_path()
    clean_path = _get_clean_data_path()
    file_path = _get_dyads_file_path()
    path = f"{base_path}/{clean_path}/{file_path}"
    return path


def read_data(file="TreatHS"):
    if file == "TreatHS":
        path = get_treat_file_path()
    elif file == "dyads_es_max":
        path = get_dyads_file_path()
    else:
        msg = f"file {file} not implemented"
        raise Exception(msg)
    raw_df = pd.read_stata(path)
    return raw_df


def filter_first_half_shock(df):
    treatment_filter_ss = df["treatment_subsample"] == 1
    treatment_filter_ss = treatment_filter_ss | (df["placebo_subsample"] == 2)
    treatment_filter_df = df.loc[treatment_filter_ss].copy()
    return treatment_filter_df


def recode_tau(df, months=None):
    if months is None:
        months = 24

    recode_df = df.copy()

    step = 6
    distance = int(months / step)
    bins = int(2 * distance + 1)
    bin_ls = [months - i * step for i in range(bins)]
    bin_ls.reverse()
    bin_ls[0] += -1

    label_ls = [i for i in range(-distance, distance)]

    tau_ss = recode_df["tau"].copy()
    recode_df["ttt"] = tau_ss
    recode_df["tau"] = pd.cut(tau_ss, bins=bin_ls, labels=label_ls, right=False)
    recode_df["tau"] = recode_df["tau"].astype(int)
    return recode_df


def cov_cluster_2way(model, cluster1, cluster2):
    cov1 = cov_cluster(model, cluster1)
    cov2 = cov_cluster(model, cluster2)
    cov12 = cov_cluster(model, [f"{i}_{j}" for i, j in zip(cluster1, cluster2)])  # intersection
    return cov1 + cov2 - cov12


def extract_relevant_values(ss, regex_str):
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


def get_append_series():
    ss = pd.Series(0, index=[-1], dtype=float)
    return ss


def get_confidence_list():
    confidence_ls = [0.95, 0.9]
    return confidence_ls


def get_confidence_color(confidence):
    confidence_color_dd = {
        0.90: "black",
        0.95: "gray",
    }
    color = confidence_color_dd[confidence]
    return color


def export_plot(name, panel_plot):
    export_path = get_export_path()
    path = f"{export_path}/{name}"
    panel_plot.savefig(path)
