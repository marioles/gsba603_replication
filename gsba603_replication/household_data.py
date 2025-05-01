import pandas as pd


def _get_base_path():
    base_path = str()
    return base_path


def _get_file_path():
    file_path = str()
    return file_path


def generate_path(wave_path):
    base_path = _get_base_path()
    file_path = _get_file_path()
    path = f"{base_path}/{wave_path}/{file_path}"
    return path


def get_label_dict():
    label_dd = {
        "cfo0a3": "Month",
        "cfo0a4": "Plot #",
        "cfo0a5": "Crop #",
        "any_input": "Used any inputs",
        "any_lexchange": "Any free labor",
        "any_hired": "Any hired labor",
        "n_hired": "# of workers hired",
        "cash_hired": "Labor expenses",
        "inkind_hired": "Inkind value of labor expenses",
        "any_brokers": "Hired people through brokers",
        "n_hhlabor": "# Hh members working on crop",
        "any_hhlabor": "Any hh members working on crop",
        "any_eqrent": "Any equipement/animals borrowed/rented",
        "any_hheq": "Any equipment or animals from hh",
        "any_harvest": "Any harvest"
    }
    return label_dd


def get_keep_list():
    keep_ls = [
        "province",
        "village",
        "household",
        "month",
        "cfo0a4",
        "cfo0a5",
        "any_input",
        "any_lexchange",
        "any_hired",
        "n_hired",
        "cash_hired",
        "inkind_hired",
        "any_brokers",
        "n_hhlabor",
        "any_hhlabor",
        "any_eqrent",
        "any_hheq",
        "any_harvest",
    ]
    return keep_ls


def _get_apply_rename_list():
    apply_rename_ls = [
        "136-144",
        "145-160",
        "161-172",
    ]
    return apply_rename_ls


def _get_rename_dict():
    rename_dd = {
        "Province": "province",
        "Village": "village",
        "Household": "household",
        "Month": "month",
    }
    return rename_dd


def apply_rename(df, wave_path):
    copy_df = df.copy()
    apply_rename_ls = _get_apply_rename_list()
    if wave_path in apply_rename_ls:
        rename_dd = _get_rename_dict()
        copy_df = copy_df.rename(columns=rename_dd)
    return copy_df


def read_household_file(wave_path):
    path = generate_path(wave_path=wave_path)
    crop_df = pd.read_stata(path)

    crop_df = apply_rename(df=crop_df, wave_path=wave_path)

    crop_df["any_input"] = crop_df["cfo4a"] == 1
    crop_df["any_lexchange"] = crop_df["cfo5a"] == 1
    crop_df["any_hired"] = crop_df["cfo6a"] == 1
    crop_df["n_hired"] = crop_df.apply(lambda x: 0 if x["any_hired"] == 0 else x["cfo6b"], axis=1)
    crop_df["cash_hired"] = crop_df.apply(lambda x: 0 if x["any_hired"] == 0 else x["cfo6l"], axis=1)
    crop_df["inkind_hired"] = crop_df.apply(lambda x: 0 if x["any_hired"] == 0 else x["cfo6m"], axis=1)
    crop_df["any_brokers"] = crop_df["cfo6n"] == 1

    crop_df["n_hhlabor"] = crop_df["cfo7a"]
    crop_df["any_hhlabor"] = crop_df.apply(lambda x: not pd.isna(x["n_hhlabor"]) and x["n_hhlabor"] > 0, axis=1)
    crop_df["any_eqrent"] = crop_df["cfo8a"] == 1
    crop_df["any_hheq"] = crop_df["cfo9a"] == 1
    crop_df["any_harvest"] = crop_df["cfo10a"] == 1

    keep_ls = get_keep_list()
    keep_df = crop_df.loc[:, keep_ls].copy()
    return keep_df


def get_wave_path_ls():
    wave_path_ls = [
        "01-88",
        "89-104",
        "105-120",
        "121-135",
        "136-144",
        "145-160",
        "161-172",
    ]
    return wave_path_ls


def process_household_data():
    wave_path_ls = get_wave_path_ls()
    concat_ls = list()
    for wave_path in wave_path_ls:
        try:
            append_df = read_household_file(wave_path=wave_path)
            concat_ls.append(append_df)
        except Exception as exc:
            msg = str(exc)
            print(msg)
    df = pd.concat(concat_ls, axis=0)
    df = df.reset_index()
    return df
