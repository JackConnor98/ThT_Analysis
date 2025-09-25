import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.optimize import curve_fit
import math
import re

# ------------------------------------------------------------
### Features to add: ###
# Selectable fit equations
# Click and drag to select wells
# Make tmax use the sigmoid
# Be able to load multiple plate and handle normalisation seperately
# ------------------------------------------------------------

# ------------------------------------------------------------
# Initial variables
# ------------------------------------------------------------
data_columns = ["protein", "conc", "wells"]
df = pd.DataFrame(columns=data_columns)
#fit_data = pd.DataFrame(columns=["Well", "t50"])
fit_data = pd.DataFrame()
time_label = None
flatliners = []
selected_wells = []
selections = []
well_buttons = []
df_tidy = None
# ------------------------------------------------------------
# Functions
# ------------------------------------------------------------

# ------------------------------------------------------------
# Curve fit equations

def fitting_eq1(t,l,k,th,y0,yend):
    # t = time, l = lambda, k = kappa, th = theta
    # y = M(t)/m(0) - this is normalised ThT intensities
    # Equation 15 from Michaels_Knowles_PyseRevE_2019
    # This is valid for self-assembly pathways with primary nucleation, secondary processes, and elongation
    # It's not quite as accurate as some of the other solutions, but it's more general (and more mathematically insightful), and it's still a very good approximation
    y = (yend-y0) * (1 - (1 + ((l**2) / (2*((k)**2)*th) * np.exp(k*t))) ** (-th)) + y0
    return y

def fitting_eq2(t,nc,l,y0,yend):
    # nc = length of fibril
    # Equation 1 from Cohen_Knowles_JCP_2011a
    # It's valid for cases where there's primary nucleation and elongation, but no secondary processes 
    # Incorporating secondary processes into the maths was Ferrone and later Knowles' innovation
    # The actual references for Oosawa's exact solution are hard to get hold of and phrased in a lot of outdated nomenclature, but Cohen et al. (Knowles lab) give a good presentation of it in their first 2011 paper
    # There's no seeding here
    # Had to fix the nc and y0 lower limits as >0
    sqrt_nc_over_2 = math.sqrt(nc/2)
    sech_arg = sqrt_nc_over_2 * l * t    
    sech_term = (1/np.cosh(sech_arg)) ** (2/nc)
    y = (yend- y0) * (1 - sech_term) + y0
    return y

def fitting_eq3(t,kappa,l,nc,n2,y0,yend):
    # This equation is valid for cases where there's primary nucleation, 
    # secondary processes, and elongation - it's like the first equation, 
    # but it's a slightly better approximation of the early time, although 
    # it's less physically meaningful and interpretable which is why they've 
    # been working on new equations since then. It can describe secondary 
    # nucleation (n2 > 0), yor fragmentation (n2 = 0), although in practice 
    # to get it to work you will need to set n2 at a finite but very small 
    # value for fragmentation, eg. 1e-3. I think there is actually a better 
    #approach we can take with the fragmentation (ie. we can take the limit 
    # of eq. (1) as n2 -> 0, which will give us a slightly different equation 
    # that you can model which won't need n2 as a parameter)
    kinf = np.sqrt(((2 * (kappa ** 2))/(n2 * (n2 + 1)))+ (2 * (l ** 2))/nc)
    Cplus = (l **2)/(2 * (kappa ** 2))
    Cminus = -(l **2)/(2 * (kappa ** 2))
    kinf_tilde = np.sqrt((kinf ** 2) - (4 * Cplus * Cminus * (kappa ** 2)))
    Bplus = (kinf + kinf_tilde)/(2 * kappa)
    Bminus = (kinf - kinf_tilde)/(2 * kappa)
    first_term = (Bplus + Cplus)/(Bplus + (Cplus * np.exp(kappa * t)))
    second_term = (Bminus + (Cplus * np.exp(kappa * t)))/(Bminus + Cplus)
    temp_y = ((first_term * second_term) ** ((kinf ** 2)/(kappa * kinf_tilde))) 
    y = (yend - y0) * (1 - (temp_y * np.exp(-kinf * t))) + y0
    return y

def parse_limits(text):
    """
    Parse a string like "1,5", "1 - 5", or "1-5" into (min, max).
    Returns None if invalid.
    """
    if not text:
        return None
    try:
        # Replace hyphen with comma if it's being used as a range
        # (but keep negative numbers intact)
        cleaned = text.replace(" ", "")
        if "-" in cleaned[1:]:  # skip the first char so "-5,10" still works
            cleaned = cleaned.replace("-", ",", 1)  # only replace first hyphen
        parts = cleaned.split(",")
        if len(parts) != 2:
            return None
        return tuple(map(float, parts))
    except ValueError:
        print("⚠️ Invalid limits. Use format: min,max or min-max")
        return None

# ------------------------------------------------------------

def back_to_data_selection():
    main_frame.pack_forget()
    initial_frame.pack()
    clear_all()

def toggle_well(button, well_name):
    if well_name in selected_wells:
        selected_wells.remove(well_name)
        button.config(bg="lightgrey")  # Deselect
    else:
        selected_wells.append(well_name)
        button.config(bg="lightblue")  # Select

def finalize_selection():
    global selections
    protein_name = protein_name_entry.get()
    protein_concentration = protein_concentration_entry.get()
    protein_colour = colour_entry.get()
    
    current_data = {
        "protein": [protein_name],
        "conc": [protein_concentration],
        "wells": [", ".join(selected_wells)]
    }
    if palette_var.get() == "manual":
        current_data["colour"] = [protein_colour]
    else:
        current_data["colour"] = [""]

    current_df = pd.DataFrame(current_data)

    global df
    df = pd.concat([df, current_df], ignore_index=True)

    # Store the current wells as the last selection
    selections.append(list(selected_wells))

    for button in well_buttons:
        if button.cget('bg') == 'lightblue':
            button.config(bg='orange')

    selected_wells.clear()
    update_dataframe_display()

def update_dataframe_display():
    df_display.config(state=tk.NORMAL)
    df_display.delete(1.0, tk.END)
    df_display.insert(tk.END, df.to_string(index=False))
    df_display.config(state=tk.DISABLED)

def update_fit_parameters_display():
    fit_display.config(state=tk.NORMAL)  # make editable
    fit_display.delete(1.0, tk.END)      # clear old content
    fit_display.insert(tk.END, fit_data.to_string(index=False))
    #fit_display.insert(tk.END, str(fit_data))  # insert new data
    fit_display.config(state=tk.DISABLED)  # lock it

def clear_last():
    global df, fit_data, selections
    df = df[:-1]

    wells_to_clear = selections.pop()

    for well_name in wells_to_clear:
        if well_name in well_button_map:
            well_button_map[well_name].config(bg='lightgrey')

     # Remove any rows from fit_data where "Well" matches a cleared well
    fit_data = fit_data[~fit_data["Well"].isin(wells_to_clear)].reset_index(drop=True)

    # If there are no rows left, reset fit_data entirely
    if fit_data.empty:
        fit_data = pd.DataFrame()

    selected_wells.clear()
    update_dataframe_display()
    update_fit_parameters_display()

def clear_all():
    global df, fit_data
    df = pd.DataFrame()
    update_dataframe_display()

    fit_data = pd.DataFrame()
    update_fit_parameters_display()

    for button in well_buttons:
        if button.cget('bg') in ("lightblue", "orange"):
            button.config(bg='lightgrey')

    selected_wells.clear()

def choose_data_file():

    global df_tidy

    file_path = filedialog.askopenfilename(
        title="Select your data file",
        filetypes=[
            ("CSV and Excel files", ("*.csv", "*.xlsx")), 
            ("CSV Files", "*.csv"),
            ("Excel Files", "*.xlsx"),
            ("All files", "*") 
        ]
    )

    if file_path:
        try:
            # Read and store tidy data
            df_tidy = read_data(file_path)

            # Hide initial frame and show the main one
            initial_frame.pack_forget()
            main_frame.pack()

            return file_path

        except Exception as e:
            print(f"Error loading file: {e}")
            return

def time_to_hours(t):
    """
    Convert strings like '0 h 8 min' or '2 h' or '45 min' to hours (float).
    """
    if pd.isna(t):  # in case of NaN
        return None
    
    text = str(t).lower().strip()
    hours = 0
    minutes = 0
    
    # extract hours
    h_match = re.search(r'(\d+)\s*h', text)
    if h_match:
        hours = int(h_match.group(1))
    
    # extract minutes
    m_match = re.search(r'(\d+)\s*min', text)
    if m_match:
        minutes = int(m_match.group(1))
    
    return hours + minutes/60.0

def read_data(file_path):

    global time_label

    # Read the entire sheet first to find headers
    temp_df = pd.read_excel(file_path, header=None)
    
    # Find row with headers
    for i, row in temp_df.iterrows():
        if any("Content" in str(cell) for cell in row):
            header_row = i
            break
    
    # Now read properly
    df = pd.read_excel(file_path, 
                      header=header_row)
    
    # Clean column names
    df.columns = [str(col).replace('\n', ' ') for col in df.columns]

    # Handling Well vs Well Row and Well Col
    if df.columns[0] == "Well":
        # Split Well into Well Row and Well Col
        df["Well Row"] = df["Well"].str.extract(r"([A-Za-z]+)")
        df["Well Col"] = df["Well"].str.extract(r"(\d+)").astype("Int64")
        df = df.drop(columns=["Well"])


    meta_cols = ['Well Row', 'Well Col', 'Content']

    # Melt wide to long
    df_long = df.melt(id_vars=meta_cols, var_name='measurement', value_name='value')

    # Detect the time column, whatever its unit (h or s)
    time_label = df_long.loc[df_long['Content'].str.startswith("Time"), 'Content'].unique()[0]

    # Extract the Time mapping
    time_map = df_long[df_long['Content'] == time_label][['measurement', 'value']].set_index('measurement')['value']

    # Remove the Time row
    df_long = df_long[df_long['Content'] != time_label]

    # Add Time column with the actual label
    df_long[time_label] = df_long['measurement'].map(time_map)

    # Final tidy dataframe: one row per well, time, and sample
    df_tidy = df_long.drop(columns=['measurement']).rename(columns={'Content': 'Sample', 'value': 'Fluorescence'})
    
    # Create Well identifier
    df_tidy['Well'] = df_tidy['Well Row'].astype(str).str.strip() + df_tidy['Well Col'].astype(int).astype(str)

    # Convert time to hours
    if df_tidy[time_label].astype(str).str.contains(r'(h|min)', case=False).any():
        df_tidy[time_label] = df_tidy[time_label].apply(time_to_hours)

    # Identify flatliners to avoid fitting them
    find_flatliners(df_tidy)

    return df_tidy

def find_flatliners(data):
    
    global flatliners

    data = data.copy()
    max_data = data["Fluorescence"].max()
    
    max_by_well = data.groupby("Well")["Fluorescence"].max()

    # Wells whose max < 10% of the overall max
    #flatliners = max_by_well[max_by_well < 0.01 * max_data].index.tolist()
    flatliners = list(set(flatliners) | set(
    max_by_well[max_by_well < 0.1 * max_data].index
    ))

def normalise_data(data):
    data = data.copy()
    min_val = data["Fluorescence"].min()
    max_val = data["Fluorescence"].max()
    
    data["Fluorescence"] = (data["Fluorescence"] - min_val) / (max_val - min_val)

    return data

def calculate_t50(x, y):
    max_y = y.max()
    t50 = x[np.where(y >= 0.5 * max_y)[0][0]]

    return t50

def calculate_tmax(x, y):
    max_y = y.max()
    tmax = x[np.where(y >= max_y)[0][0]]

    return tmax

def calculate_tlag(x, y, t50):

    # Here tlag is calculated by fitting a sigmoid function to the data
    # This approach was taken from: https://doi.org/10.1080/13506129.2017.1304905

    def sigmoid(t, yi, yf, t50, tau):
        return yi + (yf - yi) / (1 + np.exp(-(t - t50)/tau))

    
    initial_guess = [min(y), max(y), t50, 1]

    popt, _ = curve_fit(sigmoid, x, y, p0=initial_guess, maxfev=10000)

    # popt contains the fitted parameters: yi, yf, t50, tau
    yi_fit, yf_fit, t50_fit, tau_fit = popt

     # Calculate tlag
    tlag = t50_fit - 2*tau_fit

    ymax = yf_fit  # maximum value of sigmoid

    #return tlag, ymax
    return tlag

def fit_curve(well, x, y):

    equation = equation_var.get()
    normalise_choice = normalise_var.get()

    # Check if well is a flatliner
    if well in flatliners:
        print(f"Curve Fit -- Skipping flatliner well: {well}")
        return np.array([]), np.array([])
    else:
        print(f"Fitting data for well: {well}")

        if equation == "1":
            try:
                if normalise_choice == "Global" or normalise_choice == "Local":

                    # Initial guesses for parameters (you can tune these)
                    l0 = 1
                    k0 = 1
                    th0 = 1
                    y0 = 0
                    yend = 1
                    p0 = [l0, k0, th0, y0, yend]
                
                else:
                    p0 = [0.1, 0.1, 1, y[0], y[-1]]
                
                popx, _ = curve_fit(fitting_eq1, x, y, p0=p0, maxfev=10000)

                # param_names = ["l", "k", "th", "y0", "yend"]
                # for name, val in zip(param_names, popx):
                #     print(f"{name} = {val:.4f}")

                # Create smooth time points for plotting fitted curve
                x_fit = np.linspace(x.min(), x.max(), 200)
                y_fit = fitting_eq1(x_fit, *popx)

                return x_fit, y_fit, popx

            except Exception as e:
                print(f"Could not fit data for well {well}: {e}")

                return np.array([]), np.array([])
            
        if equation == "2":

            try:            
                print("Equation 2: Not implemented Yet")

                return np.array([]), np.array([])

            except Exception as e:
                print(f"Could not fit data for well {well}: {e}")

                return np.array([]), np.array([]), np.array([])
            
        if equation == "3":

            try:
                print("Equation 3: Not implemented Yet")

                return np.array([]), np.array([]), np.array([])

            except Exception as e:
                print(f"Could not fit data for well {well}: {e}")

                return np.array([]), np.array([]), np.array([])

def make_fit_data(well, x, y):
    global fit_data, flatliners
    
    equation = equation_var.get()

    if equation == "none":
            
        if fit_data.empty:
            fit_data = pd.DataFrame(columns=["Well", "t50", "tmax", "tlag"])
        
        # Check if well is a flatliner
        if well in flatliners:
            fit_data.loc[len(fit_data)] = [well, "NA", "NA", "NA"]
            
        else:
            if well not in fit_data["Well"].values:
                # Calculate t50 and add to fit_data
                t50 = calculate_t50(x, y)
                tmax = calculate_tmax(x, y)
                tlag = calculate_tlag(x, y, t50)
                
                fit_data.loc[len(fit_data)] = [well, t50, tmax, tlag]

    if equation == "1":
        if fit_data.empty:
            fit_data = pd.DataFrame(columns=["Well", "l", "k", "th", "y0", "yend", "t50", "tmax", "tlag"])

        # Check if well is a flatliner
        if well in flatliners:
            fit_data.loc[len(fit_data)] = [well, "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
        else:
            if well not in fit_data["Well"].values:
                try:
                    # Fit the curve to get parameters
                    _, _, popx = fit_curve(well, x, y)

                    l, k, th, y0_fit, yend_fit = popx

                    # Calculate t50, tmax and tlag from fitted curve
                    x_fit = np.linspace(x.min(), x.max(), 200)
                    y_fit = fitting_eq1(x_fit, *popx)

                    t50 = calculate_t50(x_fit, y_fit)
                    tmax = calculate_tmax(x_fit, y_fit)
                    tlag = calculate_tlag(x_fit, y_fit, t50)

                    fit_data.loc[len(fit_data)] = [well, l, k, th, y0_fit, yend_fit, t50, tmax, tlag]

                except Exception as e:
                    print(f"Could not fit data for well {well}: {e}")
                    fit_data.loc[len(fit_data)] = [well, "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
                        
    update_fit_parameters_display()

def plot_last_selected():
    global df_tidy, fit_data, time_label
    
    if df_tidy is None:
        print("No data loaded yet!")
        return

    if df.empty:
        print("No wells selected!")
        return
    
    last_row = df.iloc[-1]
    wells_str = last_row["wells"]
    protein_name = last_row["protein"]
    protein_conc = last_row["conc"]
    protein_col = last_row.get("colour", None)  # Might not exist

    wells_list = [w.strip() for w in wells_str.split(',')]

    # Create a "Well" identifier in the tidy DataFrame
    df_tidy['Well'] = df_tidy['Well Row'].astype(str).str.strip() + df_tidy['Well Col'].astype(int).astype(str)


    # Optional - Global Normalisation
    normalise_choice = normalise_var.get()
    if normalise_choice == "Global":
        df_for_plot = normalise_data(df_tidy)
    else:  
        df_for_plot = df_tidy

    # Filter for only the selected wells
    plot_df = df_for_plot[df_for_plot['Well'].isin(wells_list)]

    plt.figure(figsize=(10, 5))

    # Decide how to colour
    if protein_col == "":
        palette_name = palette_var.get()
        cmap = plt.colormaps.get_cmap(palette_name).resampled(len(wells_list)) # N distinct colours
        colours = [cmap(i) for i in range(len(wells_list))]
    else:
        colours = [protein_col if protein_col and protein_col.strip() else None] * len(wells_list)

    # Plot each well separately
    for idx, well in enumerate(wells_list):
        sub_df = plot_df[plot_df['Well'] == well]
        if sub_df.empty:
            continue

        # Optional - Local Normalisation
        if normalise_choice == "Local":
            sub_df = normalise_data(sub_df)

        # Making legend customisable to user input
        if protein_name.strip() == "" and protein_conc.strip() == "":
            label = f"{well}"
        elif protein_name.strip() == "":
            label = f"{protein_conc}µM ({well})"
        elif protein_conc.strip() == "":
            label = f"{protein_name} ({well})"
        else:
            label = f"{protein_name} {protein_conc}µM ({well})"

        x = sub_df[time_label].values
        y = sub_df['Fluorescence'].values

        # Plot raw data
        plt.scatter(x, y, s=20, label=label, color=colours[idx], alpha = 0.75)

        # --- Fit curve ---
        equation = equation_var.get()

        if equation != "none":
            x_fit, y_fit, _ = fit_curve(well, x, y)

            if x_fit.size > 0 and y_fit.size > 0:
                # Plot fitted curve with a dashed line and same colour but label it as fit
                plt.plot(x_fit, y_fit, linestyle='--', color=colours[idx])

        make_fit_data(well, x, y)

    # Making plot customisable using user input
    x_label = xlab_entry.get() if xlab_entry.get().strip() else time_label
    y_label = ylab_entry.get() if ylab_entry.get().strip() else "Fluorescence"

    plt.xlabel(x_label, fontsize = 24)
    plt.ylabel(y_label, fontsize = 24)

    # Handling axis limits
    
    # Grab limits (if provided, split by comma or dash)
    xlim_text = xlim_entry.get().strip()
    ylim_text = ylim_entry.get().strip()

    xlim_vals = parse_limits(xlim_text)
    ylim_vals = parse_limits(ylim_text)
    
    # If normalised, set y limits to 0-1 if not specified
    if normalise_choice in ("Global", "Local"):
        if not ylim_vals:
            plt.ylim(0, 1)

    if xlim_vals:
        plt.xlim(*xlim_vals)
    if ylim_vals:
        plt.ylim(*ylim_vals)

    plt.tick_params(axis="both", labelsize = 18)
    # Enable minor ticks
    ax = plt.gca()  # get current axes
    ax.minorticks_on()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()

def plot_all_selected():
    global df_tidy

    if df_tidy is None:
        print("No data loaded yet!")
        return

    if df.empty:
        print("No wells selected!")
        return

    # Build a "Well" identifier in the tidy DataFrame
    df_tidy['Well'] = (
        df_tidy['Well Row'].astype(str).str.strip() +
        df_tidy['Well Col'].astype(int).astype(str)
    )

    # Optional - Global Normalisation
    normalise_choice = normalise_var.get()
    if normalise_choice == "Global":
        df_for_plot = normalise_data(df_tidy)
    else:
        df_for_plot = df_tidy


    plt.figure(figsize=(8, 5))

    palette_name = palette_var.get()

    # Assign colours per row in df
    if palette_name != "manual":
        cmap = plt.colormaps.get_cmap(palette_name).resampled(len(df))
        row_colours = [cmap(i) for i in range(len(df))]
    else:
        row_colours = [
            (row["colour"] if isinstance(row.get("colour", None), str) and row["colour"].strip() else None)
            for _, row in df.iterrows()
        ]

    # Loop over rows in df
    for idx, row in df.iterrows():
        wells = [w.strip() for w in row["wells"].split(',')]
        protein_name = row["protein"]
        protein_conc = row["conc"]
        colour = row_colours[idx]

        #label = f"{protein_name} {protein_conc}µM"

        # Making legend customisable to user input
        if protein_name.strip() == "" and protein_conc.strip() == "":
            label = f"Group {idx + 1}"
        elif protein_name.strip() == "":
            label = f"{protein_conc}µM"
        elif protein_conc.strip() == "":
            label = f"{protein_name}"
        else:
            label = f"{protein_name} {protein_conc}µM"

        # Plot all wells in this row with the same colour and label
        for well in wells:
            sub_df = df_for_plot[df_for_plot["Well"] == well]
            if sub_df.empty:
                continue

            # Optional - Local Normalisation
            if normalise_choice == "Local":
                sub_df = normalise_data(sub_df)

            x = sub_df[time_label].values
            y = sub_df['Fluorescence'].values

            # Plot raw data
            plt.scatter(x, y, s=20, label=label if well == wells[0] else None, color=colour, alpha = 0.75)

            # --- Fit curve ---
            equation = equation_var.get()

            if equation != "none":
                x_fit, y_fit, _ = fit_curve(well, x, y)

                if x_fit.size > 0 and y_fit.size > 0:
                    # Plot fitted curve with a dashed line and same colour but label it as fit
                    plt.plot(x_fit, y_fit, linestyle='--', color=colour)

            make_fit_data(well, x, y)

    # Making plot customisable using user input
    x_label = xlab_entry.get() if xlab_entry.get().strip() else time_label
    y_label = ylab_entry.get() if ylab_entry.get().strip() else "Fluorescence"

    plt.xlabel(x_label, fontsize = 24)
    plt.ylabel(y_label, fontsize = 24)

    # Handling axis limits
    
    # Grab limits (if provided, split by comma or dash)
    xlim_text = xlim_entry.get().strip()
    ylim_text = ylim_entry.get().strip()

    xlim_vals = parse_limits(xlim_text)
    ylim_vals = parse_limits(ylim_text)
    
    # If normalised, set y limits to 0-1 if not specified
    if normalise_choice in ("Global", "Local"):
        if not ylim_vals:
            plt.ylim(0, 1)

    if xlim_vals:
        plt.xlim(*xlim_vals)
    if ylim_vals:
        plt.ylim(*ylim_vals)

    plt.tick_params(axis="both", labelsize = 18)
    # Enable minor ticks
    ax = plt.gca()  # get current axes
    ax.minorticks_on()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()

def save_last_plot():
    # Ask user where to save
    file_path = tk.filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
    )
    if file_path:
        plt.savefig(file_path)
        print(f"Plot saved to {file_path}")

def save_tidy_data():
    global df_tidy

    if df_tidy is None or df_tidy.empty:
        print("No tidy data to save!")
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        title="Save tidy data as..."
    )

    if file_path:
        try:
            df_tidy.to_csv(file_path, index=False)
            print(f"Tidy data saved to {file_path}")
        except Exception as e:
            print(f"Error saving file: {e}")

    if file_path:
        try:
            df_tidy.to_csv(file_path, index=False)
            print(f"Tidy data saved to {file_path}")
        except Exception as e:
            print(f"Error saving file: {e}")

def save_selected_data():
    global df_tidy, df

    if df_tidy is None or df_tidy.empty:
        print("No tidy data to save!")
        return

    if df.empty:
        print("No wells selected!")
        return

    # Create Well identifier in tidy df (same as plotting)
    df_tidy['Well'] = df_tidy['Well Row'].astype(str).str.strip() + df_tidy['Well Col'].astype(int).astype(str)

    # Build list of all wells from df['wells']
    wells_list = [w.strip() for wells_str in df['wells'] for w in wells_str.split(',')]

    # Filter tidy df for selected wells only
    filtered_df = df_tidy[df_tidy['Well'].isin(wells_list)]

    if filtered_df.empty:
        print("No matching wells found in tidy data!")
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        title="Save selected wells data as..."
    )

    if file_path:
        try:
            filtered_df.to_csv(file_path, index=False)
            print(f"Selected wells data saved to {file_path}")
        except Exception as e:
            print(f"Error saving file: {e}")

def save_fit_parameters():
    global fit_data

    if fit_data.empty:
        print("No fit parameters to save!")
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        title="Save fit parameters as..."
    )

    if file_path:
        try:
            fit_data.to_csv(file_path, index=False)
            print(f"Fit parameters saved to {file_path}")
        except Exception as e:
            print(f"Error saving file: {e}")

# ------------------------------------------------------------
# Root window
# ------------------------------------------------------------
root = tk.Tk()
root.geometry("1100x600")
root.title("ThT Analysis")

# ------------------------------------------------------------
# Initial screen
# ------------------------------------------------------------
initial_frame = tk.Frame(root)
initial_frame.pack(fill="both", expand=True)

tk.Label(initial_frame, text="Welcome to ThT Analysis", font=("Arial", 24, "bold")).pack(pady=20)
tk.Label(initial_frame, text="Created by Jack Connor", font= ("Arial", 16)).pack(pady=5)
tk.Button(initial_frame, text="Load Data", command=choose_data_file, font=("Arial", 14), width=15).pack(pady=20)

# ------------------------------------------------------------
# Main screen
# ------------------------------------------------------------
main_frame = tk.Frame(root)

# Input frame at top
input_frame = tk.Frame(main_frame)
input_frame.grid(row=0, column=0, columnspan=12, pady=10)

tk.Label(input_frame, text="Protein Name:").grid(row=0, column=0, sticky="e")
protein_name_entry = tk.Entry(input_frame, width=10)
protein_name_entry.grid(row=0, column=1, padx=5)

tk.Label(input_frame, text="Protein Concentration (µM):").grid(row=0, column=2, sticky="e")
protein_concentration_entry = tk.Entry(input_frame, width=10)
protein_concentration_entry.grid(row=0, column=3, padx=5)

# Variable to store selected palette
palette_var = tk.StringVar(value="viridis")  # Default palette
palette_label = tk.Label(input_frame, text="Colour Palette:")
palette_label.grid(row=0, column=4, padx=5)
palette_dropdown = ttk.Combobox(input_frame, textvariable=palette_var, state="readonly", width = 10)
palette_dropdown['values'] = ("viridis", "plasma", "coolwarm", "manual")
palette_dropdown.grid(row=0, column=5, padx=5)

# Box to manually set the colour
colour_input = tk.Label(input_frame, text="Colour:")
colour_input.grid(row=0, column=6, sticky="e")
colour_entry = tk.Entry(input_frame, width=10)
colour_entry.grid(row=0, column=7, padx=5)

# --- Show/hide logic ---
def on_palette_change(event=None):
    if palette_var.get() == "manual":
        colour_input.grid()
        colour_entry.grid()
    else:
        colour_input.grid_remove()
        colour_entry.grid_remove()

palette_dropdown.bind("<<ComboboxSelected>>", on_palette_change)

# Hide entry initially if not manual
on_palette_change()

# Variable to store selected fit equation
equation_var = tk.StringVar(value="none")  # Default equation
equation_label = tk.Label(input_frame, text="Fit Equation:")
equation_label.grid(row=0, column=8, padx=5)
equation_dropdown = ttk.Combobox(input_frame, textvariable=equation_var, state="readonly", width=5)
equation_dropdown['values'] = ("none", "1", "2", "3")
equation_dropdown.grid(row=0, column=9, padx=5)

# Variable to store selected normalisation
normalise_var = tk.StringVar(value="none")  
normalise_label = tk.Label(input_frame, text="Normalisation:")
normalise_label.grid(row=0, column=10, padx=5)
normalise_dropdown = ttk.Combobox(input_frame, textvariable=normalise_var, state="readonly", width=7)
normalise_dropdown['values'] = ("none", "Global", "Local")
normalise_dropdown.grid(row=0, column=11, padx=5)



tk.Label(input_frame, text="X-axis Label:").grid(row=1, column=0, sticky="e")
xlab_entry = tk.Entry(input_frame, width=10)
xlab_entry.grid(row=1, column=1, padx=5)

tk.Label(input_frame, text="X-axis Limits:").grid(row=2, column=0, sticky="e")
xlim_entry = tk.Entry(input_frame, width=10)
xlim_entry.grid(row=2, column=1, padx=5)

tk.Label(input_frame, text="Y-axis Label:").grid(row=1, column=2, sticky="e")
ylab_entry = tk.Entry(input_frame, width=10)
ylab_entry.grid(row=1, column=3, padx=5)

tk.Label(input_frame, text="Y-axis Limits:").grid(row=2, column=2, sticky="e")
ylim_entry = tk.Entry(input_frame, width=10)
ylim_entry.grid(row=2, column=3, padx=5)




# ----------------------------------------------------------
# Assigning well buttons and placing them in their own frame

# Mapping well names to well buttons
well_button_map = {}       # maps well name -> button
all_well_names = []        # keeps well names in order

# Create a frame just for wells
wells_frame = tk.Frame(main_frame)
wells_frame.grid(row=1, column=0, columnspan=8, pady=10)

# Wells grid
for row in range(8):
    for col in range(12):
        well_name = f"{chr(65 + row)}{col + 1}"
        btn = tk.Button(wells_frame, text=well_name, width=4, height=2, bg="lightgray")
        btn.config(command=lambda b=btn, wn=well_name: toggle_well(b, wn))
        btn.grid(row=row + 1, column=col, padx=2, pady=2)
        well_buttons.append(btn)
        well_button_map[well_name] = btn
        all_well_names.append(well_name)

# # Adding selected wells information
# df_display = tk.Text(wells_frame, width=60, height=25, font=("Arial", 12))
# df_display.grid(row=1, column=13, rowspan=8, padx=2, pady=2)

# ----------------------------------------------------------

# Create a frame just for wells
display_frame = tk.Frame(main_frame)
display_frame.grid(row=1, column=8, columnspan=4, pady=10)

# Adding selected wells information
df_display = tk.Text(display_frame, width=60, height=12, font=("Arial", 12))
df_display.grid(row=0, column=0, padx=2, pady=2)

# Showing fit parameters
fit_display = tk.Text(display_frame, width=60, height=12, font=("Arial", 12))
fit_display.grid(row=1, column=0, padx=2, pady=2)

# ----------------------------------------------------------

# Adding function buttons

# Finalize button
back_btn = tk.Button(main_frame, text="Back", command=back_to_data_selection)
back_btn.grid(row=0, column=12)

# Finalize button
finalize_btn = tk.Button(main_frame, text="Select Wells", command=finalize_selection)
finalize_btn.grid(row=9, column=0, pady=10)

# Adding a plot last button
plot_btn = tk.Button(main_frame, text="Plot Last", command=plot_last_selected)
plot_btn.grid(row=9, column=1, columnspan=1, pady=10)

# Clear last
clear_btn = tk.Button(main_frame, text="Clear Last", command=clear_last)
clear_btn.grid(row=9, column=2, columnspan=1, pady=10)

# Adding a plot all button
plot_all_btn = tk.Button(main_frame, text="Plot All", command=plot_all_selected)
plot_all_btn.grid(row=10, column=1, columnspan=1, pady=10)

# Clear all
clear_all_btn = tk.Button(main_frame, text="Clear All", command=clear_all)
clear_all_btn.grid(row=10, column=2, columnspan=1, pady=10)

# Save last plot
save_plot_btn = tk.Button(main_frame, text="Save Last Plot", command=save_last_plot)
save_plot_btn.grid(row=11, column=1, columnspan=1, pady=10)

# Save selected data
save_selected_btn = tk.Button(main_frame, text="Download Selected", command=save_selected_data)
save_selected_btn.grid(row=9, column=3, columnspan=3, pady=10)

# Save tidy data
save_btn = tk.Button(main_frame, text="Download All", command=save_tidy_data)
save_btn.grid(row=10, column=3, columnspan=3, pady=10)

# Save tidy data
save_fit_btn = tk.Button(main_frame, text="Download Fit Data", command=save_fit_parameters)
save_fit_btn.grid(row=11, column=3, columnspan=3, pady=10)

# ------------------------------------------------------------
root.mainloop()
