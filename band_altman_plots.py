import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def bland_altman_plot(data1, data2, title, filename):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)
    sd = np.std(diff, axis=0)

    plt.figure(figsize=(8, 4))
    plt.scatter(mean, diff, color='blue', marker='o')
    plt.axhline(md, color='blue', linestyle='--')  # Bias line
    upper_limit = md + 1.96 * sd
    lower_limit = md - 1.96 * sd
    plt.axhline(upper_limit, color='green', linestyle='--')  # Upper agreement limit
    plt.axhline(lower_limit, color='green', linestyle='--')  # Lower agreement limit
    plt.text(mean.max() * 0.5, md, f'Bias = {md:.2f}', verticalalignment='bottom', color='blue')
    plt.text(mean.max() * 0.5, upper_limit, f' Upper LoA = {upper_limit:.2f}', verticalalignment='bottom', color='green')
    plt.text(mean.max() * 0.5, lower_limit, f'Lower LoA = {lower_limit:.2f}', verticalalignment='top', color='green')
    plt.xlim(0, 7)  # Set x-axis range
    plt.ylim(-4, 5)  # Set y-axis range
    plt.title(title)
    plt.xlabel('Mean of Two Measurements')
    plt.ylabel('Difference Between Two Measurements')
    plt.savefig(filename)
    plt.close()

# Load your data here
data = pd.read_csv(r"C:\Users\krupa\OneDrive\Documents\PSU\spring_24\Research\Data_Scoring\new_project\data_processing\data_compiled .csv")

# Example usage with your specific data groups
# Ensure to replace 'data['gpt 3.5']', 'data['gpt 4']', etc., with the correct data columns
bland_altman_plot(data['gpt 3.5'], data['standard'], 'Bland-Altman Plot: GPT 3.5 and Standard', 'bland_altman_plot_gpt_3_5_and_standard.png')
bland_altman_plot(data['gpt 4'], data['standard'], 'Bland-Altman Plot: GPT 4o and Standard', 'bland_altman_plot_gpt_4_and_standard.png')
bland_altman_plot(data['gpt 4-mini'], data['standard'], 'Bland-Altman Plot: GPT 4o Mini and Standard', 'bland_altman_plot_gpt_4_mini_and_standard.png')
