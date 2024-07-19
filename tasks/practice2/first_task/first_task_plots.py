import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from CSV file
csv_filename = 'qfrc_inverse_data.csv'
df = pd.read_csv(csv_filename)

# Plotting
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, inner='quartile')
plt.xlabel('Joint')
plt.ylabel('Torque')
plt.title('Distribution of Torques in Different Joints')
plt.xticks(ticks=[0, 1, 2, 3], labels=['pitch', 'roll', 'yaw', 'elbow'])
plt.grid(True)
plt.tight_layout()
plt.show()