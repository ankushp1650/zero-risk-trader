import pandas as pd
import matplotlib.pyplot as plt

# Sample data based on your table
data = {
    'Stock': ['RELIANCE.BSE', 'BHARTIARTL.BSE', 'TCS.BSE', 'ICICIBANK.BSE', 'HDFCBANK.BSE'],
    'P&L': [-231.91, -510, -1729.53, -323.2, 1500]
}
df = pd.DataFrame(data)
def bar_chart_view(df):
    if df.empty:
        return "<p class='empty-state'>No holdings available.</p>"
    fig, ax = plt.subplots(figsize=(12, 6))  # Create the figure and axes

    bar_width = 0.25
    # Plotting the P&L bar chart
    plt.figure(figsize=(10, 6))
    colors = ['red' if val < 0 else 'green' for val in df['P&L']]
    plt.bar(df['Stock'], df['P&L'], color=colors)
    plt.xlabel('Stock')
    plt.ylabel('P&L')
    plt.title('Profit & Loss (P&L) for Each Stock')
    plt.axhline(0, color='black', linewidth=0.8)  # Line at y=0 for reference
    plt.show()
    # Encode the figure to base64 HTML image
    img_html = fig_to_base64(fig)
    return img_html