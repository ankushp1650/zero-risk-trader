import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.io as pio
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO


def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)  # Close the figure to release memory
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{img_base64}" />'


import matplotlib.pyplot as plt

def pnl_bar_chart_view(df):
    if df.empty:
        return "<p class='empty-state'>No holdings available.</p>"

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Determine colors based on P&L values
    colors = ['red' if val < 0 else 'green' for val in df['pnl']]

    # Plotting the P&L bar chart on the axes
    bars = ax.bar(df['stock_symbol'], df['pnl'], color=colors)
    ax.set_xlabel('Stock')
    ax.set_ylabel('P&L')
    ax.set_title('Profit Vs Loss')
    ax.axhline(0, color='black', linewidth=0.8)  # Line at y=0 for reference

    # Display the values on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,   # X position: center of the bar
            height,                              # Y position: at the top of the bar
            f'{height:.2f}',                     # Text label: formatted height
            ha='center',                         # Horizontal alignment
            va='bottom' if height >= 0 else 'top',  # Vertical alignment (above for positive, below for negative)
            fontsize=10,                         # Font size for readability
            color='black'                        # Text color
        )

    # Encode the figure to base64 HTML image
    img_html = fig_to_base64(fig)

    # Close the figure to release memory
    plt.close(fig)

    return img_html



def quantity_bar_graph(df):
    if df.empty:
        return "<p class='empty-state'>No holdings available.</p>"
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(df["stock_symbol"], df["quantity"], color='skyblue')

    # Add quantities on top of each bar
    for bar, quantity in zip(bars, df["quantity"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{quantity}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )

    # Labels and title
    ax.set_xlabel("Stock", fontsize=14)
    ax.set_ylabel("Quantity", fontsize=14)
    ax.set_title("Stocks Quantity", fontsize=16)
    plt.xticks(rotation=45, ha='right')

    # Make layout tight and convert figure to HTML
    plt.tight_layout()
    img_html = fig_to_base64(fig)
    return img_html


def pie_chart_view(df, column_name, title):
    if df.empty:
        return "<p class='empty-state'>No holdings available.</p>"
    # Check if DataFrame contains data and the column exists
    if column_name not in df.columns or df.empty:
        return "<p>No data available to display pie chart</p>"

    # Filter negative values and replace NaN and infinite values with zero
    sizes = df[column_name].abs().replace([np.inf, -np.inf], np.nan).fillna(0)

    # If all values are zero after filtering, return an informative message
    if sizes.sum() == 0:
        return f"<p>No valid data for pie chart {title}</p>"

    # Generate the pie chart
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        sizes,
        labels=df['stock_symbol'],
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.tab10.colors
    )
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('equal')  # Equal aspect ratio to make the pie chart circular

    # Convert to base64 for HTML display
    img_html = fig_to_base64(fig)
    return img_html


import matplotlib.pyplot as plt


def line_chart_view(df):
    if df.empty:
        return "<p class='empty-state'>No holdings available.</p>"

    # Calculate Percentage Change
    df['Percentage_Change'] = (df['current_value'] - df['investment_value']) / df['investment_value'] * 100

    # Create a new figure with axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the Percentage Change line
    ax.plot(
        df['stock_symbol'],
        df['Percentage_Change'],
        marker='o',
        color='blue',
        linestyle='-',
        linewidth=2,
        markersize=8,
        label='Percentage Change'
    )

    # Set labels and title
    ax.set_xlabel('Stock', fontsize=14)
    ax.set_ylabel('Percentage Change (%)', fontsize=14)
    ax.set_title('Percentage Change', fontsize=16, fontweight='bold')

    # Customize ticks
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)

    # Add a horizontal line at y=0
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # Add grid, legend, and layout adjustments
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    plt.tight_layout()

    # Convert the figure to base64 HTML format
    img_html = fig_to_base64(fig)

    # Close the figure to release memory
    plt.close(fig)

    return img_html


def bar_chart_view(df):
    if df.empty:
        return "<p class='empty-state'>No holdings available.</p>"
    fig, ax = plt.subplots(figsize=(12, 6))  # Create the figure and axes

    bar_width = 0.25
    x = range(len(df['stock_symbol']))

    # Create bars for Invested and Current Values
    invested_bars = ax.bar(x, df['investment_value'], width=bar_width, color='skyblue', edgecolor='black',
                           label='Invested Value', alpha=0.8)
    current_bars = ax.bar([p + bar_width for p in x], df['current_value'], width=bar_width, color='lightgreen',
                          edgecolor='black', label='Current Value', alpha=0.6)

    # Add labels, title, and axis formatting
    ax.set_xlabel('Stock', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    ax.set_title('Invested vs Current', fontsize=16, fontweight='bold')
    ax.set_xticks([p + bar_width / 2 for p in x])
    ax.set_xticklabels(df['stock_symbol'], rotation=45, ha='right', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    max_value_in_plot = max(df['investment_value'].max(), df['current_value'].max())
    ax.set_ylim(0, max_value_in_plot + 15000)

    # Show differences above bars with sign and color
    for i in range(len(df)):
        invested_value = df['investment_value'][i]
        current_value = df['current_value'][i]
        difference = current_value - invested_value  # Adjusted to reflect gain if current > invested

        # Set color and sign based on difference
        if difference > 0:
            sign = '+'  # Gain
            color = 'green'
        elif difference < 0:
            sign = '-'  # Loss
            color = 'red'
        else:
            sign = ''  # No sign for zero difference
            color = 'black'

        # Annotate difference just above the higher value bar
        max_value = max(invested_value, current_value)
        ax.text(i + bar_width / 2, max_value + 5000, f'Diff: {sign}{abs(difference):.2f}',
                ha='center', va='bottom', fontsize=10, color=color)

    # Encode the figure to base64 HTML image
    img_html = fig_to_base64(fig)
    return img_html


def generate_graphs(table2):
    # Ensure the relevant columns are of numeric type
    table2['Close'] = pd.to_numeric(table2['Close'], errors='coerce')
    table2['Open'] = pd.to_numeric(table2['Open'], errors='coerce')
    table2['High'] = pd.to_numeric(table2['High'], errors='coerce')
    table2['Low'] = pd.to_numeric(table2['Low'], errors='coerce')
    table2['Volume'] = pd.to_numeric(table2['Volume'], errors='coerce')

    # Calculate moving averages
    table2['SMA20'] = table2['Close'].rolling(window=20).mean()
    table2['SMA50'] = table2['Close'].rolling(window=50).mean()

    # Prepare data for the main candlestick chart with hovertext
    candlestick = go.Candlestick(
        x=table2['Date'],
        open=table2['Open'],
        high=table2['High'],
        low=table2['Low'],
        close=table2['Close'],
        name="Candlestick",
        increasing_line_color='green',
        decreasing_line_color='red',
        hovertext=(
                "<b>Date:</b> %{x}<br>" +
                "<b>Open:</b> %{open:.2f}<br>" +
                "<b>High:</b> %{high:.2f}<br>" +
                "<b>Low:</b> %{low:.2f}<br>" +
                "<b>Close:</b> %{close:.2f}<extra></extra>"
        )
    )

    # Prepare data for the moving averages
    sma20 = go.Scatter(
        x=table2['Date'],
        y=table2['SMA20'],
        mode='lines',
        name='20-Day SMA',
        line=dict(color='orange', width=2),
        hoverinfo="none"  # Disable hover for SMA lines
    )

    sma50 = go.Scatter(
        x=table2['Date'],
        y=table2['SMA50'],
        mode='lines',
        name='50-Day SMA',
        line=dict(color='blue', width=2),
        hoverinfo="none"
    )

    # Prepare data for the brush chart (small candlestick)
    brush_candlestick = go.Candlestick(
        x=table2['Date'],
        open=table2['Open'],
        high=table2['High'],
        low=table2['Low'],
        close=table2['Close'],
        name="Brush Candlestick",
        increasing_line_color='gold',
        decreasing_line_color='tomato',
        opacity=0.5
    )

    # Create a subplot with 2 rows, one for the main candlestick and one for the brush chart
    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.1,
                           row_heights=[0.8, 0.2],
                           specs=[[{"type": "candlestick"}],
                                  [{"type": "candlestick"}]])

    # Add the main candlestick and moving averages to the first row
    fig.add_trace(candlestick, row=1, col=1)
    fig.add_trace(sma20, row=1, col=1)
    fig.add_trace(sma50, row=1, col=1)

    # Add the brush candlestick to the second row
    fig.add_trace(brush_candlestick, row=2, col=1)

    # Update layout for the figure
    fig.update_layout(
        title=dict(
            text="Candlestick Chart with Synced Brush Chart",
            font=dict(size=24, family="Arial, Bold")
        ),
        xaxis_title=dict(
            text="Date",
            font=dict(size=18, family="Arial, Bold")
        ),
        yaxis_title=dict(
            text="Price",
            font=dict(size=18, family="Arial, Bold")
        ),
        height=700,
        xaxis_rangeslider_visible=False,  # Disable the default range slider
        hovermode='x',  # Hover mode for better interactivity
        legend=dict(
            x=0,
            y=1,
            traceorder='normal',
            font=dict(size=14, family="Arial, Bold")
        ),
        xaxis=dict(
            rangeslider=dict(visible=False),  # Keeping the default range slider disabled
            type="date"
        )
    )

    # Customize second x-axis (brush chart) to highlight its range selection feature
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],  # Hide weekends in both main and brush charts
        row=2, col=1,
        rangeslider=dict(visible=True)  # Enable range slider on the brush chart
    )

    # Drop the SMA columns to keep the original DataFrame clean
    table2.drop(columns=['SMA20', 'SMA50'], inplace=True)

    # Convert the figure to an HTML string
    fig_html = pio.to_html(fig, full_html=False)

    return fig_html
