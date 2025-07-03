from bokeh.io import curdoc, export_png
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.layouts import gridplot, Spacer 
from bokeh.models import FactorRange, Range1d, NumeralTickFormatter
import pandas as pd

curdoc().theme = "dark_minimal"

df = pd.read_csv("benchmarkData.csv")

df["average_cpu_time_s"] = df["average_cpu_time_us"] / 1e6
df["average_cuda_time_s"] = df["average_cuda_time"]
df["average_construction_time_s"] = df["average_construction_time_s"]
df["average_accuracy_percent"] = df["average_accuracy"] * 100


df["whom"] = df["whom"].astype(str)
whom_categories = df["whom"].unique().tolist() 
x_range = FactorRange(*whom_categories)

source = ColumnDataSource(df)

height = 375 
width = 650 
bar_width = 0.15 


common_figure_kwargs = {
    "x_range": x_range,
    "height": height,
    "width": width,
    "toolbar_location": None, 
    "tools": "",             
    "outline_line_color": None,
    "background_fill_color": "#202020",
    "border_fill_color": "#202020",
    "min_border_left": 50, 
    "min_border_right": 50,
    "min_border_top": 20,
    "min_border_bottom": 20,
}

color_palette = {
    "cpu": "#89CFF0",  
    "cuda": "#F36AD5",
    "construct": "#F2694D", 
    "accuracy": "#A452F6" 
}

def create_static_bar_plot(title, y_column, y_label, color, y_range_override=None):
    """Helper function to create a standardized static bar plot."""
    p = figure(title=title, **common_figure_kwargs)

    # Apply specific y_range if provided
    if y_range_override:
        p.y_range = y_range_override

    # Add vbar glyph
    p.vbar(x="whom",
           top=y_column,
           width=bar_width,
           color=color,
           source=source,
           line_color="white",
           line_width=1
    )

    p.xaxis.axis_label = "Model Type"
    p.yaxis.axis_label = y_label
    p.xaxis.major_label_orientation = 0.8 
    p.yaxis.formatter = NumeralTickFormatter(format="0.00")
    p.xaxis.axis_label_text_font_size = "12pt" 
    p.xaxis.major_label_text_font_size = "10pt" 
    p.yaxis.major_label_text_font_size = "10pt"
    p.xaxis.axis_label_text_color = "lightgrey" 
    p.yaxis.axis_label_text_color = "lightgrey"
    p.xaxis.major_label_text_color = "lightgrey" 
    p.yaxis.major_label_text_color = "lightgrey"
    p.xaxis.axis_line_color = "lightgrey" 
    p.yaxis.axis_line_color = "lightgrey"
    p.xaxis.major_tick_line_color = "lightgrey" 
    p.yaxis.major_tick_line_color = "lightgrey"


    p.xgrid.grid_line_color = None 
    p.ygrid.grid_line_alpha = 0.2  
    p.ygrid.grid_line_color = "lightgrey" 

    return p

cpu_fig = create_static_bar_plot(
    title="Average CPU Time to Train (Seconds)",
    y_column="average_cpu_time_s",
    y_label="Time (s)",
    color=color_palette["cpu"]
)

cuda_fig = create_static_bar_plot(
    title="Average CUDA Time to Train (Seconds)",
    y_column="average_cuda_time_s",
    y_label="Time (s)",
    color=color_palette["cuda"]
)

construct_fig = create_static_bar_plot(
    title="Average Model Construction Time (Seconds)",
    y_column="average_construction_time_s",
    y_label="Time (s)",
    color=color_palette["construct"]
)

accuracy_fig = create_static_bar_plot(
    title="Average Accuracy (%)",
    y_column="average_accuracy_percent",
    y_label="Accuracy (%)",
    color=color_palette["accuracy"],
    y_range_override=Range1d(0, 100) 
)
accuracy_fig.yaxis.formatter = NumeralTickFormatter(format="0.") 

grid = gridplot([[cpu_fig, cuda_fig], [construct_fig, accuracy_fig]],
                toolbar_location=None, 
                sizing_mode="fixed")
export_png(grid, filename="BenchmarkLinearModel.png")
show(grid)