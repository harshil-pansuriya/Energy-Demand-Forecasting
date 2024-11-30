import io
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

def read_data(file_path, parse_dates=None):
    try:
        return pd.read_csv(file_path, parse_dates=parse_dates)
    except Exception as e:
        print(f"Error reading data: {str(e)}")
        return None

def plot_to_image(fig):
    # Convert a figure to PIL image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)
