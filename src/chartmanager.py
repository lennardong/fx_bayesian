import itertools
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict

import plotly.graph_objects as go
import seaborn as sns
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

from src import *
from src.datamanager import AlphaVantageDataManager


@dataclass
class CandleStickCols:
    open: str
    high: str
    low: str
    close: str


class ChartManager:
    def __init__(self, df, title="FX Chart"):
        self.df = df
        self.fig_fx = go.Figure()
        self.fig_0to100 = go.Figure()
        self.title = title

    def add_candlestick(self, cols: CandleStickCols):
        self.fig_fx.add_trace(
            go.Candlestick(
                x=self.df.index,
                open=self.df[cols.open],
                high=self.df[cols.high],
                low=self.df[cols.low],
                close=self.df[cols.close],
                name="Candlesticks",
                whiskerwidth=0.8,
                line_width=1.5,
            )
        )
        return self

    def add_linetrace(
        self,
        data: Dict[datetime, float],
        title: str,
        colour: str,
        fig_0to100: bool = False,
    ):
        figure = self.fig_0to100 if fig_0to100 else self.fig_fx
        figure.add_trace(
            go.Scatter(
                x=list(data.keys()),
                y=list(data.values()),
                mode="lines",
                line=dict(color=colour, width=1),
                name=title,
            )
        )
        return self

    def add_vline_date(self, key_dates: Dict[date, str]):
        y_min = self.df[["open", "high", "low", "close"]].min().min()
        y_max = self.df[["open", "high", "low", "close"]].max().max()
        for date_key, label in key_dates.items():
            date_datetime = datetime.combine(date_key, datetime.min.time())
            self.fig_fx.add_trace(
                go.Scatter(
                    x=[date_datetime, date_datetime],
                    y=[y_min, y_max],
                    mode="lines+text",
                    line=dict(color="black", width=0.5, dash="dot"),
                    name=label,
                    text=["", label],
                    showlegend=False,
                )
            )
        return self

    def add_today_line(self):
        today = datetime.now().date()
        today_datetime = datetime.combine(today, datetime.min.time())

        # Get the y-axis range from the DataFrame
        y_min = self.df[["open", "high", "low", "close"]].min().min()
        y_max = self.df[["open", "high", "low", "close"]].max().max()

        self.fig_fx.add_trace(
            go.Scatter(
                x=[today_datetime, today_datetime],
                y=[y_min, y_max],
                mode="lines+text",
                line=dict(color="black", width=2),
                name="TODAY",
                text=["", "TODAY"],
                showlegend=False,
            )
        )
        return self

    def add_weekend_highlights(self, fig: Figure):
        # Get the date range from the data
        date_range = pd.date_range(start=self.df.index.min(), end=self.df.index.max())

        # Define Saturday (5) and Sunday (6)
        FRIDAY = 4
        SATURDAY = 5
        SUNDAY = 6

        # Identify weekends
        weekends = date_range[date_range.dayofweek.isin([FRIDAY, SATURDAY])]

        # Group consecutive weekend days
        weekend_groups = []
        current_group = []
        for date in weekends:
            # If the current group is empty or the date is consecutive, add to the current group
            if not current_group or (date - current_group[-1]).days == 1:
                current_group.append(date)
            else:
                # If not consecutive, start a new group
                weekend_groups.append(current_group)
                current_group = [date]

        # Add the last group if it exists
        if current_group:
            weekend_groups.append(current_group)

        # Add highlight for each weekend group
        for weekend in weekend_groups:
            # Highlight from the start of Saturday to the end of Sunday
            fig.add_vrect(
                x0=weekend[0],  # Start of Saturday
                x1=weekend[-1],  # End of Sunday
                fillcolor="white",
                opacity=1,
                layer="below",
                line_width=0,
            )

        return fig

    def show(self):
        # Create a figure with 1 row and 2 columns
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(None, None),
            row_heights=[0.3, 0.7],  # 30% for the first chart, 70% for the second
            shared_xaxes=True,  # This will make all x-axes zoom and scale together
            vertical_spacing=0.05,  # Adjust the spacing between subplots as needed
        )

        # Add traces from fig_0to100 to the second subplot
        for trace in self.fig_0to100.data:
            fig.add_trace(trace, row=1, col=1)

        # Add traces from fig_fx to the first subplot
        for trace in self.fig_fx.data:
            fig.add_trace(trace, row=2, col=1)

        x_min = min(
            min(trace.x) if isinstance(trace.x, tuple) else trace.x.min()
            for trace in fig.data
            if trace.x is not None
        )
        x_max = max(
            max(trace.x) if isinstance(trace.x, tuple) else trace.x.max()
            for trace in fig.data
            if trace.x is not None
        )
        # Add Weenend highlights
        fig = self.add_weekend_highlights(fig)
        # Update layout for both subplots

        fig.update_layout(height=800, width=1500, title_text=self.title)
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )

        fig.update_xaxes(showgrid=False, range=[x_min, x_max], row=1, col=1)
        fig.update_xaxes(showgrid=False, range=[x_min, x_max], row=2, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Price", row=2, col=1)

        # Show the combined figure
        fig.show()


if __name__ == "__main__":
    df = AlphaVantageDataManager("SGD", "MYR").df
    c_cols = CandleStickCols(
        open=ColName.OPEN, high=ColName.HIGH, low=ColName.LOW, close=ColName.CLOSE
    )
    key_dates = {
        date(2024, 6, 1): "Important Date 1",
        date(2024, 9, 15): "Important Date 2",
        date(2024, 12, 15): "Important Date 3",
    }

    chart = ChartManager(df, "SGD/MYR Exchange Rate")
    chart.add_candlestick(c_cols)
    # chart.add_moving_average({date: df["MA"].loc[date] for date in df.index})
    chart.add_vline_date(key_dates)
    chart.add_today_line()
    chart.show()
