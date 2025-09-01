import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from loguru import logger


class Visual():
    def __init__(self,
                 x=None, y=None, z=None,
                 title="Untitled",
                 xaxis_title="x",
                 yaxis_title="y",
                 zaxis_title="z",
                 method=None,
                 width=600,
                 height=400,
                 subplots=False,
                 rows=2,
                 cols=1,
                 ) -> None:

        self.x = x
        self.y = y
        self.z = z
        self.title = title
        self.xaxis_title = xaxis_title
        self.yaxis_title = yaxis_title
        self.width = width
        self.height = height
        self.subplots = subplots
        if subplots:
            self.fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True)
        else:
            self.fig = go.Figure()
        self.fig.update_layout(
            width=self.width,
            height=self.height,
            xaxis_title=self.xaxis_title,
            yaxis_title=self.yaxis_title,
            title=self.title,
            font_family="Arial",
            font_size=18,
        )

    @property
    def figure(self):
        return self.fig

    def update_xaxes(self, range):
        self.fig.update_xaxes(range=range)

    def update_yaxes(self, range, reverse=False):
        self.fig.update_yaxes(range=range)

    def reverse_y(self):
        self.fig.update_yaxes(autorange="reversed")

    def update_yaxes(self, range):
        self.fig.update_yaxes(range=range)

    def heatmap(self, colorscale="Viridis", update=False, styles=None, animates=None, caxis=None, **kwargs):
        assert self.z is not None, "z is None"
        trace = go.Heatmap(z=self.z, x=self.x, y=self.y, colorscale=colorscale)
        self.fig.add_trace(trace)
        if caxis is not None:
            assert "cmin" in caxis.keys() and "cmax" in caxis.keys(
            ), "cmin and cmax must be in caxis"

            self.fig.update_coloraxes(cmin=caxis["cmin"], cmax=caxis["cmax"])
            self.fig["data"][0]["zmin"] = caxis["cmin"]
            self.fig["data"][0]["zmax"] = caxis["cmax"]
        if update:
            assert styles is not None or animates is not None, "styles and animates are None"
            self._update(styles, animates, **kwargs)

    def scatter(self, extra_y=None, extra_x=None, marker_size=30, marker_color="red", update=False, styles=None, animates=None, **kwargs):
        if extra_x is None and extra_y is None:
            trace = go.Scatter(x=self.x,
                               y=self.y,
                               mode="markers",
                               marker=dict(size=marker_size,
                                           color=marker_color,)
                               )

        self.fig.add_trace(trace)
        if update:
            assert styles is not None or animates is not None, "styles and animates are None"
            self._update(styles, animates, **kwargs)

    def line(self, axis=0, threshold=None, gt=None, bl=None, update=False, styles=None, animates=None, **kwargs):
        assert self.y is not None, "y is None"
        if len(self.y.shape) == 2:
            for i in range(self.y.shape[axis]):
                trace = go.Scatter(y=np.take(
                    self.y, i, axis=axis), x=self.x, mode="lines")
                assert trace is not None, "trace is None"
                self.fig.add_trace(trace)
        elif len(self.y.shape) == 1:
            trace = go.Scatter(y=self.y, x=self.x, mode="lines")
            assert trace is not None, "trace is None"
            self.fig.add_trace(trace)

        if threshold is not None:
            self.fig.add_hline(y=threshold, line_dash="dash", line_color="red")

        if gt is not None:
            self.fig.add_scatter(
                x=self.x, y=gt, mode="lines", line_color="red", name="gt")
        if bl is not None:
            self.fig.add_scatter(
                x=self.x, y=bl, mode="lines", line_color="orange", name="Baseline", line_dash="dash")
        # trace = go.Scatter(y=self.y, x=self.x, mode="lines")
        # self.fig.add_trace(trace)
        if update:
            self._update(styles=styles, animates=animates, **kwargs)

    def histogram(self, update=False, styles=None, animates=None, add_Box=False, add_Violin=False, **kwargs):
        assert self.x is not None, "x is None"
        trace = go.Histogram(x=self.x,
                             autobinx=False,
                             nbinsx=5)
        if self.subplots:
            self.fig.add_trace(trace, row=1, col=1)
            if add_Box:
                self.fig.add_trace(go.Box(x=self.x,
                                          marker_symbol='line-ns-open',
                                          boxpoints='all',
                                          name="rug"), row=2, col=1)
            elif add_Violin:
                self.fig.add_trace(go.Violin(x=self.x,
                                             box_visible=True,
                                             meanline_visible=True,
                                             name="violin"), row=2, col=1)
        else:
            self.fig.add_trace(trace)
        if update:
            self._update(styles=styles, animates=animates, **kwargs)

    def violin(self, update=False, styles=None, animates=None, **kwargs):
        assert self.x is not None, "x is None"
        trace = go.Violin(x=self.x,
                          box_visible=True,
                          meanline_visible=True,
                          name="violin")
        self.fig.add_trace(trace)
        if update:
            self._update(styles=styles, animates=animates, **kwargs)

    # def scatter(self, update=False, styles=None, animates=None, **kwargs):
    #     assert self.y is not None, "y is None"
    #     trace = go.Scatter(x=self.x, y=self.y, mode="markers")
    #     self.fig.add_trace(trace)
    #     if update:
    #         self._update(styles=styles, animates=animates, **kwargs)

    def _update(self, styles, animates, **kwargs):
        self.fig.update_layout(
            updatemenus=self.create_update_menus(
                **dict(styles=styles, animates=animates)),
        )
        if animates is not None and "frames" in kwargs.keys() and kwargs["frames"] is not None:
            self.fig["frames"] = kwargs["frames"]

    def _create_restyle_button(self, styles):
        buttons = [RestyleButton().restyle_button for _ in range(len(styles))]
        for idx, button in enumerate(buttons):
            button["args"] = ["type", styles[idx]]
            button["label"] = styles[idx]
        return list(buttons)

    def _create_animate_button(self, animates):
        animate_buttons = []
        if "Play" in animates:
            play_button = AnimatePlayButton().animate_button
            animate_buttons.append(play_button)
        if "Pause" in animates:
            pause_button = AnimatePauseButton().animate_button
            animate_buttons.append(pause_button)
        return list(animate_buttons)

    def create_update_menus(self, **kwargs):
        buttons = []
        if "styles" in kwargs.keys() and kwargs["styles"] is not None:
            restyle_button = self._create_restyle_button(kwargs["styles"])
            buttons.append(restyle_button)
        if "animates" in kwargs.keys() and kwargs["animates"] is not None:
            animate_button = self._create_animate_button(kwargs["animates"])
            buttons.append(animate_button)

        from itertools import chain
        buttons = list(chain.from_iterable(buttons))

        update_menu = dict(
            type="buttons",
            buttons=buttons,
        )
        update_menu["direction"] = "left"
        update_menu["showactive"] = False
        update_menu["x"] = 0.1
        update_menu["xanchor"] = "left"
        update_menu["y"] = 0
        update_menu["yanchor"] = "top"

        self.update_menu = [update_menu]
        return self.update_menu


class RestyleButton():
    def __init__(self):
        self.restyle_button = dict(method="restyle")


class AnimateButton():
    def __init__(self) -> None:
        self.animate_button = dict(method="animate")


class AnimatePlayButton(AnimateButton):
    def __init__(self) -> None:
        super().__init__()
        self.animate_button["args"] = [None]
        self.animate_button["label"] = "Play"


class AnimatePauseButton(AnimateButton):
    def __init__(self) -> None:
        super().__init__()
        self.animate_button["args"] = [[None], {"frame": {"duration": 0, "redraw": False},
                                                "mode": "immediate",
                                                "transition": {"duration": 0}}]
        self.animate_button["label"] = "Pause"
