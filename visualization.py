import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any, Union

def visualize_state(
    amps: Dict[Tuple[int, ...], complex],
    figsize: Optional[Tuple[float, float]] = None,
    cmap: Union[str, Any, None] = None,
    x_fontsize: int = 9,
    bar_width: float = 0.8,
    ylim: Tuple[float, float] = (0.0, 1.05)
) -> Tuple[plt.Figure, plt.Axes]:
    """Visualizes a bar chart of quantum state amplitude magnitudes.

    Converts a dictionary of Fock state tuples and their complex amplitudes
    into a formatted histogram, using Dirac notation for the x-axis labels.

    Args:
        amps: A dictionary mapping state tuples (e.g., (1, 0, 1)) to their
            complex probability amplitudes.
        figsize: A tuple defining the (width, height) of the figure in inches.
            If None, the width scales dynamically with the number of states.
        cmap: A Matplotlib colormap string (e.g., 'Blues') or colormap object. 
            Defaults to 'Blues' if None is provided.
        x_fontsize: The font size for the state labels on the x-axis.
        bar_width: The width of the bars in the histogram.
        ylim: A tuple representing the lower and upper bounds of the y-axis.

    Returns:
        A tuple containing the generated Matplotlib Figure and Axes objects.
    """
    if cmap is None:
        cmap = plt.colormaps.get_cmap("Blues")
    elif isinstance(cmap, str):
        cmap = plt.colormaps.get_cmap(cmap)

    amps_array = []
    basis_elements = []

    for state, amp in amps.items():
        amps_array.append(np.abs(amp))
        state_str = "".join(str(photon_count) for photon_count in state)
        basis_elements.append(f"$| {state_str} \\rangle$")

    num_states = len(basis_elements)

    if figsize is None:
        fig_width = max(4.0, num_states / 5.0)
        fig, ax = plt.subplots(figsize=(fig_width, 3.5))
    else:
        fig, ax = plt.subplots(figsize=figsize)

    x_positions = np.arange(num_states)
    
    ax.bar(
        x_positions, 
        amps_array, 
        facecolor=cmap(0.32), 
        edgecolor=cmap(0.8), 
        width=bar_width
    )
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(basis_elements, fontsize=x_fontsize, rotation=90)
    ax.set_ylim(ylim)
    ax.set_ylabel("Amplitude Magnitude")
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def create_mini_mzi_grid(
    figsize: Tuple[int, int] = (2, 2),
    n_cols: int = 20,
    even_rows: int = 10,
    odd_rows: int = 9,
    vertical_gap_factor: float = 0.0,
    highlight_groups: Optional[List[Dict[str, Any]]] = None,
    show_colorbar: bool = True,
    cbar_label: Optional[str] = None,
    cbar_vmin: Optional[float] = None,
    cbar_vmax: Optional[float] = None,
    unused_color: str = "#DFE0E2",  # Default Nature-style grey
    unused_alpha: float = 0.2       # Transparency for unused blocks
) -> plt.Figure:
    """Creates a miniature Mach-Zehnder Interferometer (MZI) grid visualization.
    
    Args:
        figsize: Tuple defining figure dimensions.
        n_cols: Number of columns in the grid.
        even_rows: Number of MZIs in even-indexed columns.
        odd_rows: Number of MZIs in odd-indexed columns.
        vertical_gap_factor: Vertical spacing between devices.
        highlight_groups: List of dicts. If a device (x, y) appears in multiple
                          dicts, the LATEST one in the list overwrites previous ones.
                          Can accept 'cmap' for scaled colors or 'color' for solid colors.
        show_colorbar: Boolean toggle for rendering the colorbar.
        cbar_label: Label for the colorbar.
        cbar_vmin: Minimum value for the colorbar scale.
        cbar_vmax: Maximum value for the colorbar scale.
        unused_color: Hex color string for devices not in highlight_groups.
        unused_alpha: Alpha (transparency) level for unused devices (0.0 to 1.0).
        
    Returns:
        The generated Matplotlib Figure.
    """
    fig = plt.figure(figsize=figsize)

    # 1. Pre-calculate the final color for every highlighted device
    final_device_colors = {}
    primary_sm = None

    if highlight_groups:
        for idx, group in enumerate(highlight_groups):
            data = group.get('data', {})
            specific_color = group.get('color')

            if specific_color:
                for (mzi_x, mzi_y), val in data.items():
                    col, row = mzi_x - 1, mzi_y - 1
                    final_device_colors[(col, row)] = specific_color
            else:
                cmap_obj = plt.get_cmap(group.get('cmap', plt.cm.viridis))
                vals = list(data.values())
                if not vals: continue

                g_min = min(vals)
                g_max = max(vals)

                for (mzi_x, mzi_y), val in data.items():
                    col, row = mzi_x - 1, mzi_y - 1
                    if g_max == g_min:
                        color = cmap_obj(0.5)
                    else:
                        color = cmap_obj((val - g_min) / (g_max - g_min))
                    final_device_colors[(col, row)] = color

                if primary_sm is None and show_colorbar:
                    vmin = cbar_vmin if cbar_vmin is not None else g_min
                    vmax = cbar_vmax if cbar_vmax is not None else g_max
                    norm = plt.Normalize(vmin=vmin, vmax=vmax)
                    primary_sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
                    primary_sm.set_array([])

    # 2. Grid Construction Logic
    base_subplot_height = 1.0 / (even_rows + (even_rows - 1) * vertical_gap_factor)
    base_subplot_width = 1.0 / n_cols

    for col in range(n_cols):
        is_even = col % 2 == 0
        n_rows = even_rows if is_even else odd_rows
        subplot_height, subplot_width = base_subplot_height, base_subplot_width

        for row in range(n_rows):
            if is_even:
                bottom = 1 - (row + 1) * subplot_height - row * subplot_height * vertical_gap_factor
            else:
                bottom = 1 - (row + 1) * subplot_height - row * subplot_height * vertical_gap_factor - subplot_height / 2

            left = col * subplot_width
            is_active = (col, row) in final_device_colors
            facecolor = final_device_colors.get((col, row), unused_color)
            current_alpha = 1.0 if is_active else unused_alpha
            edgecolor = "k" if is_active else "#A0A0A0"

            rect = patches.Rectangle(
                (left, bottom), subplot_width, subplot_height,
                facecolor=facecolor, edgecolor=edgecolor, lw=0.5,
                alpha=current_alpha,
                zorder=2 if is_active else 1
            )
            fig.add_artist(rect)

    # 4. Colorbar Logic
    if show_colorbar and primary_sm:
        ax = fig.gca()
        cax = ax.inset_axes([1.08, 0.1, 0.03, 0.8])
        plt.colorbar(primary_sm, cax=cax, label=cbar_label)

    plt.axis("off")
    return fig


# =============================================================================
# EXAMPLE USAGE / TESTING BLOCK
# =============================================================================
# if __name__ == "__main__":
#     H_DEVICES = {(1, 5): 1.0, (15, 5): 1.0}
#
#     TARGET_DEVICES = {(2, 3): 1.0, (2, 4): 1.0,
#                       (3, 4): 1.0,
#                       (4, 3): 1.0, (4, 4): 1.0,
#                       (5, 4): 1.0}
#
#     GENERATOR_DEVICES = {(2, 5): 1.0, (2, 6): 1.0,
#                          (3, 6): 1.0,
#                          (4, 5): 1.0, (4, 6): 1.0,
#                          (5, 6): 1.0}
#
#     DISCRIMINATOR_DEVICES = {(5, 5): 1.0,
#                              (6, 4): 1.0, (6, 5): 1.0,
#                              (7, 4): 1.0, (7, 5): 1.0, (7, 6): 1.0,
#                              (8, 3): 1.0, (8, 4): 1.0, (8, 5): 1.0, (8, 6): 1.0,
#                              (9, 4): 1.0, (9, 5): 1.0, (9, 6): 1.0,
#                              (10, 4): 1.0, (10, 5): 1.0,
#                              (11, 5): 1.0, }
#
#     INVERSE_UP_DEVICES = {(11, 4): 1.0,
#                           (12, 3): 1.0, (12, 4): 1.0,
#                           (13, 4): 1.0,
#                           (14, 3): 1.0, (14, 4): 1.0,
#                           # (15, 4): 1.0
#                           }
#     
#     INVERSE_DOWN_DEVICES = {(11, 6): 1.0,
#                             (12, 5): 1.0, (12, 6): 1.0,
#                             (13, 6): 1.0,
#                             (14, 5): 1.0, (14, 6): 1.0,
#                             # (15, 4): 1.0
#                             }
#
#     IDENTITY_DEVICES = {(3, 5): 1.0,
#                         (13, 5): 1.0}
#
#     # cmap_b = sns.color_palette("Blues", as_cmap=True)
#     # cmap_b = sns.color_palette("mako", as_cmap=True)
#     cmap_b = sns.color_palette("vlag", as_cmap=True)
#     cmap_vlag = sns.light_palette("seagreen", as_cmap=True)
#     # cmap_g = sns.color_palette("crest", as_cmap=True)
#     # cmap_g = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, as_cmap=True)
#     # cmap_vlag = sns.diverging_palette(220, 20, as_cmap=True)
#     
#     groups = [
#         {'data': H_DEVICES, 'color': '#F9D877'},
#         {'data': GENERATOR_DEVICES, 'color': cmap_vlag(0.5)},
#         {'data': TARGET_DEVICES, 'color': 'palevioletred'},
#         {'data': DISCRIMINATOR_DEVICES, 'color': 'cornflowerblue'},
#         {'data': INVERSE_UP_DEVICES, 'color': cmap_vlag(0.5)},
#         {'data': INVERSE_DOWN_DEVICES, 'color': cmap_vlag(0.5)},
#         {'data': IDENTITY_DEVICES, 'color': '#DFE0E2'},
#     ]
#
#     # Standard Nature/Okabe-Ito scientific hex colors
#     nature_grey = '#DFE0E2'
#
#     # Alternative definitions:
#     # groups = [
#     #     {'data': H_DEVICES,             'color': '#F9D877'},
#     #     {'data': GENERATOR_DEVICES,     'color': plt.cm.Greens(0.65, alpha=0.6)},
#     #     {'data': TARGET_DEVICES,        'color': plt.cm.Oranges(0.7, alpha = 0.6)},
#     #     {'data': DISCRIMINATOR_DEVICES, 'color': plt.cm.BuPu_r(0.5, alpha = 1)},
#     #     {'data': INVERSE_UP_DEVICES,    'color': plt.cm.Greens(0.65, alpha=0.6)},
#     #     {'data': INVERSE_DOWN_DEVICES,  'color': plt.cm.Greens(0.65, alpha=0.6)},
#     #     {'data': IDENTITY_DEVICES,      'color': nature_grey},
#     # ]
#     
#     plt.rcParams['figure.dpi'] = 600
#     fig = create_mini_mzi_grid(n_cols=20, highlight_groups=groups, show_colorbar=False)
#     fig.savefig("GAN_grid_4_mode_sample.png", bbox_inches='tight')