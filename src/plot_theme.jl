using CairoMakie

const VLFS_THEME = Theme(
  fontsize = 12,
  palette = (color = [:black, :royalblue, :firebrick, :forestgreen, :darkorange],),
  Axis = (
    xgridvisible = false,
    ygridvisible = false,
  ),
  Lines = (
    linewidth = 1.8,
  ),
  Scatter = (
    markersize = 8,
  ),
)