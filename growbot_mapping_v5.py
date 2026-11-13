"""Pure, hardware-free mapping for the original Nerdland 7 x 5 gantry.

Public plant numbers are now row-major (1-7 on top, then 8-14, ... 29-35).
Physical machine destinations remain exactly those of the original 2024 grid.
"""
from __future__ import annotations

GRID_COLUMNS = 7
GRID_ROWS = 5
PLANTS = GRID_COLUMNS * GRID_ROWS
FIRST_X_MM = 940.0
FIRST_Y_MM = -240.0
COLUMN_PITCH_MM = -120.0
ROW_PITCH_MM = -220.0
HOME_MPOS = (-1499.0, -1.0, -1.0)
AXIS_LIMITS = {"X": (-1499.0, -1.0), "Y": (-1499.0, -1.0)}


def grid_row_col(plant_number: int) -> tuple[int, int]:
    if not 1 <= plant_number <= PLANTS:
        raise ValueError("Plant number must be between 1 and 35")
    return divmod(plant_number - 1, GRID_COLUMNS)


def legacy_plant_number(plant_number: int) -> int:
    """Previous column-major number at this physical position."""
    row, column = grid_row_col(plant_number)
    return 1 + column * GRID_ROWS + row


def plant_offset_mm(plant_number: int) -> tuple[float, float]:
    """Original G54 XY offsets, now addressed by row-major plant ID."""
    row, column = grid_row_col(plant_number)
    return (FIRST_X_MM + column * COLUMN_PITCH_MM,
            FIRST_Y_MM + row * ROW_PITCH_MM)


def plant_machine_xy(plant_number: int) -> tuple[float, float]:
    x, y = plant_offset_mm(plant_number)
    return HOME_MPOS[0] + x, HOME_MPOS[1] + y


def check_xy_limits(x: float, y: float) -> None:
    for axis, target in (("X", x), ("Y", y)):
        low, high = AXIS_LIMITS[axis]
        if not low <= target <= high:
            raise ValueError(
                f"{axis} machine-coordinate target {target:.1f} mm "
                f"outside configured [{low:.1f}, {high:.1f}] mm"
            )


def optris_machine_xy(
    rgb_machine_xy: tuple[float, float],
    optris_offset_mm: tuple[float, float],
) -> tuple[float, float]:
    """Validated post-RGB thermal position (machine coordinates, millimetres)."""
    target = (float(rgb_machine_xy[0]) + float(optris_offset_mm[0]),
              float(rgb_machine_xy[1]) + float(optris_offset_mm[1]))
    check_xy_limits(*target)
    return target
