"""Simulate a dual-beam transmission sensor and a single-slit pattern."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FloatArray = NDArray[np.float64]

SEED = 42
WAVELENGTH_M = 650e-9
SLIT_WIDTH_M = 80e-6
SCREEN_DISTANCE_M = 0.60
CUVETTE_LENGTH_M = 0.01
EXTINCTION_PER_M = 160.0  # At relative concentration 1.
MODULATION_HZ = 500
FLICKER_HZ = 120
CYCLE_COUNT = 100
CALIBRATION_CONCENTRATIONS = (0.25, 0.50, 1.00)
UNKNOWN_CONCENTRATION = 0.65
UNKNOWN_LASER_SCALE = 0.80
MIN_REFERENCE_SIGNAL_V = 0.01


@dataclass(frozen=True)
class SensorTrace:
    sample_on: FloatArray
    sample_off: FloatArray
    reference_on: FloatArray
    reference_off: FloatArray


@dataclass(frozen=True)
class TransmissionReading:
    name: str
    concentration: float
    laser_scale: float
    raw_transmittance: float
    corrected_transmittance: float


def true_transmittance(concentration: float) -> float:
    return float(np.exp(-EXTINCTION_PER_M * CUVETTE_LENGTH_M * concentration))


def diffraction_phase(position_m: FloatArray) -> FloatArray:
    angle = np.arctan(position_m / SCREEN_DISTANCE_M)
    return np.pi * SLIT_WIDTH_M * np.sin(angle) / WAVELENGTH_M


def diffuse_intensity(position_m: FloatArray, concentration: float) -> FloatArray:
    # A phenomenological broad halo for the illustration.
    return 0.12 * concentration * np.exp(-0.5 * (position_m / 0.007) ** 2)


def diffraction_intensity(position_m: FloatArray, concentration: float) -> FloatArray:
    phase = diffraction_phase(position_m)
    coherent_intensity = true_transmittance(concentration) * np.sinc(phase / np.pi) ** 2
    return coherent_intensity + diffuse_intensity(position_m, concentration)


def central_taylor_intensity(
    position_m: FloatArray, concentration: float
) -> FloatArray:
    phase = diffraction_phase(position_m)
    if np.any(np.abs(phase) > 1.0):
        raise ValueError("The central diffraction approximation requires |phase| <= 1.")
    phase_squared = phase**2
    intensity_factor = 1.0 - phase_squared / 3.0 + 2.0 * phase_squared**2 / 45.0
    return (
        true_transmittance(concentration) * intensity_factor
        + diffuse_intensity(position_m, concentration)
    )


def fringe_visibility(concentration: float) -> float:
    first_zero_angle = np.arcsin(WAVELENGTH_M / SLIT_WIDTH_M)
    first_zero_m = SCREEN_DISTANCE_M * np.tan(first_zero_angle)
    center, first_zero = diffraction_intensity(
        np.array([0.0, first_zero_m]), concentration
    )
    return float((center - first_zero) / (center + first_zero))


def measure(
    concentration: float, laser_scale: float, rng: np.random.Generator
) -> SensorTrace:
    time_on = np.arange(CYCLE_COUNT) / MODULATION_HZ
    time_off = time_on + 0.5 / MODULATION_HZ
    ambient_phase = rng.uniform(0.0, 2.0 * np.pi)
    flicker_on = np.sin(2.0 * np.pi * FLICKER_HZ * time_on + ambient_phase)
    flicker_off = np.sin(2.0 * np.pi * FLICKER_HZ * time_off + ambient_phase)
    noise = rng.normal(0.0, 0.003, size=(4, CYCLE_COUNT))

    sample_on = 0.19 + 0.035 * flicker_on + noise[0]
    sample_on += 1.2 * laser_scale * true_transmittance(concentration)
    sample_off = 0.19 + 0.035 * flicker_off + noise[1]
    reference_on = 0.24 + 0.045 * flicker_on + noise[2] + 0.9 * laser_scale
    reference_off = 0.24 + 0.045 * flicker_off + noise[3]
    return SensorTrace(sample_on, sample_off, reference_on, reference_off)


def sample_reference_ratio(trace: SensorTrace) -> float:
    reference_signal = float(np.mean(trace.reference_on - trace.reference_off))
    if reference_signal <= MIN_REFERENCE_SIGNAL_V:
        raise ValueError("Reference signal is too weak for a transmission estimate.")
    sample_signal = float(np.mean(trace.sample_on - trace.sample_off))
    if sample_signal <= 0.0:
        raise ValueError("Sample signal is too weak for a transmission estimate.")
    return sample_signal / reference_signal


def collect_readings() -> list[TransmissionReading]:
    rng = np.random.default_rng(SEED)
    blank = measure(0.0, 1.0, rng)
    blank_ratio = sample_reference_ratio(blank)
    readings = [TransmissionReading("blank", 0.0, 1.0, 1.0, 1.0)]
    cases = [
        (f"calibration_{concentration:.0%}", concentration, 1.0)
        for concentration in CALIBRATION_CONCENTRATIONS
    ]
    cases.append(("unknown", UNKNOWN_CONCENTRATION, UNKNOWN_LASER_SCALE))
    for name, concentration, scale in cases:
        trace = measure(concentration, scale, rng)
        readings.append(
            TransmissionReading(
                name=name,
                concentration=concentration,
                laser_scale=scale,
                raw_transmittance=float(trace.sample_on.mean() / blank.sample_on.mean()),
                corrected_transmittance=sample_reference_ratio(trace) / blank_ratio,
            )
        )
    return readings


def fit_attenuation(calibration: list[TransmissionReading]) -> float:
    concentrations = np.array([reading.concentration for reading in calibration])
    optical_depths = -np.log([reading.corrected_transmittance for reading in calibration])
    attenuation = float(
        np.dot(concentrations, optical_depths) / np.dot(concentrations, concentrations)
    )
    if attenuation <= 0.0:
        raise ValueError("Calibration does not show positive attenuation.")
    return attenuation


def estimate_concentration(transmittance: float, attenuation: float) -> float:
    return float(-np.log(transmittance) / attenuation)


def write_summary(
    path: Path, readings: list[TransmissionReading], estimate: float
) -> None:
    rows = []
    for reading in readings:
        truth = true_transmittance(reading.concentration)
        is_unknown = reading.name == "unknown"
        rows.append(
            {
                "sample": reading.name,
                "true_concentration_pct": f"{reading.concentration * 100:.2f}",
                "laser_output_pct": f"{reading.laser_scale * 100:.2f}",
                "true_transmittance": f"{truth:.6f}",
                "raw_transmittance": f"{reading.raw_transmittance:.6f}",
                "corrected_transmittance": f"{reading.corrected_transmittance:.6f}",
                "raw_absolute_error": f"{abs(reading.raw_transmittance - truth):.6f}",
                "corrected_absolute_error": f"{abs(reading.corrected_transmittance - truth):.6f}",
                "estimated_concentration_pct": f"{estimate * 100:.4f}" if is_unknown else "",
                "concentration_absolute_error_pp": (
                    f"{abs(estimate - reading.concentration) * 100:.4f}" if is_unknown else ""
                ),
                "fringe_visibility": f"{fringe_visibility(reading.concentration):.6f}",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_results(
    path: Path, readings: list[TransmissionReading], attenuation: float, estimate: float
) -> None:
    blank, *calibration, unknown = readings
    blue, teal, amber = "#235b96", "#087c6c", "#aa5410"
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4), layout="constrained")
    fig.suptitle("Optical liquid measurement simulation", fontsize=19, fontweight="bold")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e1e7ed")
        axis.set_axisbelow(True)

    ax = axes[0, 0]
    concentration_grid = np.linspace(0.0, 1.0, 200)
    ax.plot(
        concentration_grid * 100,
        np.exp(-attenuation * concentration_grid),
        color=blue,
        label="Fitted attenuation",
    )
    standards = (blank, *calibration)
    ax.scatter(
        [item.concentration * 100 for item in standards],
        [item.corrected_transmittance for item in standards],
        facecolor="white",
        edgecolor=blue,
        zorder=3,
        label="Blank and standards",
    )
    ax.scatter(
        estimate * 100,
        unknown.corrected_transmittance,
        marker="D",
        color=teal,
        zorder=4,
        label=f"Unknown estimate ({estimate:.2%})",
    )
    ax.set(
        title="A  Calibration and concentration estimate",
        xlabel="Relative concentration (%)",
        ylabel="Transmittance",
        ylim=(0.0, 1.1),
    )
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    truth = true_transmittance(unknown.concentration)
    bars = ax.bar(
        ["Before correction", "After correction"],
        [unknown.raw_transmittance, unknown.corrected_transmittance],
        color=[amber, teal],
        width=0.55,
    )
    ax.bar_label(bars, fmt="%.4f", padding=4)
    ax.axhline(truth, color=blue, linestyle="--", label=f"True value ({truth:.4f})")
    ax.set(
        title="B  Unknown sample with 20% laser drift",
        ylabel="Transmittance",
        ylim=(0.0, 0.52),
    )
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    ax = axes[1, 0]
    positions = np.linspace(-0.012, 0.012, 801)
    for reading, color in ((blank, blue), (unknown, teal), (calibration[-1], amber)):
        ax.plot(
            positions * 1000,
            diffraction_intensity(positions, reading.concentration),
            color=color,
            label=f"{reading.concentration:.0%} synthetic sample",
        )
    ax.set(
        title="C  Single-slit diffraction",
        xlabel="Screen position (mm)",
        ylabel="Relative intensity (a.u.)",
    )
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    center_limit_angle = np.arcsin(WAVELENGTH_M / (np.pi * SLIT_WIDTH_M))
    center_limit_m = SCREEN_DISTANCE_M * np.tan(center_limit_angle)
    center = np.linspace(-center_limit_m, center_limit_m, 201)
    difference = (
        central_taylor_intensity(center, unknown.concentration)
        - diffraction_intensity(center, unknown.concentration)
    )
    ax.plot(center * 1000, difference * 1000, color=teal)
    ax.fill_between(center * 1000, difference * 1000, color=teal, alpha=0.15)
    ax.set(
        title="D  Central Taylor approximation residual",
        xlabel="Screen position within |β| ≤ 1 (mm)",
        ylabel="Taylor − exact (10⁻³ a.u.)",
    )
    ax.text(
        0.5,
        0.79,
        f"Unknown fringe visibility: {fringe_visibility(unknown.concentration):.3f}",
        ha="center",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    readings = collect_readings()
    _, *calibration, unknown = readings
    attenuation = fit_attenuation(calibration)
    estimate = estimate_concentration(unknown.corrected_transmittance, attenuation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(args.output_dir / "summary.csv", readings, estimate)
    plot_results(args.output_dir / "overview.png", readings, attenuation, estimate)

    print(
        f"Unknown concentration: {estimate:.4%} "
        f"(absolute error {abs(estimate - unknown.concentration) * 100:.4f} pp)"
    )
    if not 0.0 <= estimate <= max(item.concentration for item in calibration):
        print("Warning: estimate is outside the calibrated range.")
    print(f"Saved overview.png and summary.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
