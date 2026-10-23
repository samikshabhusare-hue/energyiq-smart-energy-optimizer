# EnergyIQ Data Documentation

This directory contains the demo and benchmark datasets used by the EnergyIQ smart energy optimizer application.

---

## 1. Datasets Overview

- **`sample_energy.csv` (Primary Demo Dataset)**:
  - Normalized real-world dataset derived from the UCI *Power Consumption of Tetouan City* dataset.
  - Spans 52,416 ten-minute intervals across 2017 (Jan 1, 2017 00:00 to Dec 30, 2017 23:50).
  - Contains three municipal power distribution zones (`Zone 1`, `Zone 2`, `Zone 3`), totaling **157,248** interval energy readings.
  - Normalized schema: `timestamp` (YYYY-MM-DD HH:MM:SS), `energy_kwh` (numeric interval energy in kWh), and `zone` (text).

- **`sample_energy_synthetic.csv` (Synthetic Test Dataset)**:
  - **Synthetic data**: 72 hourly readings across 3 days (2026-10-20 to 2026-10-22) for a single zone (`Block A`).
  - Generated to test baseline algorithms and interface contracts. Includes one intentional spike at `2026-10-22 12:00:00` (18.0 kWh vs ~8.1 kWh baseline) to verify anomaly detection.

- **`tetouan_source.zip` (Raw Source Archive)**:
  - Original source archive downloaded from the UCI Machine Learning Repository containing `Tetuan City power consumption.csv`.

---

## 2. UCI Source & Citation

- **Repository**: [UCI Machine Learning Repository — Power Consumption of Tetouan City](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city)
- **DOI**: [10.24432/C5B034](https://doi.org/10.24432/C5B034)
- **Creators**: Abdulwahed Salam and Abdelaaziz El Hibaoui (Abdelmalek Essaadi University, Tetouan, Morocco)
- **Donation Date**: August 13, 2023

### Citation
```bibtex
@misc{salam_elhibaoui_2018_tetouan,
  author       = {Salam, Abdulwahed and El Hibaoui, Abdelaaziz},
  title        = {Power Consumption of Tetouan City},
  year         = {2018},
  howpublished = {UCI Machine Learning Repository},
  doi          = {10.24432/C5B034}
}
```

---

## 3. License & Attribution

This dataset is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
- You are free to share, copy, distribute, and adapt the dataset for any purpose with appropriate credit given to the original authors and the UCI repository.
- Full license terms: https://creativecommons.org/licenses/by/4.0/legalcode

---

## 4. Source Units, Measurement Interval, and Energy Conversion

### Source Units Verification
The data originated from the Supervisory Control and Data Acquisition (SCADA) system of **Amendis**, the public utility company managing electricity distribution for Tetouan in northern Morocco. The network is partitioned into three distribution stations:
- **Zone 1**: Quads substation (commercial & residential load)
- **Zone 2**: Smir substation
- **Zone 3**: Boussafou substation

While the UCI web variable table leaves the units column unpopulated, the peer-reviewed introductory study explicitly specifies that power consumption across the three zones is recorded in **Kilowatts (kW)**:
> **Supporting Citation**:
> Salam, A., & El Hibaoui, A. (2018). *Comparison of Machine Learning Algorithms for the Power Consumption Prediction: - Case Study of Tetouan city –*. 2018 International Renewable and Sustainable Energy Conference (IRSEC), Rabat, Morocco, pp. 1-6. DOI: [10.1109/IRSEC.2018.8786016](https://doi.org/10.1109/IRSEC.2018.8786016).

The values in the dataset (Zone 1: ~13,900 to 52,200 kW; Zone 2: ~8,560 to 37,400 kW; Zone 3: ~5,935 to 47,600 kW) reflect instantaneous municipal demand on the order of 10 to 50 megawatts (MW), consistent with city distribution zones.

### Measurement Interval
Readings were captured synchronously every **10 minutes**:
$$\Delta t = 10\text{ minutes} = \frac{10}{60}\text{ hours} = \frac{1}{6}\text{ hour}$$

### Conversion to Interval Energy ($\text{kWh}$)
Because `energy_kwh` represents the cumulative energy consumed over the 10-minute interval, the instantaneous power in kW is converted to interval energy in kilowatt-hours ($\text{kWh}$) via:
$$\text{Interval Energy (kWh)} = \text{Power (kW)} \times \frac{1}{6}\text{ h} = \frac{\text{kW}}{6}$$

Each 10-minute reading was divided by $6$ and mapped to long format with its respective zone label (`Zone 1`, `Zone 2`, `Zone 3`), sorted chronologically by `timestamp` and `zone`.
