"""
=============================================================================
Nigeria Education Crisis — Data Analysis Script
=============================================================================
Author      : Oyinlola Kayode
Framework   : DIG (Describe · Introspect · Goal)
Dataset     : World Bank Education Indicators for Nigeria (via HDX)
Source URL  : https://data.humdata.org/dataset/world-bank-education-indicators-for-nigeria

USAGE:
    python analyze.py --file /path/to/education_nga.csv

DESCRIPTION:
    This script replicates the full Nigeria Education Crisis analysis.
    It follows the DIG framework — Description, Introspection, and
    Goal Setting — to profile the dataset, extract key indicators,
    compute statistics, generate charts, and export a summary CSV.

OUTPUT:
    All outputs are saved to an 'output/' folder in the working directory:
    - output/01_description_summary.txt       — dataset profile
    - output/02_spending_trend.png            — Story 1: spending over time
    - output/03_country_comparison.png        — Story 1: Nigeria vs peers
    - output/04_pupil_teacher_ratio.png       — Story 1: classroom conditions
    - output/05_oos_trend.png                 — Story 2: out-of-school trend
    - output/06_female_oos_share.png          — Story 2: gender exclusion
    - output/07_gender_literacy.png           — Story 3: gender literacy gap
    - output/08_spending_vs_literacy.png      — Story 4: spending vs outcomes
    - output/09_key_findings_summary.csv      — all key statistics

REQUIREMENTS:
    Python 3.8+
    pip install pandas matplotlib seaborn

LIMITATIONS:
    - National-level data only (no state disaggregation)
    - Literacy rate has only 6 data points (1991-2024)
    - Primary completion rate last available in 2010
    - Out-of-school data has an 11-year gap (2010-2023)
    - Correlations are indicative only — causality not established
    - Informal/Quranic education not captured in dataset

=============================================================================
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ── CONFIGURATION ─────────────────────────────────────────────────────────────

INDICATORS = {
    'oos_total':      'Children out of school, primary',
    'oos_female':     'Children out of school, primary, female',
    'oos_male':       'Children out of school, primary, male',
    'literacy_adult': 'Literacy rate, adult total (% of people ages 15 and above)',
    'literacy_m':     'Literacy rate, youth male (% of males ages 15-24)',
    'literacy_f':     'Literacy rate, youth female (% of females ages 15-24)',
    'spend':          'Government expenditure on education, total (% of GDP)',
    'completion':     'Primary completion rate, total (% of relevant age group)',
    'pupil_teacher':  'Pupil-teacher ratio, primary',
}

# World comparison data (UNESCO/World Bank 2023 estimates)
WORLD_COMPARE = {
    'Nigeria':            0.32,
    'UNESCO Minimum':     4.00,
    'Sub-Saharan avg':    4.10,
    'World average':      4.30,
    'Ghana':              4.00,
    'Kenya':              5.30,
    'South Africa':       6.20,
}

# Style settings
STYLE = {
    'bg':      '#0d1117',
    'surface': '#161b22',
    'text':    '#e6edf3',
    'muted':   '#8b949e',
    'red':     '#e85d5d',
    'amber':   '#e8a23a',
    'green':   '#3ab87a',
    'blue':    '#4a90d9',
    'pink':    '#d45f9e',
}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def setup_style():
    """Apply dark theme to all matplotlib charts."""
    plt.rcParams.update({
        'figure.facecolor':  STYLE['bg'],
        'axes.facecolor':    STYLE['surface'],
        'axes.edgecolor':    '#30363d',
        'axes.labelcolor':   STYLE['muted'],
        'axes.titlecolor':   STYLE['text'],
        'text.color':        STYLE['text'],
        'xtick.color':       STYLE['muted'],
        'ytick.color':       STYLE['muted'],
        'grid.color':        '#21262d',
        'grid.linewidth':    0.7,
        'legend.facecolor':  STYLE['surface'],
        'legend.edgecolor':  '#30363d',
        'font.family':       'DejaVu Sans',
        'font.size':         10,
        'axes.titlesize':    12,
        'axes.titleweight':  'bold',
        'axes.labelsize':    10,
        'figure.dpi':        120,
    })


def get_indicator(df, key):
    """Extract a single indicator as a clean Year/Value DataFrame."""
    name = INDICATORS[key]
    subset = df[df['Indicator Name'] == name][['Year', 'Value']].dropna()
    return subset.sort_values('Year').reset_index(drop=True)


def save(fig, path, label):
    """Save figure and print confirmation."""
    fig.savefig(path, bbox_inches='tight', facecolor=STYLE['bg'])
    plt.close(fig)
    print(f"  ✓  Saved: {path}  [{label}]")


# ── STAGE D: DESCRIPTION ──────────────────────────────────────────────────────

def stage_description(df, output_dir):
    """Profile the dataset and write a summary text file."""
    print("\n" + "="*60)
    print("  D — DESCRIPTION STAGE")
    print("="*60)

    lines = []
    lines.append("NIGERIA EDUCATION DATASET — DESCRIPTION SUMMARY")
    lines.append("="*60)
    lines.append(f"Rows              : {len(df):,}")
    lines.append(f"Columns           : {df.shape[1]}")
    lines.append(f"Unique indicators : {df['Indicator Name'].nunique():,}")
    lines.append(f"Year range        : {int(df['Year'].min())} – {int(df['Year'].max())}")
    lines.append(f"Missing values    : {df.isnull().sum().sum()}")
    lines.append(f"Country           : {df['Country Name'].unique()[0]}")
    lines.append(f"ISO3 code         : {df['Country ISO3'].unique()[0]}")
    lines.append("")
    lines.append("COLUMNS:")
    for col in df.columns:
        lines.append(f"  {col:<20} {str(df[col].dtype):<12} Sample: {str(df[col].iloc[0])[:40]}")
    lines.append("")
    lines.append("KEY INDICATORS USED IN THIS ANALYSIS:")
    for key, name in INDICATORS.items():
        subset = get_indicator(df, key)
        if not subset.empty:
            lines.append(f"  {key:<20} {len(subset)} data pts  "
                         f"({int(subset['Year'].min())}–{int(subset['Year'].max())})")
            lines.append(f"    → {name}")
        else:
            lines.append(f"  {key:<20} NOT FOUND in dataset")
    lines.append("")
    lines.append("VALUE STATISTICS (all indicators):")
    lines.append(str(df['Value'].describe()))

    summary = "\n".join(lines)
    print(summary)

    out_path = os.path.join(output_dir, "01_description_summary.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"\n  ✓  Saved: {out_path}")
    return summary


# ── STAGE I: INTROSPECTION ────────────────────────────────────────────────────

def stage_introspection(df, output_dir):
    """Run all introspection analysis and generate charts."""
    print("\n" + "="*60)
    print("  I — INTROSPECTION STAGE")
    print("="*60)
    setup_style()

    # ── STORY 1a: Spending trend ──────────────────────────────────────────────
    spend = get_indicator(df, 'spend')
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [STYLE['green'] if v > 1 else STYLE['red'] for v in spend['Value']]
    ax.fill_between(spend['Year'], spend['Value'], alpha=0.12, color=STYLE['red'])
    ax.plot(spend['Year'], spend['Value'], color=STYLE['red'], linewidth=2, zorder=3)
    ax.scatter(spend['Year'], spend['Value'], color=colors, s=50, zorder=4)
    ax.axhline(4.0, color=STYLE['amber'], linestyle='--', linewidth=1.2,
               label='UNESCO minimum (4%)')
    ax.set_title("Story 1 — Government Expenditure on Education (% of GDP)")
    ax.set_xlabel("Year")
    ax.set_ylabel("% of GDP")
    ax.legend(fontsize=9)
    ax.grid(True, axis='y')
    # Annotate key points
    ax.annotate('3.21% (1974)', xy=(1974, 3.21), xytext=(1978, 2.7),
                color=STYLE['green'], fontsize=8,
                arrowprops=dict(arrowstyle='->', color=STYLE['green'], lw=1))
    ax.annotate('0.32% (2023)', xy=(2023, 0.32), xytext=(2015, 0.8),
                color=STYLE['red'], fontsize=8,
                arrowprops=dict(arrowstyle='->', color=STYLE['red'], lw=1))
    fig.tight_layout()
    save(fig, os.path.join(output_dir, "02_spending_trend.png"), "Story 1: Spending trend")

    # ── STORY 1b: Country comparison ─────────────────────────────────────────
    countries = list(WORLD_COMPARE.keys())
    values = list(WORLD_COMPARE.values())
    bar_colors = [STYLE['red'] if c == 'Nigeria'
                  else STYLE['amber'] if 'UNESCO' in c or 'avg' in c.lower()
                  else STYLE['green'] for c in countries]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(countries, values, color=bar_colors, height=0.55, edgecolor='none')
    for bar, val in zip(bars, values):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val}%', va='center', color=STYLE['text'], fontsize=9)
    ax.axvline(4.0, color=STYLE['amber'], linestyle='--', linewidth=1.2,
               label='UNESCO minimum (4%)', alpha=0.7)
    ax.set_title("Story 1 — Education Spending: Nigeria vs Peers (% of GDP, 2023)")
    ax.set_xlabel("% of GDP")
    ax.legend(fontsize=9)
    ax.grid(True, axis='x')
    ax.invert_yaxis()
    fig.tight_layout()
    save(fig, os.path.join(output_dir, "03_country_comparison.png"),
         "Story 1: Country comparison")

    # ── STORY 1c: Pupil-teacher ratio ─────────────────────────────────────────
    pt = get_indicator(df, 'pupil_teacher')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(pt['Year'], pt['Value'], alpha=0.1, color=STYLE['amber'])
    ax.plot(pt['Year'], pt['Value'], color=STYLE['amber'], linewidth=2, marker='o',
            markersize=5, zorder=3)
    ax.axhline(40, color=STYLE['red'], linestyle='--', linewidth=1.2,
               label='UNESCO maximum (40 pupils/teacher)')
    ax.set_title("Story 1 — Pupil-Teacher Ratio, Primary Schools")
    ax.set_xlabel("Year")
    ax.set_ylabel("Pupils per teacher")
    ax.legend(fontsize=9)
    ax.grid(True, axis='y')
    fig.tight_layout()
    save(fig, os.path.join(output_dir, "04_pupil_teacher_ratio.png"),
         "Story 1: Pupil-teacher ratio")

    # ── STORY 2a: Out-of-school trend ─────────────────────────────────────────
    oos = get_indicator(df, 'oos_total')
    oos['Value_M'] = oos['Value'] / 1e6
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(oos['Year'], oos['Value_M'], alpha=0.1, color=STYLE['red'])
    ax.plot(oos['Year'], oos['Value_M'], color=STYLE['red'], linewidth=2,
            marker='o', markersize=6, zorder=3)
    ax.set_title("Story 2 — Children Out of School, Primary (Millions)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Millions of children")
    ax.grid(True, axis='y')
    # Annotate peak and latest
    peak_idx = oos['Value_M'].idxmax()
    ax.annotate(f"Peak: {oos.loc[peak_idx,'Value_M']:.2f}M ({int(oos.loc[peak_idx,'Year'])})",
                xy=(oos.loc[peak_idx,'Year'], oos.loc[peak_idx,'Value_M']),
                xytext=(oos.loc[peak_idx,'Year']-4, oos.loc[peak_idx,'Value_M']+0.3),
                color=STYLE['red'], fontsize=8,
                arrowprops=dict(arrowstyle='->', color=STYLE['red'], lw=1))
    fig.tight_layout()
    save(fig, os.path.join(output_dir, "05_oos_trend.png"),
         "Story 2: Out-of-school trend")

    # ── STORY 2b: Female share ────────────────────────────────────────────────
    oos_f = get_indicator(df, 'oos_female')
    oos_m = get_indicator(df, 'oos_male')
    merged = oos_f.merge(oos_m, on='Year', suffixes=('_f', '_m')).dropna()
    merged['pct_female'] = (merged['Value_f'] /
                            (merged['Value_f'] + merged['Value_m'])) * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(merged['Year'], merged['pct_female'], alpha=0.1,
                    color=STYLE['pink'])
    ax.plot(merged['Year'], merged['pct_female'], color=STYLE['pink'],
            linewidth=2, marker='o', markersize=6, zorder=3)
    ax.axhline(50, color=STYLE['muted'], linestyle='--', linewidth=1,
               label='50% parity line', alpha=0.6)
    ax.set_title("Story 2 — Female Share of Out-of-School Children (%)")
    ax.set_xlabel("Year")
    ax.set_ylabel("% female")
    ax.set_ylim(45, 65)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y')
    fig.tight_layout()
    save(fig, os.path.join(output_dir, "06_female_oos_share.png"),
         "Story 2: Female share")

    # ── STORY 3: Gender literacy gap ─────────────────────────────────────────
    lit_m = get_indicator(df, 'literacy_m')
    lit_f = get_indicator(df, 'literacy_f')
    gender = lit_m.merge(lit_f, on='Year', suffixes=('_m', '_f')).dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gender['Year'], gender['Value_m'], color=STYLE['blue'], linewidth=2,
            marker='o', markersize=6, label='Male youth literacy', linestyle='--')
    ax.plot(gender['Year'], gender['Value_f'], color=STYLE['pink'], linewidth=2,
            marker='o', markersize=6, label='Female youth literacy')
    ax.fill_between(gender['Year'], gender['Value_m'], gender['Value_f'],
                    alpha=0.08, color=STYLE['pink'],
                    where=(gender['Value_m'] > gender['Value_f']))
    ax.fill_between(gender['Year'], gender['Value_m'], gender['Value_f'],
                    alpha=0.08, color=STYLE['green'],
                    where=(gender['Value_f'] >= gender['Value_m']))
    ax.set_title("Story 3 — Youth Literacy Rate: Male vs Female (%)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Literacy rate (%)")
    ax.legend(fontsize=9)
    ax.grid(True, axis='y')
    # Annotate the crossover
    ax.annotate('Girls overtake boys\nfor first time (2024)',
                xy=(2024, 81.4), xytext=(2018, 87),
                color=STYLE['green'], fontsize=8,
                arrowprops=dict(arrowstyle='->', color=STYLE['green'], lw=1))
    fig.tight_layout()
    save(fig, os.path.join(output_dir, "07_gender_literacy.png"),
         "Story 3: Gender literacy gap")

    # ── STORY 4: Spending vs literacy ─────────────────────────────────────────
    lit_a = get_indicator(df, 'literacy_adult')
    spend2 = get_indicator(df, 'spend')
    merged2 = lit_a.merge(spend2, on='Year', suffixes=('_lit', '_spend')).dropna()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(lit_a['Year'], lit_a['Value'], color=STYLE['blue'], linewidth=2,
             marker='o', markersize=6, label='Adult literacy rate (%)')
    ax2.plot(spend2['Year'], spend2['Value'], color=STYLE['red'], linewidth=2,
             marker='s', markersize=6, linestyle='--', label='Govt spend % GDP')
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Adult literacy rate (%)", color=STYLE['blue'])
    ax2.set_ylabel("Govt spend % of GDP", color=STYLE['red'])
    ax1.tick_params(axis='y', labelcolor=STYLE['blue'])
    ax2.tick_params(axis='y', labelcolor=STYLE['red'])
    ax1.set_title("Story 4 — Government Spending vs Adult Literacy Rate")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
               loc='upper left')
    ax1.grid(True, axis='y', alpha=0.4)
    fig.tight_layout()
    save(fig, os.path.join(output_dir, "08_spending_vs_literacy.png"),
         "Story 4: Spending vs literacy")

    return gender, merged


# ── STAGE G: GOAL SETTING & SUMMARY ──────────────────────────────────────────

def stage_goal(df, gender_df, output_dir):
    """Print goal-setting summary and export key statistics CSV."""
    print("\n" + "="*60)
    print("  G — GOAL SETTING & KEY FINDINGS")
    print("="*60)

    spend = get_indicator(df, 'spend')
    oos   = get_indicator(df, 'oos_total')
    lit_a = get_indicator(df, 'literacy_adult')
    lit_m = get_indicator(df, 'literacy_m')
    lit_f = get_indicator(df, 'literacy_f')

    # Compute key statistics
    spend_1974  = spend[spend['Year'] == 1974]['Value'].values
    spend_2023  = spend[spend['Year'] == 2023]['Value'].values
    oos_peak    = oos['Value'].max() / 1e6
    oos_peak_yr = int(oos.loc[oos['Value'].idxmax(), 'Year'])
    oos_2023    = oos[oos['Year'] == 2023]['Value'].values
    lit_2024    = lit_a[lit_a['Year'] == 2024]['Value'].values
    lit_1991    = lit_a[lit_a['Year'] == 1991]['Value'].values

    # Gender gap
    gender_1991 = gender_df[gender_df['Year'] == 1991]
    gender_2024 = gender_df[gender_df['Year'] == 2024]

    findings = {
        'Metric': [
            'Govt education spend 1974 (% GDP)',
            'Govt education spend 2023 (% GDP)',
            'Spending decline (%)',
            'Spending decline (times)',
            'Annual funding gap to UNESCO minimum ($B)',
            'Out-of-school children peak (M)',
            'Out-of-school children peak year',
            'Out-of-school children 2023 (M)',
            'Change in OOS children (M)',
            'Adult literacy 1991 (%)',
            'Adult literacy 2024 (%)',
            'Male youth literacy 1991 (%)',
            'Female youth literacy 1991 (%)',
            'Gender literacy gap 1991 (pp)',
            'Male youth literacy 2024 (%)',
            'Female youth literacy 2024 (%)',
            'Gender literacy gap 2024 (pp)',
        ],
        'Value': [
            round(float(spend_1974[0]), 2) if len(spend_1974) else 'N/A',
            round(float(spend_2023[0]), 2) if len(spend_2023) else 'N/A',
            round((3.21 - 0.32) / 3.21 * 100, 1),
            round(3.21 / 0.32, 1),
            round(477 * 0.04 - 477 * 0.0032, 1),
            round(oos_peak, 2),
            oos_peak_yr,
            round(float(oos_2023[0]) / 1e6, 2) if len(oos_2023) else 'N/A',
            round(oos_peak - (float(oos_2023[0]) / 1e6 if len(oos_2023) else 0), 2),
            round(float(lit_1991[0]), 1) if len(lit_1991) else 'N/A',
            round(float(lit_2024[0]), 1) if len(lit_2024) else 'N/A',
            round(float(gender_1991['Value_m'].values[0]), 1) if not gender_1991.empty else 'N/A',
            round(float(gender_1991['Value_f'].values[0]), 1) if not gender_1991.empty else 'N/A',
            round(float(gender_1991['Value_m'].values[0]) - float(gender_1991['Value_f'].values[0]), 1) if not gender_1991.empty else 'N/A',
            round(float(gender_2024['Value_m'].values[0]), 1) if not gender_2024.empty else 'N/A',
            round(float(gender_2024['Value_f'].values[0]), 1) if not gender_2024.empty else 'N/A',
            round(float(gender_2024['Value_m'].values[0]) - float(gender_2024['Value_f'].values[0]), 2) if not gender_2024.empty else 'N/A',
        ],
        'Source': ['World Bank'] * 17,
        'Indicator Code': [
            'SE.XPD.TOTL.GD.ZS', 'SE.XPD.TOTL.GD.ZS', 'Computed', 'Computed', 'Computed',
            'SE.PRM.UNER', 'SE.PRM.UNER', 'SE.PRM.UNER', 'Computed',
            'SE.ADT.LITR.ZS', 'SE.ADT.LITR.ZS',
            'SE.ADT.1524.LT.MA.ZS', 'SE.ADT.1524.LT.FE.ZS', 'Computed',
            'SE.ADT.1524.LT.MA.ZS', 'SE.ADT.1524.LT.FE.ZS', 'Computed',
        ]
    }

    summary_df = pd.DataFrame(findings)
    csv_path = os.path.join(output_dir, "09_key_findings_summary.csv")
    summary_df.to_csv(csv_path, index=False)

    print("\n  KEY FINDINGS:")
    for _, row in summary_df.iterrows():
        print(f"  {row['Metric']:<45} {str(row['Value']):<12} [{row['Indicator Code']}]")

    print(f"\n  ✓  Saved: {csv_path}")
    return summary_df


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Nigeria Education Crisis — DIG Framework Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py --file education_nga.csv
  python analyze.py --file /Users/oyinlola/data/education_nga.csv
  python analyze.py --file education_nga.csv --output my_results

Dataset download:
  https://data.humdata.org/dataset/world-bank-education-indicators-for-nigeria
        """
    )
    parser.add_argument(
        '--file', '-f', required=True,
        help='Path to the education_nga.csv dataset file'
    )
    parser.add_argument(
        '--output', '-o', default='output',
        help='Output directory for charts and summaries (default: output/)'
    )
    args = parser.parse_args()

    # Validate file
    if not os.path.exists(args.file):
        print(f"\n  ERROR: File not found: {args.file}")
        print("  Please download the dataset from:")
        print("  https://data.humdata.org/dataset/world-bank-education-indicators-for-nigeria")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("\n" + "="*60)
    print("  NIGERIA EDUCATION CRISIS — DIG FRAMEWORK ANALYSIS")
    print("  Author: Oyinlola Kayode")
    print("="*60)
    print(f"\n  Input file : {args.file}")
    print(f"  Output dir : {args.output}/")

    # Load data
    print("\n  Loading dataset...")
    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        print(f"\n  ERROR loading file: {e}")
        sys.exit(1)

    print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")

    # Validate expected columns
    expected = {'Country Name', 'Country ISO3', 'Year', 'Indicator Name',
                'Indicator Code', 'Value'}
    if not expected.issubset(set(df.columns)):
        print(f"\n  ERROR: Unexpected column structure.")
        print(f"  Expected: {expected}")
        print(f"  Got:      {set(df.columns)}")
        sys.exit(1)

    # Run DIG stages
    stage_description(df, args.output)
    gender_df, _ = stage_introspection(df, args.output)
    stage_goal(df, gender_df, args.output)

    print("\n" + "="*60)
    print("  ANALYSIS COMPLETE")
    print(f"  All outputs saved to: {os.path.abspath(args.output)}/")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
