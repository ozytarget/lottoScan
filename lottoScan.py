import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# =========================================
# 1. Game configuration (Mega / Power)
# =========================================

@dataclass
class GameConfig:
    key: str
    name: str
    url: str
    n_main: int
    max_main: int
    max_special: int


MEGA_CONFIG = GameConfig(
    key="mega",
    name="Mega Millions",
    # Texas Lottery CSV – todos los sorteos nacionales de Mega Millions
    url="https://www.texaslottery.com/export/sites/lottery/Games/"
        "Mega_Millions/Winning_Numbers/megamillions.csv",
    n_main=5,
    max_main=70,   # 5 números del 1–70 :contentReference[oaicite:1]{index=1}
    max_special=24 # Mega Ball del 1–24
)

POWER_CONFIG = GameConfig(
    key="power",
    name="Powerball",
    # Texas Lottery CSV – todos los sorteos nacionales de Powerball
    url="https://www.texaslottery.com/export/sites/lottery/Games/"
        "Powerball/Winning_Numbers/powerball.csv",
    n_main=5,
    max_main=69,   # 5 números del 1–69 :contentReference[oaicite:2]{index=2}
    max_special=26 # Powerball del 1–26
)

GAMES: Dict[str, GameConfig] = {
    MEGA_CONFIG.key: MEGA_CONFIG,
    POWER_CONFIG.key: POWER_CONFIG,
}


# =========================================
# 2. Data loading
# =========================================

@st.cache_data(show_spinner=True)
def download_raw_csv(cfg: GameConfig) -> pd.DataFrame:
    """
    Download raw CSV from Texas Lottery for the given game.
    The CSV format (for both games) is documented as:
      Game Name, Month, Day, Year, Num1, Num2, Num3, Num4, Num5, SpecialBall, [Optional Extra Column]
    We read without header and parse by position.
    """
    df = pd.read_csv(cfg.url, header=None)
    # Ensure there are at least 10 columns (0..9)
    if df.shape[1] < 10:
        raise ValueError("Unexpected CSV format (less than 10 columns).")
    return df


def build_date_column(df: pd.DataFrame) -> pd.Series:
    """
    Build a datetime column from columns:
      1 = month, 2 = day, 3 = year
    """
    # Use to_numeric with errors='coerce' to avoid issues with non-numeric junk
    month = pd.to_numeric(df[1], errors="coerce")
    day = pd.to_numeric(df[2], errors="coerce")
    year = pd.to_numeric(df[3], errors="coerce")

    dates = pd.to_datetime(
        dict(year=year, month=month, day=day),
        errors="coerce"
    )
    return dates


def clean_and_filter(
    raw: pd.DataFrame,
    cfg: GameConfig,
    days_back: int
) -> pd.DataFrame:
    """
    Clean raw CSV and keep only last N days.

    Expected generic layout:
      0: Game Name
      1: Month
      2: Day
      3: Year
      4-8: Main numbers
      9: Special ball (Mega Ball or Powerball)
      10: Optional extra column (Megaplier / Power Play) – ignored
    """
    df = raw.copy()

    # Build date column
    df["draw_date"] = build_date_column(df)
    df = df.dropna(subset=["draw_date"])

    # Numeric columns for numbers (coerce invalid -> NaN, then drop)
    main_cols = [4, 5, 6, 7, 8]
    for c in main_cols + [9]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=main_cols + [9])

    # Filter last N days
    cutoff = datetime.now().date() - timedelta(days=days_back)
    df = df[df["draw_date"].dt.date >= cutoff]

    # Sort descending by date (latest first)
    df = df.sort_values("draw_date", ascending=False).reset_index(drop=True)

    # Rename columns to something more friendly
    rename_map = {
        "draw_date": "date",
        4: "n1",
        5: "n2",
        6: "n3",
        7: "n4",
        8: "n5",
        9: "special",
    }
    df = df.rename(columns=rename_map)

    # Keep only needed columns
    df = df[["date", "n1", "n2", "n3", "n4", "n5", "special"]]

    # Sanity check: values within the official ranges
    df = df[
        (df["n1"].between(1, cfg.max_main))
        & (df["n2"].between(1, cfg.max_main))
        & (df["n3"].between(1, cfg.max_main))
        & (df["n4"].between(1, cfg.max_main))
        & (df["n5"].between(1, cfg.max_main))
        & (df["special"].between(1, cfg.max_special))
    ]

    return df


# =========================================
# 3. Frequency & pattern logic
# =========================================

def compute_frequencies(df: pd.DataFrame):
    """
    Compute frequency counters for main numbers and special numbers.
    """
    from collections import Counter

    all_main = (
        df[["n1", "n2", "n3", "n4", "n5"]]
        .values
        .ravel()
        .tolist()
    )
    main_counter = Counter(all_main)
    special_counter = Counter(df["special"].tolist())
    return main_counter, special_counter


def split_ranges(max_main: int) -> Dict[str, range]:
    """
    Split main number space into low/mid/high ranges.
    """
    third = max_main // 3
    low = range(1, third + 1)
    mid = range(third + 1, 2 * third + 1)
    high = range(2 * third + 1, max_main + 1)
    return {"low": low, "mid": mid, "high": high}


def generate_main_candidate(
    cfg: GameConfig,
    main_counter,
    max_attempts: int = 300
) -> Optional[List[int]]:
    """
    Generate a main-number candidate with:
      - Weighted random by frequency
      - At least one number in each range (low/mid/high)
      - Mix of even and odd
    """
    pool = list(range(1, cfg.max_main + 1))
    # +1 to keep zero-frequency numbers still possible
    weights = [main_counter.get(n, 0) + 1 for n in pool]
    ranges = split_ranges(cfg.max_main)

    for _ in range(max_attempts):
        # oversample then deduplicate
        draw = random.choices(pool, weights=weights, k=cfg.n_main * 2)
        nums = sorted(set(draw))[: cfg.n_main]
        if len(nums) < cfg.n_main:
            continue

        # range constraints
        has_low = any(n in ranges["low"] for n in nums)
        has_mid = any(n in ranges["mid"] for n in nums)
        has_high = any(n in ranges["high"] for n in nums)
        if not (has_low and has_mid and has_high):
            continue

        # even/odd mix
        evens = sum(n % 2 == 0 for n in nums)
        odds = cfg.n_main - evens
        if evens == 0 or odds == 0:
            # reject all-even or all-odd
            continue

        return nums

    return None


def choose_special_candidate(
    cfg: GameConfig,
    special_counter
) -> int:
    """
    Choose a special ball number (Mega Ball / Powerball),
    favoring the most frequent ones.
    """
    if not special_counter or len(special_counter) == 0:
        return random.randint(1, cfg.max_special)

    top = special_counter.most_common(6)
    nums, freqs = zip(*top)
    weights = [f + 1 for f in freqs]
    return random.choices(list(nums), weights=weights, k=1)[0]


def generate_combinations(
    cfg: GameConfig,
    main_counter,
    special_counter,
    n_combos: int
) -> List[Tuple[List[int], int]]:
    """
    Generate N candidate combos (main numbers, special ball),
    respecting pattern rules.
    """
    combos: List[Tuple[List[int], int]] = []
    seen = set()
    attempts = 0
    max_attempts = n_combos * 80

    while len(combos) < n_combos and attempts < max_attempts:
        attempts += 1
        main_nums = generate_main_candidate(cfg, main_counter)
        if not main_nums:
            continue

        special = choose_special_candidate(cfg, special_counter)
        combo_key = (*main_nums, special)
        if combo_key in seen:
            continue

        seen.add(combo_key)
        combos.append((main_nums, special))

    return combos


# =========================================
# 4. Streamlit UI
# =========================================

def main():
    st.set_page_config(
        page_title="Lottery Scanner – Mega Millions & Powerball",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for casino-like styling + mobile responsive
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        .main {
            background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
        }
        
        /* Mobile first - optimize for small screens */
        @media (max-width: 600px) {
            .stApp {
                padding: 0 8px !important;
            }
            
            h1 {
                font-size: 32px !important;
                margin-bottom: 8px !important;
            }
            
            h2 {
                font-size: 20px !important;
                margin-top: 12px !important;
            }
            
            h3 {
                font-size: 16px !important;
            }
            
            [data-testid="column"] {
                padding: 4px !important;
            }
            
            .ball {
                width: 45px !important;
                height: 45px !important;
                margin: 4px !important;
                font-size: 16px !important;
                line-height: 45px !important;
            }
            
            .special-ball {
                width: 55px !important;
                height: 55px !important;
                margin: 6px !important;
                font-size: 20px !important;
                line-height: 55px !important;
            }
            
            .combo-card {
                padding: 12px !important;
                border-radius: 12px !important;
                margin: 8px 0 !important;
            }
            
            .stMetric {
                padding: 12px !important;
            }
            
            .stDataFrame {
                font-size: 12px !important;
            }
        }
        
        /* Tablet - 600px to 1200px */
        @media (min-width: 600px) and (max-width: 1200px) {
            h1 {
                font-size: 40px !important;
            }
            
            .ball {
                width: 55px !important;
                height: 55px !important;
                font-size: 18px !important;
                line-height: 55px !important;
            }
            
            .special-ball {
                width: 65px !important;
                height: 65px !important;
                font-size: 22px !important;
                line-height: 65px !important;
            }
        }
        
        /* Desktop - 1200px+ */
        @media (min-width: 1200px) {
            h1 {
                font-size: 48px !important;
            }
            
            .ball {
                width: 60px;
                height: 60px;
                font-size: 24px;
                line-height: 60px;
            }
            
            .special-ball {
                width: 70px;
                height: 70px;
                font-size: 28px;
                line-height: 70px;
            }
        }
        
        /* Common styles for all screen sizes */
        .stMetric {
            background: linear-gradient(135deg, #ff6b6b 0%, #ff8e53 100%);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(255, 107, 107, 0.3);
            color: white;
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
            color: #1a1a2e;
            font-weight: bold;
            font-size: 16px;
            border: none;
            box-shadow: 0 8px 32px rgba(255, 215, 0, 0.4);
            transition: all 0.3s;
            width: 100%;
        }
        
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 12px 40px rgba(255, 215, 0, 0.6);
        }
        
        .ball {
            display: inline-block;
            background: radial-gradient(circle at 30% 30%, #FFD700, #FFA500);
            border-radius: 50%;
            font-weight: bold;
            color: #1a1a2e;
            text-align: center;
            box-shadow: 0 8px 20px rgba(255, 215, 0, 0.4), inset -2px -2px 5px rgba(0,0,0,0.3);
            border: 2px solid #FFB300;
            transition: all 0.2s ease;
        }
        
        .ball:hover {
            transform: scale(1.15);
            box-shadow: 0 12px 30px rgba(255, 215, 0, 0.6), inset -2px -2px 5px rgba(0,0,0,0.3);
        }
        
        .special-ball {
            display: inline-block;
            background: radial-gradient(circle at 30% 30%, #FF1493, #FF69B4);
            border-radius: 50%;
            font-weight: bold;
            color: white;
            text-align: center;
            box-shadow: 0 12px 28px rgba(255, 20, 147, 0.5), inset -2px -2px 5px rgba(0,0,0,0.3);
            border: 2px solid #FF1493;
            transition: all 0.2s ease;
        }
        
        .special-ball:hover {
            transform: scale(1.15);
            box-shadow: 0 16px 40px rgba(255, 20, 147, 0.7), inset -2px -2px 5px rgba(0,0,0,0.3);
        }
        
        .combo-card {
            background: linear-gradient(135deg, #2d3561 0%, #1a1a2e 100%);
            border: 3px solid #FFD700;
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 12px 48px rgba(255, 215, 0, 0.3);
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .combo-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 16px 60px rgba(255, 215, 0, 0.5);
        }
        
        h1 {
            color: #FFD700;
            text-shadow: 0 0 30px rgba(255, 215, 0, 0.6), 0 0 60px rgba(255, 107, 157, 0.3);
        }
        
        h2 {
            color: #FF6B9D;
            text-shadow: 0 0 15px rgba(255, 107, 157, 0.3);
        }
        
        h3 {
            color: #FFD700 !important;
            text-shadow: 0 0 10px rgba(255, 215, 0, 0.4);
        }
        
        .stSubheader {
            color: #FFD700 !important;
        }
        
        .stDataFrame {
            width: 100%;
            overflow-x: auto !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎲 Lottery Scanner ")
    st.caption(
        "Statistical scanner (non-predictive). Analyzes recent history "
        "and suggests combinations based on dispersion and frequency patterns."
    )

    # =====================================
    # Luck Check (Humorous Section)
    # =====================================
    import hashlib
    
    # Generate a "luck score" based on current date (changes daily)
    today = datetime.now().strftime("%Y-%m-%d")
    luck_hash = int(hashlib.md5(today.encode()).hexdigest(), 16) % 100
    
    # Fun luck messages
    if luck_hash < 10:
        luck_msg = "🚨 **SALADO LEVEL: EXTREME** 🚨 | Eres tan mala suerte que los boletos te evitan. Mejor no vengas. 😂"
        luck_color = "🔴"
    elif luck_hash < 25:
        luck_msg = "⚠️ **CURSED MODE ACTIVATED** ⚠️ | Hasta el universo dice 'no por hoy'. Considera quedarte en cama."
        luck_color = "🟠"
    elif luck_hash < 50:
        luck_msg = "😐 **Neutrally Unfortunate** | Ni bueno ni malo... solo mediocre. Como la mayoría."
        luck_color = "🟡"
    elif luck_hash < 75:
        luck_msg = "✨ **Vibes are decent today** ✨ | La suerte no es tu enemigo... todavía."
        luck_color = "🟢"
    else:
        luck_msg = "🌟 **BLESSED BY THE LOTTERY GODS** 🌟 | Hoy es tu día. Ve y compra todos los boletos. (Aviso: esto es broma)"
        luck_color = "🟢"
    
    with st.container():
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(255, 215, 0, 0.1), rgba(255, 20, 147, 0.1)); 
                    border: 2px solid #FFD700; border-radius: 10px; padding: 15px; margin-bottom: 20px; text-align: center;">
            <h3 style="margin: 0; color: #FFD700;">☠️ TODAY'S LUCK STATUS ☠️</h3>
            <p style="font-size: 18px; margin: 10px 0; color: #FF6B9D;"><b>{luck_msg}</b></p>
            <small style="color: #999;">*Generated daily. If you're actually cursed, we can't help you.* 😅</small>
        </div>
        """, unsafe_allow_html=True)

    # =====================================
    # Controls - Responsive layout
    # =====================================
    col1, col2, col3 = st.columns(3)
    
    with col1:
        game_choice = st.selectbox(
            "Game",
            options=["mega", "power"],
            format_func=lambda x: "Mega Millions" if x == "mega" else "Powerball",
        )
    
    with col2:
        days_back = st.slider(
            "Days to analyze",
            min_value=15,
            max_value=180,
            value=60,
            step=5,
        )
    
    with col3:
        n_combos = st.slider(
            "Combinations",
            min_value=3,
            max_value=20,
            value=8,
            step=1,
        )
    
    cfg = GAMES[game_choice]
    st.info(
        f"{cfg.name}: 5 numbers from 1–{cfg.max_main} + 1 special from 1–{cfg.max_special}"
    )

    # =====================================
    # Execute button
    # =====================================
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        run_button = st.button("🔍 Run Scanner", use_container_width=True, type="primary")

    if not run_button:
        st.stop()

    # =====================================
    # Execute scanner
    # =====================================
    try:
        with st.spinner("Analyzing data..."):

            raw = download_raw_csv(cfg)
            df = clean_and_filter(raw, cfg, days_back)
    except Exception as e:
        st.error(f"Error al descargar o procesar los datos: {e}")
        st.stop()

    if df.empty:
        st.warning(
            "No se encontraron sorteos dentro del rango de días seleccionado. "
            "Prueba ampliando la ventana de días."
        )
        st.stop()

    # Compute frequencies
    main_counter, special_counter = compute_frequencies(df)

    # =====================================
    # Numbers analysis
    # =====================================
    all_possible_main = set(range(1, cfg.max_main + 1))
    all_possible_special = set(range(1, cfg.max_special + 1))
    
    drawn_main = set(main_counter.keys())
    drawn_special = set(special_counter.keys())
    
    # Get least frequent numbers (not zero frequency)
    least_frequent_main = sorted(main_counter.items(), key=lambda x: x[1])[:10]
    least_frequent_special = sorted(special_counter.items(), key=lambda x: x[1])[:10]

    st.divider()

    # =====================================
    # Stats cards - Simple 4-column layout
    # =====================================
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Draws", len(df))
    
    with col2:
        st.metric("📅 From", df['date'].max().strftime('%m/%d/%Y'))
    
    with col3:
        st.metric("📅 To", df['date'].min().strftime('%m/%d/%Y'))
    
    with col4:
        most_common = main_counter.most_common(1)[0][0] if main_counter else "—"
        st.metric("🔥 Top Number", f"{int(most_common):02d}" if most_common != "—" else "—")

    st.divider()

    # Generate combinations
    combos = generate_combinations(cfg, main_counter, special_counter, n_combos)
    
    if not combos:
        st.warning(
            "No se pudieron generar combinaciones. Prueba reduciendo el número o cambiando el rango."
        )
        st.stop()
    # =====================================
    # Suggested Combinations - Clean Cards
    # =====================================
    st.subheader(f"🎯 {n_combos} Suggested Combinations")
    
    # Create a nice grid of combinations
    cols = st.columns(min(4, n_combos))
    
    for idx, (main_nums, special) in enumerate(combos):
        with cols[idx % len(cols)]:
            st.markdown(f"""
            <div class="combo-card">
                <h3 style="margin-top: 0;">#{idx + 1}</h3>
                <div style="margin: 10px 0; text-align: center;">
                    {''.join([f'<span class="ball">{n:02d}</span>' for n in main_nums])}
                </div>
                <div style="margin-top: 12px; border-top: 2px dashed #FFD700; padding-top: 12px;">
                    <div style="color: #FF6B9D; font-size: 12px; margin-bottom: 6px;">Special</div>
                    <span class="special-ball">{special:02d}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()

    # =====================================
    # Historial como referencia al final
    # =====================================
    st.subheader("📅 Draw History (Reference)")
    st.dataframe(df.head(30), use_container_width=True, hide_index=True)
    
    st.divider()

    # =====================================
    # Least Frequent Numbers
    # =====================================
    st.subheader("📉 Least Frequent Numbers")
    
    col_less_main, col_less_spec = st.columns(2)
    
    with col_less_main:
        st.markdown("**Main Numbers:**")
        if least_frequent_main:
            for num, freq in least_frequent_main:
                st.markdown(f"- **{num:02d}** ({int(freq)} times)")
        else:
            st.markdown("No data")
    
    with col_less_spec:
        st.markdown("**Special Numbers:**")
        if least_frequent_special:
            for num, freq in least_frequent_special:
                st.markdown(f"- **{num:02d}** ({int(freq)} times)")
        else:
            st.markdown("No data")
    
    st.divider()
    
    # =====================================
    # Final Disclaimer (with humor)
    # =====================================
    col_warning1, col_warning2, col_warning3 = st.columns(3)
    
    with col_warning1:
        st.markdown("""
        <div style="text-align: center; padding: 10px; background: rgba(255, 107, 157, 0.1); border-radius: 8px;">
            <p style="margin: 0; font-size: 14px;">
            ⚠️ **If you're salado (cursed):**<br>
            Just close this app and walk away slowly. 🚶
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_warning2:
        st.caption(
            "📌 Statistical scanner based on frequencies and dispersion patterns. "
            "Each combination has equal mathematical probability."
        )
    
    with col_warning3:
        st.markdown("""
        <div style="text-align: center; padding: 10px; background: rgba(255, 215, 0, 0.1); border-radius: 8px;">
            <p style="margin: 0; font-size: 14px;">
            💰 **Disclaimer:**<br>
            We accept no responsibility if you lose. We were just having fun. 😄
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    random.seed()
    main()
