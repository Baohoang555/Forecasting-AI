

from pathlib import Path
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent

# Nạp cấu hình từ file .env nằm cạnh app.py; biến môi trường hệ thống vẫn được ưu tiên.
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)


def _configured_password():
    """Read the dashboard password from Streamlit secrets or environment."""
    password = os.getenv("DASHBOARD_PASSWORD", "").strip()
    if password:
        return password
    try:
        return str(st.secrets.get("DASHBOARD_PASSWORD", "")).strip()
    except Exception:
        return ""


def check_password():
    expected_password = _configured_password()

    def password_entered():
        if expected_password and st.session_state.get("password", "") == expected_password:
            st.session_state["password_correct"] = True
            st.session_state.pop("password", None)
        else:
            st.session_state["password_correct"] = False

    if not expected_password:
        st.error("Chưa cấu hình mật khẩu dashboard. Hãy tạo file .env cạnh app.py và đặt DASHBOARD_PASSWORD=...")
        st.stop()

    if "password_correct" not in st.session_state:
        st.text_input("Mật khẩu truy cập hệ thống:", type="password", on_change=password_entered, key="password")
        st.caption("Liên hệ Quản trị viên MLOps để nhận quyền truy cập.")
        return False
    if not st.session_state["password_correct"]:
        st.text_input("Mật khẩu truy cập hệ thống:", type="password", on_change=password_entered, key="password")
        st.error("Mật khẩu không chính xác.")
        return False
    return True


# Streamlit yêu cầu set_page_config trước các lệnh hiển thị khác.
st.set_page_config(
    page_title="Canteen Meal Budget Forecasting System",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not check_password():
    st.stop()

# ==============================================================================
# CẤU HÌNH GIAO DIỆN & STYLE CHUẨN DOANH NGHIỆP
# ==============================================================================

# Custom CSS giao diện hiện đại (Executive Modern UI)
st.markdown(
    """
<style>
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 4px;
    }
    .metric-subtitle {
        font-size: 0.8rem;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 8px 8px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)

PRICE_44 = 44_000
PRICE_70 = 70_000


# ==============================================================================
# TẢI DỮ LIỆU AN TOÀN (FALLBACK & SMART CACHING)
# ==============================================================================
@st.cache_data
def load_all_data():
    base_dir = BASE_DIR

    # 1. Dự báo mới nhất: chọn tháng mới nhất có file nhân viên.
    forecast_dir = base_dir / "results/forecast"
    employee_forecasts = sorted(forecast_dir.glob("forecast_*_employee.csv"))
    f_emp_path = employee_forecasts[-1] if employee_forecasts else None
    forecast_month_from_file = f_emp_path.name.replace("forecast_", "").replace("_employee.csv", "") if f_emp_path else None
    f_dept_path = forecast_dir / f"forecast_{forecast_month_from_file}_department.csv" if forecast_month_from_file else None

    df_emp = pd.read_csv(f_emp_path) if f_emp_path and f_emp_path.exists() else pd.DataFrame()
    df_dept = pd.read_csv(f_dept_path) if f_dept_path and f_dept_path.exists() else pd.DataFrame()

    # Luôn lưu lại tên Tổ/Nhóm ban đầu từ máy quẹt thẻ
    if not df_emp.empty and "department" in df_emp.columns:
        df_emp["team_con"] = df_emp["department"]

    # -------------------------------------------------------------------------
    # 👉 ÁNH XẠ TRỰC TIẾP TỪ MÃ SỐ NHÂN VIÊN (Mã NV) SANG PHÒNG BAN
    # -------------------------------------------------------------------------
    mapping_path = base_dir / "data" / "hr.xlsx"
    hr_mapping_available = mapping_path.exists()
    df_map = pd.read_excel(mapping_path) if hr_mapping_available else pd.DataFrame()
    hr_mapped_count = 0

    if not df_map.empty and not df_emp.empty:
        # Chuẩn hóa tên cột file nhân sự
        df_map.columns = df_map.columns.astype(str).str.strip()

        # Tự động nhận diện cột Mã NV và cột Phòng ban
        col_id = next((c for c in df_map.columns if any(k in c.lower() for k in ["mã nv", "ma nv", "manv", "employee_id"])), None)
        col_dept = next((c for c in df_map.columns if any(k in c.lower() for k in ["phòng ban", "phong ban", "bộ phận", "department"])), None)

        if not col_id or not col_dept:
            raise ValueError(
                f"Không tìm thấy cột Mã NV và Phòng ban trong {mapping_path.name}."
            )

        # Chuẩn hóa mã nhân viên và tên phòng ban trước khi ghép.
        df_map[col_id] = pd.to_numeric(df_map[col_id], errors="coerce")
        df_map[col_dept] = df_map[col_dept].astype("string").str.strip().replace("", pd.NA)
        df_emp["employee_id"] = pd.to_numeric(df_emp["employee_id"], errors="coerce")

        map_lookup = (
            df_map[[col_id, col_dept]]
            .rename(columns={col_id: "_hr_employee_id", col_dept: "_hr_department"})
            .dropna()
            .drop_duplicates(subset=["_hr_employee_id"])
        )

        df_emp = df_emp.merge(map_lookup, left_on="employee_id", right_on="_hr_employee_id", how="left")
        hr_mapped_count = int(df_emp["_hr_department"].notna().sum())
        df_emp["department"] = df_emp["_hr_department"].fillna(df_emp["team_con"])
        df_emp.drop(columns=["_hr_employee_id", "_hr_department"], errors="ignore", inplace=True)

    # -------------------------------------------------------------------------
    # 👉 TỰ ĐỘNG TỔNG HỢP LẠI BẢNG PHÒNG BAN THEO ĐỊNH MỨC MỚI
    # -------------------------------------------------------------------------
    if not df_emp.empty:
        df_dept = (
            df_emp.groupby("department")
            .agg(
                so_nhan_vien=("employee_id", "count"),
                tong_du_bao=("pred_cost", "sum")
            )
            .reset_index()
            .sort_values("tong_du_bao", ascending=False)
        )
    else:
        df_dept = pd.DataFrame()

    # 2. Đánh giá ngân sách & Phân khúc
    b_eval_path = base_dir / "results/metrics/company_budget_eval.csv"
    s_eval_path = base_dir / "results/metrics/segment_eval.csv"
    out_path = base_dir / "results/metrics/top_outliers.csv"

    df_budget_eval = pd.read_csv(b_eval_path) if b_eval_path.exists() else pd.DataFrame()
    df_segment = pd.read_csv(s_eval_path) if s_eval_path.exists() else pd.DataFrame()
    df_outliers = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()

    # 3. Model Benchmark
    def read_priority(file_new, file_old):
        if (base_dir / f"results/metrics/{file_old}").exists():
            return pd.read_csv(base_dir / f"results/metrics/{file_old}")
        if (base_dir / f"results/metrics/{file_new}").exists():
            return pd.read_csv(base_dir / f"results/metrics/{file_new}")
        if (base_dir / file_old).exists():
            return pd.read_csv(base_dir / file_old)
        if (base_dir / file_new).exists():
            return pd.read_csv(base_dir / file_new)
        return pd.DataFrame()

    df_ml = read_priority("ml_models_summary_new.csv", "ml_models_summary.csv")
    df_base = read_priority("baselines_summary_new.csv", "baselines_summary.csv")
    df_ens = read_priority("ensemble_multi_summary_new.csv", "ensemble_multi_summary.csv")
    df_dl = read_priority("dl_models_summary_new.csv", "dl_models_summary.csv")
    time_series_path = base_dir / "results" / "metrics" / "time_series_models_summary.csv"
    df_time_series = (
        pd.read_csv(time_series_path) if time_series_path.exists() else pd.DataFrame()
    )
    same_split_path = (
        base_dir / "results" / "metrics" / "same_split_budget_model_comparison.csv"
    )
    df_same_split = (
        pd.read_csv(same_split_path) if same_split_path.exists() else pd.DataFrame()
    )

    # Không tự tạo số liệu benchmark giả khi thiếu file nguồn.

    return {
        "emp": df_emp,
        "dept": df_dept,
        "hr_mapping_available": hr_mapping_available,
        "hr_mapped_count": hr_mapped_count,
        "hr_unmapped_count": max(len(df_emp) - hr_mapped_count, 0),
        "budget_eval": df_budget_eval,
        "segment": df_segment,
        "outliers": df_outliers,
        "ml": df_ml,
        "base": df_base,
        "ens": df_ens,
        "dl": df_dl,
        "time_series": df_time_series,
        "same_split_comparison": df_same_split,
    }


# 👉 ĐẢM BẢO DÒNG NÀY Ở SÁT LỀ TRÁI (KHÔNG THỤT LỀ):
data = load_all_data()
# ==============================================================================
# SIDEBAR — CHUYỂN ĐỔI NGÔN NGỮ & BỘ LỌC ĐIỀU HÀNH
# ==============================================================================
# ==============================================================================
# SIDEBAR — BẢNG ĐIỀU HÀNH & BỘ LỌC ĐA CHIỀU (EXECUTIVE UPGRADE)
# ==============================================================================
with st.sidebar:
    # 1. HEADER THƯƠNG HIỆU & CHUYỂN ĐỔI NGÔN NGỮ
    col_logo, col_lang = st.columns([1, 2])
    with col_logo:
        st.image("https://cdn-icons-png.flaticon.com/512/3448/3448609.png", width=55)
    with col_lang:
        lang_choice = st.radio(
            "Language / Ngôn ngữ:",
            options=["Tiếng Việt", "English"],
            horizontal=True,
            index=0,
            label_visibility="collapsed"
        )
    is_en = (lang_choice == "English")

    def tr(vi: str, en: str) -> str:
        return en if is_en else vi

    st.markdown(f"### 🍱 {tr('CĂNG-TIN EXECUTIVE', 'CANTEEN EXECUTIVE')}")
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-top: -8px; margin-bottom: 12px;">
            <span style="height: 9px; width: 9px; background-color: #10b981; border-radius: 50%; display: inline-block;"></span>
            <span style="font-size: 11.5px; font-weight: 600; color: #059669; text-transform: uppercase; letter-spacing: 0.04em;">
                {tr('Hệ thống sẵn sàng vận hành', 'System Production Ready')}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. THẺ THÔNG TIN KỲ DỰ BÁO & MÔ HÌNH MLOPS
    forecast_month = "N/A"
    if not data["emp"].empty and "forecast_month" in data["emp"].columns:
        forecast_month = str(data["emp"]["forecast_month"].dropna().iloc[0]) if data["emp"]["forecast_month"].notna().any() else "N/A"

    st.markdown(
        f"""
        <div style="background-color: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; margin-bottom: 16px; font-size: 12.5px; line-height: 1.6;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #64748b;">{tr('Kỳ lập ngân sách:', 'Budget Cycle:')}</span>
                <span style="font-weight: 700; color: #1e293b;">{forecast_month}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #64748b;">{tr('Mô hình cốt lõi:', 'Core Model:')}</span>
                <span style="font-weight: 700; color: #2563eb;">CatBoost (M3)</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #64748b;">{tr('Xử lý nhân sự mới:', 'Cold-start:')}</span>
                <span style="font-weight: 600; color: #475569;">Group-Mean</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #64748b;">{tr('Nguồn kiểm định:', 'Ground Truth:')}</span>
                <span style="font-weight: 600; color: #059669;">eCMS Total Amount</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 3. MINI SNAPSHOT METRICS (3 CHỈ SỐ GHIM CỐ ĐỊNH)
    if not data["emp"].empty:
        sb_total_cost = data["emp"]["pred_cost"].sum()
        sb_total_staff = len(data["emp"])
        
        c_sb1, c_sb2 = st.columns(2)
        with c_sb1:
            st.metric(
                label=tr("Tổng Quỹ Dự Trù", "Total Budget"),
                value=f"{sb_total_cost/1e9:.2f} {tr('tỷ', 'B')}"
            )
        with c_sb2:
            st.metric(
                label=tr("Quy Mô Nhân Sự", "Total Staff"),
                value=f"{sb_total_staff:,}"
            )

    st.markdown("---")

    # 4. BỘ LỌC PHÂN TÍCH TOÀN CỤC (GLOBAL FILTERS)
    st.markdown(f"#### 🔍 {tr('Bộ Lọc Dữ Liệu', 'Global Filters')}")

    # Lọc phòng ban thông minh (kèm số nhân sự)
    all_dept_label = tr("Tất cả phòng ban", "All Departments")
    dept_options = [all_dept_label]

    if not data["emp"].empty and "department" in data["emp"].columns:
        dept_counts = data["emp"]["department"].value_counts()
        # Gom danh sách kèm số lượng nhân viên
        dept_options += [
            f"{d} ({dept_counts[d]} {tr('NV', 'staff')})"
            for d in sorted(dept_counts.index.tolist())
        ]

    selected_dept_raw = st.selectbox(
        tr("Chọn phòng ban / Khối:", "Select Department:"),
        options=dept_options,
        index=0,
        help=tr("Lọc dữ liệu trên toàn bộ các biểu đồ và bảng nhân sự", "Filters data across all dashboard tabs")
    )

    # Tách lấy tên phòng ban gốc (bỏ phần đếm số nhân viên trong ngoặc)
    if selected_dept_raw == all_dept_label:
        selected_dept = all_dept_label
    else:
        selected_dept = selected_dept_raw.rsplit(" (", 1)[0]

    # Tra cứu mã nhân viên nhanh
    search_emp = st.text_input(
        tr("Tra cứu Mã Nhân Viên (ID):", "Search Employee ID:"),
        placeholder=tr("Nhập ID: 1001, 1008, 2578...", "e.g., 1001, 1008, 2578..."),
        help=tr("Nhập mã nhân viên để tự động lọc nhanh ở Tab 3", "Type ID to instantly highlight in Tab 3")
    )

    st.markdown("---")

    
# ==============================================================================
# TIÊU ĐỀ TRANG & TABS ĐIỀU HÀNH
# ==============================================================================
st.title(tr("🍱 HỆ THỐNG DỰ BÁO NGÂN SÁCH SUẤT ĂN CĂNG-TIN", "🍱 CANTEEN MEAL BUDGET FORECASTING SYSTEM"))
st.caption(tr(
    f"Báo cáo Kế hoạch Dự trù Ngân sách Vận hành — Tháng {forecast_month}",
    f"Operational Meal Budget & Supply Planning Report — Month {forecast_month}"
))

tab1, tab2, tab3, tab4 = st.tabs([
    tr("🎯 TỔNG QUAN ĐIỀU HÀNH", "🎯 EXECUTIVE OVERVIEW"),
    tr("🏢 CHI PHÍ PHÒNG BAN", "🏢 DEPARTMENT EXPENSES"),
    tr("👤 TRA CỨU NHÂN VIÊN", "👤 EMPLOYEE LOOKUP"),
    tr("🔬 ĐỐI SOÁT MÔ HÌNH", "🔬 MODEL BENCHMARK"),
])

# ------------------------------------------------------------------------------
# TAB 1: TỔNG QUAN ĐIỀU HÀNH (EXECUTIVE OVERVIEW)
# ------------------------------------------------------------------------------
with tab1:
    total_budget = data["emp"]["pred_cost"].sum() if not data["emp"].empty else 0
    total_staff = len(data["emp"]) if not data["emp"].empty else 0
    total_n44 = data["emp"]["pred_N44"].sum() if not data["emp"].empty else 0
    total_n70 = data["emp"]["pred_N70"].sum() if not data["emp"].empty else 0
    total_meals = total_n44 + total_n70
    avg_per_staff = total_budget / total_staff if total_staff > 0 else 0
    avg_per_day = avg_per_staff / 22 if avg_per_staff > 0 else 0

    cost_n44 = total_n44 * PRICE_44
    cost_n70 = total_n70 * PRICE_70

    n_catboost = (data["emp"]["source"] == "CatBoost").sum() if not data["emp"].empty else 0
    n_fallback = data["emp"]["source"].astype(str).str.contains("Group-Mean", case=False, na=False).sum() if not data["emp"].empty and "source" in data["emp"].columns else 0
    ai_coverage_pct = (n_catboost / total_staff * 100) if total_staff > 0 else 0

    mom_diff = 0
    mom_pct = 0.0
    has_mom = False
    if not data["budget_eval"].empty:
        df_b_sorted = data["budget_eval"].sort_values("month")
        last_actual = df_b_sorted["Tien_Thuc_Te"].iloc[-1]
        last_month_name = df_b_sorted["month"].iloc[-1]
        mom_diff = total_budget - last_actual
        mom_pct = (mom_diff / last_actual) * 100 if pd.notna(last_actual) and last_actual != 0 else np.nan
        has_mom = True

    backtest_variance_pct = np.nan
    backtest_month = "N/A"
    if not data["budget_eval"].empty and "Ti_Le_Lech_%" in data["budget_eval"].columns:
        _latest_eval = data["budget_eval"].sort_values("month").iloc[-1]
        backtest_variance_pct = pd.to_numeric(_latest_eval["Ti_Le_Lech_%"], errors="coerce")
        backtest_month = str(_latest_eval.get("month", "N/A"))
    backtest_variance_label = f"{backtest_variance_pct:+.2f}%" if pd.notna(backtest_variance_pct) else "N/A"
    mom_pct_label = f"{mom_pct:+.2f}%" if pd.notna(mom_pct) else "N/A"

    # Khối tóm tắt điều hành
    briefing_title = tr("Báo Cáo Trình Ban Lãnh Đạo & Kế Toán Quản Trị", "Executive & Management Accounting Brief")
    briefing_h2 = tr(f"Kế Hoạch Ngân Sách Dự Trù Suất Ăn: <span style='color: #38bdf8;'>{total_budget:,.0f} đ</span>",
                     f"Projected Canteen Meal Budget: <span style='color: #38bdf8;'>{total_budget:,.0f} VND</span>")
    briefing_body = tr(
        f"• <b>Quy mô & Định mức:</b> Phục vụ <b>{total_staff:,} nhân sự</b> với định mức bình quân <b>{avg_per_staff:,.0f} đ/người/tháng</b> (~{avg_per_day:,.0f} đ/ngày làm việc).<br>"
        f"• <b>Biến động ngân sách:</b> Dự kiến {'tăng nhẹ' if mom_diff >= 0 else 'giảm'} <b>{abs(mom_diff):,.0f} đ ({mom_pct_label})</b> so với thực chi tháng {last_month_name if has_mom else 'trước'}.<br>"
        f"• <b>Độ tin cậy thuật toán:</b> <b>{ai_coverage_pct:.1f}%</b> nhân sự được dự báo bằng CatBoost M3; sai lệch ngân sách kiểm định gần nhất ({backtest_month}) là <b>{backtest_variance_label}</b>.",
        f"• <b>Scope & Quota:</b> Serving <b>{total_staff:,} employees</b> at an average of <b>{avg_per_staff:,.0f} VND/person/month</b> (~{avg_per_day:,.0f} VND/workday).<br>"
        f"• <b>Budget Variance:</b> Projected {'slight increase' if mom_diff >= 0 else 'decrease'} of <b>{abs(mom_diff):,.0f} VND ({mom_pct_label})</b> MoM compared to month {last_month_name if has_mom else 'prior'}.<br>"
        f"• <b>Algorithmic Confidence:</b> <b>{ai_coverage_pct:.1f}%</b> employees forecasted by CatBoost M3; latest backtest budget variance ({backtest_month}) is <b>{backtest_variance_label}</b>."
    )

    st.markdown(
        f"""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); color: white; padding: 24px; border-radius: 16px; margin-bottom: 24px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span style="background-color: #3b82f6; color: white; font-size: 11px; font-weight: 700; padding: 4px 12px; border-radius: 20px; text-transform: uppercase;">{briefing_title}</span>
            <span style="font-size: 13px; color: #94a3b8;">{tr('Kỳ dự báo:', 'Period:')} <b>{forecast_month}</b></span>
        </div>
        <h2 style="color: white; margin: 0 0 10px 0; font-size: 22px; font-weight: 700;">{briefing_h2}</h2>
        <p style="color: #cbd5e1; font-size: 14px; line-height: 1.6; margin: 0;">{briefing_body}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(f"#### 📌 {tr('Các Chỉ Số Tài Chính & Vận Hành Trọng Yếu', 'Key Financial & Operational Metrics')}")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{tr('Tổng Ngân Sách', 'Total Budget')}</div>
            <div class="metric-value">{total_budget/1e9:.2f} {tr('tỷ', 'B')}</div>
            <div class="metric-subtitle" style="color: {'#10b981' if mom_diff >= 0 else '#ef4444'};">{mom_pct_label} vs {last_month_name if has_mom else 'MoM'}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{tr('Quy Mô Nhân Sự', 'Headcount')}</div>
            <div class="metric-value">{total_staff:,}</div>
            <div class="metric-subtitle" style="color: #3b82f6;">{tr('Số nhân sự trong file dự báo', 'Employees in forecast file')}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{tr('Tổng Suất Ăn', 'Total Meals')}</div>
            <div class="metric-value">{total_meals:,}</div>
            <div class="metric-subtitle" style="color: #64748b;">TB: {(total_meals/total_staff) if total_staff else 0:.1f} {tr('suất/người', 'meals/staff')}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{tr('Chi Phí / Người', 'Cost / Employee')}</div>
            <div class="metric-value">{avg_per_staff/1e3:,.0f}k</div>
            <div class="metric-subtitle" style="color: #64748b;">~{avg_per_day/1e3:,.0f}k {tr('đ/ngày công', 'VND/workday')}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{tr('Tỷ Lệ Dùng CatBoost', 'CatBoost Coverage')}</div>
            <div class="metric-value" style="color: #2563eb;">{ai_coverage_pct:.1f}%</div>
            <div class="metric-subtitle" style="color: #64748b;">{n_catboost:,} ML | {n_fallback:,} Fallback</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{tr('Sai Lệch Kiểm Định', 'Backtest Variance')}</div>
            <div class="metric-value" style="color: #059669;">{backtest_variance_label}</div>
            <div class="metric-subtitle" style="color: #64748b;">{tr('Kỳ kiểm định: ' + backtest_month, 'Backtest month: ' + backtest_month)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_chart_left, col_chart_right = st.columns([3, 2])

    with col_chart_left:
        st.subheader(tr("📈 Xu Hướng Ngân Sách: Lịch Sử Thực Tế & Tháng Tới", "📈 Budget Trend: Historical Actuals vs Upcoming Forecast"))
        if not data["budget_eval"].empty:
            df_b = data["budget_eval"].copy()
            label_actual = tr("Thực Tế Chi (eCMS)", "Actual Expense (eCMS)")
            label_fc = tr("Dự Báo AI (CatBoost)", "AI Forecast (CatBoost)")

            # Hàm chuẩn hóa nhãn tháng để tránh bị Plotly ép sang trục ngày liên tục
            def format_month_label(m_str):
                parts = str(m_str).split("-")
                if len(parts) == 2:
                    return f"{tr('Tháng', 'Month')} {parts[1]}/{parts[0]}"
                return f"{tr('Tháng', 'Month')} {m_str}"

            # 1. Chuỗi lịch sử thực tế
            hist_series = pd.DataFrame({
                "Kỳ": df_b["month"].apply(format_month_label),
                "Số_Tiền": df_b["Tien_Thuc_Te"],
                "Phân_Loại": label_actual,
            })

            # 2. Điểm dự báo tháng tới (cột màu xanh)
            fc_label = f"{format_month_label(forecast_month)} ({tr('Dự Báo AI', 'AI Forecast')})"
            forecast_point = pd.DataFrame([{
                "Kỳ": fc_label,
                "Số_Tiền": total_budget,
                "Phân_Loại": label_fc,
            }])

            comb_trend = pd.concat([hist_series, forecast_point], ignore_index=True)

            fig_trend = px.bar(
                comb_trend,
                x="Kỳ",
                y="Số_Tiền",
                color="Phân_Loại",
                text="Số_Tiền",
                color_discrete_map={
                    label_actual: "#94a3b8",
                    label_fc: "#2563eb",
                },
            )
            fig_trend.update_traces(
                texttemplate="%{text:,.0f} đ", textposition="outside"
            )
            # 👉 KHÓA CHẶT type="category" ĐỂ CỘT DỰ BÁO LUÔN ĐƯỢC HIỂN THỊ
            fig_trend.update_layout(
                xaxis=dict(
                    type="category",
                    title=tr("Kỳ Ngân Sách", "Budget Period")
                ),
                yaxis=dict(
                    title=tr("Tổng Tiền (VNĐ)", "Total Cost (VND)"),
                    range=[0, max(comb_trend["Số_Tiền"].max() * 1.22, 1.6e9)],
                ),
                height=380,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            st.caption(tr(
                "Lưu ý: các tháng lịch sử là số thực tế; điểm dự báo kỳ tới là kết quả mô hình. Hãy đối chiếu cùng kỳ và cùng phạm vi trước khi kết luận xu hướng.",
                "Note: historical months show actuals while the upcoming point is a model forecast. Compare like-for-like periods and scopes before interpreting the trend."
            ))
        else:
            st.info(tr("Chưa có dữ liệu lịch sử đối soát ngân sách.", "Historical budget audit data not available."))
    with col_chart_right:
        st.subheader(tr("🍱 Cơ Cấu Suất Ăn & Doanh Số Dự Kiến", "🍱 Meal Mix & Expected Catering Procurement"))
        if total_meals > 0:
            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=[
                            f"{tr('Việt/Chay', 'VN/Veg')} 44k ({total_n44:,})",
                            f"{tr('Món Âu', 'Western')} 70k ({total_n70:,})",
                        ],
                        values=[cost_n44, cost_n70],
                        hole=0.58,
                        marker_colors=["#3b82f6", "#f59e0b"],
                        textinfo="percent+label",
                    )
                ]
            )
            fig_donut.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320, showlegend=False)
            st.plotly_chart(fig_donut, use_container_width=True)

            procure_df = pd.DataFrame([
                {
                    tr("Loại Suất Ăn", "Meal Category"): tr("Cơm Việt & Món Chay", "Vietnamese & Vegetarian"),
                    tr("Đơn Giá", "Unit Price"): f"{PRICE_44:,.0f} đ",
                    tr("Số Lượng", "Quantity"): f"{total_n44:,} {tr('suất', 'meals')}",
                    tr("Thành Tiền", "Subtotal"): f"{cost_n44:,.0f} đ",
                    tr("Tỷ Trọng", "Share"): f"{(cost_n44/total_budget*100) if total_budget else 0:.1f}%",
                },
                {
                    tr("Loại Suất Ăn", "Meal Category"): tr("Món Âu Cao Cấp", "Western Premium Meal"),
                    tr("Đơn Giá", "Unit Price"): f"{PRICE_70:,.0f} đ",
                    tr("Số Lượng", "Quantity"): f"{total_n70:,} {tr('suất', 'meals')}",
                    tr("Thành Tiền", "Subtotal"): f"{cost_n70:,.0f} đ",
                    tr("Tỷ Trọng", "Share"): f"{(cost_n70/total_budget*100) if total_budget else 0:.1f}%",
                },
            ])
            st.dataframe(procure_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Phân khúc hành vi
    st.subheader(tr("👥 Phân Khúc Hành Vi Tiêu Dùng Của Cán Bộ Công Nhân Viên", "👥 Employee Consumption Behavior Segmentation"))
    if not data["emp"].empty:
        df_seg_emp = data["emp"].copy()
        bins = [-1, 0, 300000, 800000, float("inf")]
        seg_labels = [
            tr("Không ăn (0 đ)", "Zero meal (0 VND)"),
            tr("Ăn ít (<300k)", "Low (<300k VND)"),
            tr("Ăn vừa (300k - 800k)", "Moderate (300k - 800k VND)"),
            tr("Ăn nhiều (>=800k)", "Heavy (>=800k VND)"),
        ]
        df_seg_emp["Phan_Khuc"] = pd.cut(df_seg_emp["pred_cost"], bins=bins, labels=seg_labels)

        seg_stats = (
            df_seg_emp.groupby("Phan_Khuc", observed=False)
            .agg(
                So_Nhan_Su=("employee_id", "count"),
                Tong_Ngan_Sach=("pred_cost", "sum"),
                TB_Nhan_Su=("pred_cost", "mean"),
            )
            .reset_index()
        )
        seg_stats["Ty_Le_Nhan_Su"] = (seg_stats["So_Nhan_Su"] / total_staff * 100).round(1)
        seg_stats["Ty_Le_Ngan_Sach"] = (seg_stats["Tong_Ngan_Sach"] / total_budget * 100).round(1) if total_budget else 0.0

        col_seg_table, col_seg_bar = st.columns([3, 2])

        with col_seg_table:
            disp_seg = seg_stats.copy()
            disp_seg.columns = [
                tr("Phân Khúc Tiêu Dùng", "Consumption Segment"),
                tr("Số Nhân Sự", "Headcount"),
                tr("Tỷ Lệ NV (%)", "Staff Share (%)"),
                tr("Tổng Chi Phí (VNĐ)", "Total Cost (VND)"),
                tr("Tỷ Trọng Ngân Sách (%)", "Budget Share (%)"),
                tr("Định Mức TB/Người", "Avg/Staff (VND)"),
            ]
            st.dataframe(
                disp_seg.style.format({
                    disp_seg.columns[1]: "{:,}",
                    disp_seg.columns[2]: "{:.1f}%",
                    disp_seg.columns[3]: "{:,.0f} đ",
                    disp_seg.columns[4]: "{:.1f}%",
                    disp_seg.columns[5]: "{:,.0f} đ",
                }),
                hide_index=True,
                use_container_width=True,
            )

        with col_seg_bar:
            fig_seg = px.bar(
                seg_stats,
                x="Phan_Khuc",
                y="Tong_Ngan_Sach",
                color="Phan_Khuc",
                text="Ty_Le_Ngan_Sach",
                title=tr("Tỷ Trọng Đóng Góp Ngân Sách Theo Nhóm", "Budget Contribution by Segment"),
                color_discrete_sequence=["#94a3b8", "#38bdf8", "#3b82f6", "#1d4ed8"],
            )
            fig_seg.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_seg.update_layout(
                showlegend=False,
                height=300,
                yaxis=dict(title=tr("Tổng Tiền (VNĐ)", "Total (VND)")),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_seg, use_container_width=True)

    st.markdown("---")

    # Khuyến nghị điều hành
    st.subheader(tr("🎯 Khuyến Nghị Điều Hành & Vận Hành Thực Tế", "🎯 Operational & Governance Recommendations"))
    rec_c1, rec_c2, rec_c3 = st.columns(3)

    with rec_c1:
        st.markdown(f"""
        <div style="background-color: #f8fafc; border-left: 4px solid #2563eb; padding: 16px; border-radius: 8px;">
            <b style="color: #1e293b;">{tr('1. Phòng Kế Toán / Tài Chính', '1. Finance & Accounting')}</b>
            <p style="font-size: 13px; color: #475569; margin-top: 8px; line-height: 1.5;">
                {tr(f'• Dùng mức dự báo hiện tại <b>{total_budget:,.0f} VNĐ</b> làm cơ sở tham khảo; cần được Tài chính phê duyệt trước khi giải ngân.',
                    f'• Use the current forecast of <b>{total_budget:,.0f} VND</b> as a planning reference; Finance approval is required before disbursement.')}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with rec_c2:
        st.markdown(f"""
        <div style="background-color: #f8fafc; border-left: 4px solid #f59e0b; padding: 16px; border-radius: 8px;">
            <b style="color: #1e293b;">{tr('2. Ban Hành Chính & Nhà Thầu Căng-Tin', '2. Administration & Canteen Contractor')}</b>
            <p style="font-size: 13px; color: #475569; margin-top: 8px; line-height: 1.5;">
                {tr(f'• Dùng dự báo đặt hàng tham khảo: <b>{total_n44:,} suất Việt/Chay</b> và <b>{total_n70:,} suất Âu</b> ({(total_n70/total_meals*100) if total_meals else 0:.1f}% tổng suất).<br>• Xác nhận lại với đăng ký thực tế trước khi chốt đơn.',
                    f'• Forecast planning quantity: <b>{total_n44:,} VN/Veg meals</b> and <b>{total_n70:,} Western meals</b> ({(total_n70/total_meals*100) if total_meals else 0:.1f}% of meals).<br>• Confirm against actual registrations before finalizing orders.')}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with rec_c3:
        st.markdown(f"""
        <div style="background-color: #f8fafc; border-left: 4px solid #10b981; padding: 16px; border-radius: 8px;">
            <b style="color: #1e293b;">{tr('3. Quản Trị Nhân Sự (HR)', '3. Human Resources (HR)')}</b>
            <p style="font-size: 13px; color: #475569; margin-top: 8px; line-height: 1.5;">
                {tr(f'• Rà soát <b>{n_fallback:,} nhân sự dùng Group-Mean</b> để xác nhận đăng ký suất ăn.<br>• Đối chiếu <b>{int((data["emp"]["pred_cost"] == 0).sum()) if not data["emp"].empty else 0} nhân sự dự báo 0 đồng</b> với HR trước khi thay đổi đơn đặt suất ăn.',
                    f'• Review <b>{n_fallback:,} staff using Group-Mean</b> to confirm meal registration.<br>• Verify <b>{int((data["emp"]["pred_cost"] == 0).sum()) if not data["emp"].empty else 0} staff forecast at zero cost</b> with HR before changing meal orders.')}
            </p>
        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 2: CHI PHÍ THEO PHÒNG BAN (BAO GỒM MỌI QUY MÔ ĐƠN VỊ)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# TAB 2: CHI PHÍ THEO PHÒNG BAN (ĐÃ LÀM SẠCH GIAO DIỆN)
# ------------------------------------------------------------------------------
with tab2:
    st.markdown(f"### 🏢 {tr('Phân Bổ Ngân Sách & Định Mức Chi Tiết Theo Phòng Ban', 'Detailed Budget Allocation by Department')}")
    st.caption(tr(
        f"Dữ liệu dự báo chi phí suất ăn kỳ Tháng {forecast_month} — Cơ sở giao chỉ tiêu ngân sách bộ phận",
        f"Meal expense forecast for Month {forecast_month} — Baseline for departmental quota allocation"
    ))

    if not data["hr_mapping_available"]:
        st.warning(tr(
            "Không tìm thấy data/hr.xlsx; chi phí vẫn đang nhóm theo tên team trong file dự báo.",
            "data/hr.xlsx was not found; expenses are still grouped by team names from the forecast."
        ))
    elif data["hr_unmapped_count"]:
        st.warning(tr(
            f"{data['hr_unmapped_count']:,} nhân viên không khớp Mã NV trong hr.xlsx; các trường hợp này vẫn giữ tên team gốc.",
            f"{data['hr_unmapped_count']:,} employees have no matching ID in hr.xlsx and remain under their original team names."
        ))

    if not data["emp"].empty:
        # Cột mặc định luôn là Phòng ban lớn chính thức
        active_col = "department"

        # 1. TỔNG HỢP DỮ LIỆU ĐA CHIỀU CẤP ĐƠN VỊ
        df_dept_full = (
            data["emp"].groupby(active_col)
            .agg(
                so_nhan_vien=("employee_id", "count"),
                tong_du_bao=("pred_cost", "sum"),
                suat_n44=("pred_N44", "sum"),
                suat_n70=("pred_N70", "sum"),
                dinh_muc_tb=("pred_cost", "mean"),
            )
            .reset_index()
            .rename(columns={active_col: "department"})
        )

        # Giữ toàn bộ phòng ban, bao gồm cả các đơn vị chỉ có một nhân viên.
        df_dept_full = df_dept_full.reset_index(drop=True)

        total_co_cost = float(df_dept_full["tong_du_bao"].sum())
        df_dept_full["ty_trong_quy_%"] = (
            df_dept_full["tong_du_bao"] / total_co_cost * 100
        ).round(2) if total_co_cost else 0.0
        df_dept_full["tien_n44"] = df_dept_full["suat_n44"] * PRICE_44
        df_dept_full["tien_n70"] = df_dept_full["suat_n70"] * PRICE_70
        df_dept_full["ty_le_au_%"] = np.where(
            df_dept_full["tong_du_bao"] > 0,
            (df_dept_full["tien_n70"] / df_dept_full["tong_du_bao"] * 100).round(1),
            0.0
        )
        df_dept_full = df_dept_full.sort_values("tong_du_bao", ascending=False).reset_index(drop=True)

        # 3. 4 THẺ CHỈ SỐ ĐIỀU HÀNH
        df_dept_full = df_dept_full.sort_values("tong_du_bao", ascending=False).reset_index(drop=True)
        top_k_count = min(10, len(df_dept_full))
        top1_name = df_dept_full.iloc[0]["department"] if not df_dept_full.empty else "N/A"
        top1_cost = df_dept_full.iloc[0]["tong_du_bao"] if not df_dept_full.empty else 0
        top1_staff = df_dept_full.iloc[0]["so_nhan_vien"] if not df_dept_full.empty else 0
        top10_share = (df_dept_full.head(top_k_count)["tong_du_bao"].sum() / total_co_cost * 100) if total_co_cost else 0.0
        dept_staff_total = int(df_dept_full["so_nhan_vien"].sum())
        avg_dept_cost_per_person = total_co_cost / dept_staff_total if dept_staff_total else 0.0

        c_d1, c_d2, c_d3, c_d4 = st.columns(4)
        with c_d1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{tr('Tổng Số Phòng Ban Chuẩn', 'Standard Departments')}</div>
                <div class="metric-value">{len(df_dept_full)} {tr('đơn vị', 'depts')}</div>
                <div class="metric-subtitle">{tr('Bao gồm mọi quy mô phòng ban', 'Includes departments of all sizes')}</div>
            </div>
            """, unsafe_allow_html=True)

        with c_d2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{tr('Mức Tập Trung Ngân Sách', 'Budget Concentration')}</div>
                <div class="metric-value" style="color: #2563eb;">{top10_share:.1f}%</div>
                <div class="metric-subtitle">{tr(f'Tập trung tại Top {top_k_count} đơn vị', f'Concentrated in Top {top_k_count} units')}</div>
            </div>
            """, unsafe_allow_html=True)

        with c_d3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{tr('Đơn Vị Chi Phí Cao Nhất', 'Top Expensed Department')}</div>
                <div class="metric-value" style="font-size: 1.35rem;">{top1_name}</div>
                <div class="metric-subtitle" style="color: #64748b;">{top1_cost:,.0f} đ ({top1_staff} {tr('NV', 'staff')})</div>
            </div>
            """, unsafe_allow_html=True)

        with c_d4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{tr('Định Mức TB Toàn Công Ty', 'Company Avg Quota')}</div>
                <div class="metric-value" style="color: #059669;">{avg_dept_cost_per_person:,.0f} đ</div>
                <div class="metric-subtitle">{tr(f'Bình quân dữ liệu: {avg_dept_cost_per_person:,.0f} đ/người', f'Data average: {avg_dept_cost_per_person:,.0f} VND/staff')}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # 4. ĐIỀU KHIỂN BỘ LỌC BIỂU ĐỒ
        c_ctrl1, c_ctrl2 = st.columns([2, 2])
        with c_ctrl1:
            chart_limit_opts = [x for x in [10, 15, 20, 30] if x < len(df_dept_full)] + [len(df_dept_full)]
            chart_limit = st.selectbox(
                tr("Phạm vi hiển thị biểu đồ:", "Chart Display Limit:"),
                options=chart_limit_opts,
                format_func=lambda x: f"Top {x} {tr('Đơn Vị Lớn Nhất', 'Largest Units')}" if x < len(df_dept_full) else tr(f"Toàn Bộ {len(df_dept_full)} Đơn Vị", f"All {len(df_dept_full)} Units"),
                index=0
            )
        with c_ctrl2:
            sort_metric = st.selectbox(
                tr("Tiêu chí xếp hạng biểu đồ:", "Chart Ranking Metric:"),
                options=[
                    tr("Tổng Ngân Sách (VNĐ)", "Total Budget (VND)"),
                    tr("Định Mức Bình Quân/Người (VNĐ)", "Average per Staff (VND)"),
                    tr("Số Lượng Nhân Sự", "Headcount"),
                    tr("Tỷ Lệ Món Âu (%)", "Western Meal Share (%)")
                ],
                index=0
            )

        sort_map = {
            tr("Tổng Ngân Sách (VNĐ)", "Total Budget (VND)"): "tong_du_bao",
            tr("Định Mức Bình Quân/Người (VNĐ)", "Average per Staff (VND)"): "dinh_muc_tb",
            tr("Số Lượng Nhân Sự", "Headcount"): "so_nhan_vien",
            tr("Tỷ Lệ Món Âu (%)", "Western Meal Share (%)"): "ty_le_au_%"
        }
        df_chart_data = df_dept_full.sort_values(sort_map[sort_metric], ascending=False).head(chart_limit).copy()

        # HAI BIỂU ĐỒ CHUYÊN SÂU
        col_chart_left, col_chart_right = st.columns([3, 2])

        with col_chart_left:
            st.subheader(f"📊 {tr(f'Cơ Cấu Chi Phí Theo Loại Món (Top {chart_limit} Đơn Vị)', f'Cost Breakdown by Meal Type (Top {chart_limit} Units)')}")
            chart_melted = pd.melt(
                df_chart_data,
                id_vars=["department"],
                value_vars=["tien_n44", "tien_n70"],
                var_name="Loai_Mon",
                value_name="So_Tien"
            )
            chart_melted["Loai_Mon"] = chart_melted["Loai_Mon"].map({
                "tien_n44": tr("Cơm Việt / Chay (44k)", "Vietnamese / Veg (44k)"),
                "tien_n70": tr("Món Âu Cao Cấp (70k)", "Western Premium (70k)")
            })

            fig_stacked = px.bar(
                chart_melted,
                x="So_Tien",
                y="department",
                color="Loai_Mon",
                orientation="h",
                labels={"So_Tien": tr("Chi Phí Dự Báo (VNĐ)", "Forecast Cost (VND)"), "department": tr("Phòng Ban / Đơn Vị", "Department / Unit"), "Loai_Mon": tr("Loại Suất Ăn", "Meal Category")},
                color_discrete_map={
                    tr("Cơm Việt / Chay (44k)", "Vietnamese / Veg (44k)"): "#3b82f6",
                    tr("Món Âu Cao Cấp (70k)", "Western Premium (70k)"): "#f59e0b"
                }
            )
            fig_stacked.update_layout(
                yaxis=dict(autorange="reversed"),
                barmode="stack",
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_stacked, use_container_width=True)

        with col_chart_right:
            st.subheader(f"🎯 {tr('Ma Trận Định Vị: Quy Mô vs Định Mức', 'Positioning Matrix: Headcount vs Quota')}")
            st.caption(tr("Bong bóng = Tổng ngân sách | Màu sắc = Tỷ lệ chi tiêu món Âu", "Bubble size = Total budget | Color = Western meal share"))
            fig_matrix = px.scatter(
                df_dept_full,
                x="so_nhan_vien",
                y="dinh_muc_tb",
                size="tong_du_bao",
                color="ty_le_au_%",
                hover_name="department",
                labels={
                    "so_nhan_vien": tr("Số Lượng Nhân Sự", "Headcount"),
                    "dinh_muc_tb": tr("Định Mức TB/Người (VNĐ)", "Avg Quota / Staff (VND)"),
                    "ty_le_au_%": tr("Tỷ Lệ Món Âu (%)", "Western Meal Share (%)"),
                    "tong_du_bao": tr("Tổng Ngân Sách", "Total Budget")
                },
                color_continuous_scale="Viridis",
            )
            fig_matrix.add_hline(
                y=avg_dept_cost_per_person,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=tr(f"Bình quân: {avg_dept_cost_per_person/1000:.0f}k", f"Average: {avg_dept_cost_per_person/1000:.0f}k"),
                annotation_position="bottom right"
            )
            fig_matrix.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_matrix, use_container_width=True)

        st.markdown("---")

        # 5. BẢNG DỮ LIỆU ĐỊNH MỨC CHI TIẾT
        st.subheader(f"📋 {tr('Bảng Báo Cáo Định Mức Chi Tiết', 'Detailed Departmental Quota Table')}")
        search_dept_kw = st.text_input(tr("🔍 Lọc nhanh theo tên phòng ban:", "🔍 Quick filter by dept name:"), placeholder="Admin, Finance, Hull, Piping, Team...")
        # Chỉ loại đơn vị có đúng 1 nhân sự khỏi BẢNG báo cáo chi tiết.
        # Không lọc df_dept_full gốc để không làm thay đổi KPI và biểu đồ tổng quan.
        display_table = df_dept_full[df_dept_full["so_nhan_vien"] > 1].copy()
        excluded_single_staff = int((df_dept_full["so_nhan_vien"] == 1).sum())
        st.caption(tr(
            f"Bảng chỉ hiển thị phòng ban/bộ phận có từ 2 nhân sự trở lên. Đã ẩn {excluded_single_staff} đơn vị chỉ có 1 người; các KPI và biểu đồ tổng quan vẫn tính toàn bộ dữ liệu.",
            f"This table shows departments/units with at least 2 staff. Hidden {excluded_single_staff} single-person units; overview KPIs and charts still use the full dataset."
        ))
        if search_dept_kw.strip() != "":
            display_table = display_table[display_table["department"].str.contains(search_dept_kw.strip(), case=False, na=False)]

        table_output = display_table[[
            "department", "so_nhan_vien", "suat_n44", "suat_n70",
            "tong_du_bao", "ty_trong_quy_%", "dinh_muc_tb", "ty_le_au_%"
        ]].copy()
        table_output.columns = [
            tr("Phòng Ban / Bộ Phận", "Department"),
            tr("Số Nhân Sự", "Headcount"),
            tr("Suất Việt/Chay (44k)", "VN/Veg Meals (44k)"),
            tr("Suất Âu (70k)", "Western Meals (70k)"),
            tr("Tổng Dự Báo (VNĐ)", "Total Forecast (VND)"),
            tr("Tỷ Trọng Quỹ (%)", "Fund Share (%)"),
            tr("Định Mức TB/Người (VNĐ)", "Avg Quota/Staff (VND)"),
            tr("Tỷ Lệ Món Âu (%)", "Western Share (%)"),
        ]

        st.dataframe(
            table_output.style.format({
                table_output.columns[1]: "{:,}",
                table_output.columns[2]: "{:,}",
                table_output.columns[3]: "{:,}",
                table_output.columns[4]: "{:,.0f} đ",
                table_output.columns[5]: "{:.2f}%",
                table_output.columns[6]: "{:,.0f} đ",
                table_output.columns[7]: "{:.1f}%",
            }).highlight_max(subset=[table_output.columns[4]], color="#dbeafe"),
            use_container_width=True,
            height=350,
        )

        csv_dept_export = table_output.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label=f"📥 {tr('Tải Báo Cáo Định Mức Phòng Ban (CSV)', 'Download Department Budget Report (CSV)')}",
            data=csv_dept_export,
            file_name=f"Department_Budget_Report_{forecast_month}.csv",
            mime="text/csv",
        )

        st.markdown("---")

        # 6. 3 NHÓM ĐẶC THÙ
        st.subheader(f"💡 {tr('Chú ý', 'Caution')}")
        tab_insight1, tab_insight2, tab_insight3 = st.tabs([
            tr(f"🏆 Top {top_k_count} Đơn Vị Trọng Điểm", f"🏆 Top {top_k_count} Major Units"),
            tr("🥩 Đơn Vị Tiêu Thụ Món Âu Đột Biến", "🥩 High Western Meal Consumption"),
            tr("⚠️ Đơn Vị Định Mức Thấp Bất Thường", "⚠️ Exceptionally Low Quota Units")
        ])

        with tab_insight1:
            top_k_df = df_dept_full.head(top_k_count)
            st.markdown(tr(
                f"- **Chi phối ngân sách:** Top {top_k_count} đơn vị dẫn đầu có tổng chi phí là **{top_k_df['tong_du_bao'].sum():,.0f} đ**, chiếm **{top10_share:.1f}%** toàn bộ ngân sách công ty.<br>"
                f"- **Đặc điểm:** Đây là các đơn vị có tổng chi phí dự báo cao trong kỳ; dữ liệu dự báo không xác nhận tỷ lệ chuyên cần thực tế.<br>"
                f"- **Khuyến nghị:** Đối chiếu danh sách đăng ký/chấm công thực tế trước khi chốt định mức.",
                f"- **Budget Concentration:** Top {top_k_count} units account for **{top_k_df['tong_du_bao'].sum():,.0f} VND**, representing **{top10_share:.1f}%** of company forecast budget.<br>"
                f"- **Profile:** These units have the highest forecast costs; forecast data does not verify actual attendance.<br>"
                f"- **Action:** Reconcile forecasts against actual registration/attendance before finalizing quotas."
            ), unsafe_allow_html=True)

        with tab_insight2:
            high_au = df_dept_full[df_dept_full["suat_n70"] >= 15].sort_values("ty_le_au_%", ascending=False)
            if high_au.empty:
                high_au = df_dept_full[df_dept_full["suat_n70"] > 0].sort_values("ty_le_au_%", ascending=False).head(10)
            st.markdown(tr(
                "- **Các đơn vị tiêu thụ món Âu đáng chú ý:** Tập trung chủ yếu ở các bộ phận văn phòng, chuyên gia và quản lý.<br>"
                "- **Khuyến nghị:** Chuẩn bị nguyên liệu bò/mì Ý tập trung đúng ca làm việc của các bộ phận này, tránh dư thừa tại các line cơm thông thường.",
                "- **Notable Western Meal Units:** Primarily concentrated in office, expat, and management divisions.<br>"
                "- **Action:** Focus Western meal ingredients on these specific work shifts to eliminate expensive waste."
            ), unsafe_allow_html=True)
            st.dataframe(
                high_au[["department", "so_nhan_vien", "suat_n70", "tien_n70", "ty_le_au_%"]].rename(columns={
                    "department": tr("Phòng Ban", "Department"), "so_nhan_vien": tr("Số NV", "Headcount"),
                    "suat_n70": tr("Số Suất Âu", "Western Meals"), "tien_n70": tr("Tiền Món Âu (đ)", "Western Cost (VND)"),
                    "ty_le_au_%": tr("Tỷ Lệ Món Âu (%)", "Western Share (%)")
                }).style.format({
                    tr("Tiền Món Âu (đ)", "Western Cost (VND)"): "{:,.0f} đ",
                    tr("Tỷ Lệ Món Âu (%)", "Western Share (%)"): "{:.1f}%"
                }),
                hide_index=True,
                use_container_width=True
            )

        with tab_insight3:
            low_cost = df_dept_full[(df_dept_full["dinh_muc_tb"] < 500000) & (df_dept_full["so_nhan_vien"] >= 2)].sort_values("dinh_muc_tb")
            st.markdown(tr(
                "- **Đơn vị định mức thấp (< 500k/người):** Các nhóm nhân sự làm ngoài hiện trường, bảo vệ đi ca có phụ cấp riêng, hoặc đối tượng khách không ăn thường xuyên.<br>"
                "- **Khuyến nghị:** Hành chính không đặt suất ăn cố định; chỉ nấu theo số lượt quẹt thẻ thực tế phát sinh trong ngày.",
                "- **Low Quota Units (< 500k/staff):** Fieldwork teams, security with separate allowances, or irregular guest diners.<br>"
                "- **Action:** Avoid pre-ordering fixed meal portions; prepare meals strictly per daily check-in logs."
            ), unsafe_allow_html=True)
            if not low_cost.empty:
                st.dataframe(
                    low_cost[["department", "so_nhan_vien", "tong_du_bao", "dinh_muc_tb"]].rename(columns={
                        "department": tr("Phòng Ban", "Department"), "so_nhan_vien": tr("Số NV", "Headcount"),
                        "tong_du_bao": tr("Tổng Chi Phí (đ)", "Total Cost (VND)"), "dinh_muc_tb": tr("Định Mức TB/Người (đ)", "Avg Quota/Staff (VND)")
                    }).style.format({
                        tr("Tổng Chi Phí (đ)", "Total Cost (VND)"): "{:,.0f} đ",
                        tr("Định Mức TB/Người (đ)", "Avg Quota/Staff (VND)"): "{:,.0f} đ"
                    }),
                    hide_index=True,
                    use_container_width=True
                )
    else:
        st.warning(tr("⚠️ Chưa tìm thấy dữ liệu dự báo phòng ban.", "⚠️ Department forecast data not found."))

# ------------------------------------------------------------------------------
# TAB 3: TRA CỨU CHI TIẾT NHÂN VIÊN (EMPLOYEE LOOKUP)
# ------------------------------------------------------------------------------
with tab3:
    st.markdown(f"### 👤 {tr('Tra Cứu Dự Báo Cấp Cá Nhân & Quản Trị Định Mức Suất Ăn', 'Individual Employee Forecast & Meal Quota Lookup')}")
    st.caption(tr(
        f"Kỳ dự báo Tháng {forecast_month} — Bao phủ {len(data['emp']):,} nhân sự trong file dự báo",
        f"Forecast Period: Month {forecast_month} — {len(data['emp']):,} employees in the forecast file"
    ))

    if not data["emp"].empty:
        df_emp_all = data["emp"].copy()
        df_emp_all["tong_suat"] = df_emp_all["pred_N44"] + df_emp_all["pred_N70"]
        df_emp_all["tien_n44"] = df_emp_all["pred_N44"] * PRICE_44
        df_emp_all["tien_n70"] = df_emp_all["pred_N70"] * PRICE_70

        bins = [-1, 0, 300000, 800000, float("inf")]
        seg_labels = [
            tr("Không ăn (0 đ)", "Zero meal (0 VND)"),
            tr("Ăn ít (<300k)", "Low (<300k VND)"),
            tr("Ăn vừa (300k - 800k)", "Moderate (300k - 800k VND)"),
            tr("Ăn nhiều (>=800k)", "Heavy (>=800k VND)"),
        ]
        df_emp_all["phan_khuc"] = pd.cut(df_emp_all["pred_cost"], bins=bins, labels=seg_labels)

        df_emp_all["khau_vi"] = np.where(
            df_emp_all["tong_suat"] == 0, tr("Không ăn", "No meal"),
            np.where(df_emp_all["pred_N70"] > 0, tr("Có ăn món Âu (70k)", "Eats Western (70k)"), tr("Chỉ ăn Việt/Chay (44k)", "Only VN/Veg (44k)"))
        )

        st.markdown(f"#### 🔍 {tr('Soi Chi Tiết Hồ Sơ Cá Nhân (Employee Spotlight)', 'Employee Spotlight Profile')}")
        col_search_id, col_search_dept = st.columns([2, 3])

        with col_search_id:
            all_ids = sorted(df_emp_all["employee_id"].unique())
            default_index = 0
            if search_emp.strip() != "":
                matched_ids = [i for i in all_ids if str(i).startswith(search_emp.strip())]
                if matched_ids:
                    default_index = all_ids.index(matched_ids[0])

            selected_emp_id = st.selectbox(
                tr("Chọn hoặc gõ Mã Nhân Viên (ID):", "Select or Search Employee ID:"),
                options=all_ids,
                index=default_index,
            )

        emp_record = df_emp_all[df_emp_all["employee_id"] == selected_emp_id].iloc[0]
        emp_n44 = int(emp_record["pred_N44"])
        emp_n70 = int(emp_record["pred_N70"])
        emp_cost = int(emp_record["pred_cost"])
        emp_total_meals = int(emp_record["tong_suat"])
        emp_dept = str(emp_record["department"])
        emp_source = str(emp_record["source"])
        meals_per_workday_est = emp_total_meals / 22 if 22 > 0 else 0

        with col_search_dept:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div>
                        <span style="font-size: 1.25rem; font-weight: 800; color: #38bdf8;">{tr('NV', 'Emp')} #{selected_emp_id}</span>
                        <span style="font-size: 0.9rem; color: #94a3b8; margin-left: 8px;">| {emp_dept}</span>
                    </div>
                    <span style="background-color: {'#10b981' if emp_source == 'CatBoost' else '#f59e0b'}; color: white; font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 20px;">
                        {'AI CatBoost' if emp_source == 'CatBoost' else tr('Group-Mean (NV Mới)', 'Group-Mean (New Staff)')}
                    </span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 12px; border-top: 1px solid #334155; padding-top: 12px;">
                    <div>
                        <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase;">{tr('Dự Báo Chi Phí', 'Projected Cost')}</div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #f8fafc;">{emp_cost:,.0f} đ</div>
                    </div>
                    <div>
                        <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase;">{tr('Tổng Suất Ăn', 'Total Meals')}</div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #38bdf8;">{emp_total_meals} {tr('bữa', 'meals')}</div>
                    </div>
                    <div>
                        <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase;">{tr('Việt (44k) / Âu (70k)', 'VN (44k) / West (70k)')}</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #e2e8f0;">{emp_n44} V / {emp_n70} Â</div>
                    </div>
                    <div>
                        <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase;">{tr('Suất/ngày (ước tính)', 'Meals/day (estimated)')}</div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: '#34d399';">{meals_per_workday_est:.1f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown(f"#### 🎯 {tr('Bộ Lọc Danh Sách Nhân Sự Đa Chiều', 'Multi-Dimensional Employee Filter')}")
        flt_col1, flt_col2, flt_col3, flt_col4 = st.columns(4)

        with flt_col1:
            dept_opts = [all_dept_label] + sorted(df_emp_all["department"].dropna().unique().tolist())
            dept_init_idx = dept_opts.index(selected_dept) if selected_dept in dept_opts else 0
            filter_dept = st.selectbox(tr("Phòng Ban:", "Department:"), dept_opts, index=dept_init_idx, key="t3_dept")

        with flt_col2:
            filter_seg = st.selectbox(
                tr("Phân Khúc Chi Tiêu:", "Expense Segment:"),
                [tr("Tất cả phân khúc", "All Segments")] + seg_labels
            )

        with flt_col3:
            filter_taste = st.selectbox(
                tr("Khẩu Vị Suất Ăn:", "Meal Preference:"),
                [tr("Tất cả khẩu vị", "All Preferences"), tr("Chỉ ăn Việt/Chay (44k)", "Only VN/Veg (44k)"), tr("Có ăn món Âu (70k)", "Eats Western (70k)"), tr("Không ăn", "No meal")]
            )

        with flt_col4:
            filter_source = st.selectbox(
                tr("Nguồn Dự Báo:", "Forecast Method:"),
                [tr("Tất cả nguồn", "All Methods"), tr("Mô hình CatBoost AI", "CatBoost AI Model"), tr("Group-Mean (Cold-start)", "Group-Mean (Cold-start)")]
            )

        filtered_df = df_emp_all.copy()
        if filter_dept != all_dept_label:
            filtered_df = filtered_df[filtered_df["department"] == filter_dept]
        if filter_seg != tr("Tất cả phân khúc", "All Segments"):
            filtered_df = filtered_df[filtered_df["phan_khuc"] == filter_seg]
        if filter_taste != tr("Tất cả khẩu vị", "All Preferences"):
            filtered_df = filtered_df[filtered_df["khau_vi"] == filter_taste]
        if filter_source == tr("Mô hình CatBoost AI", "CatBoost AI Model"):
            filtered_df = filtered_df[filtered_df["source"] == "CatBoost"]
        elif filter_source == tr("Group-Mean (Cold-start)", "Group-Mean (Cold-start)"):
            filtered_df = filtered_df[filtered_df["source"] != "CatBoost"]

        sub_total_staff = len(filtered_df)
        sub_total_cost = filtered_df["pred_cost"].sum()
        sub_avg_cost = (sub_total_cost / sub_total_staff) if sub_total_staff > 0 else 0
        sub_au_count = (filtered_df["pred_N70"] > 0).sum()
        sub_au_pct = (sub_au_count / sub_total_staff * 100) if sub_total_staff > 0 else 0

        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
        with kpi_c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{tr('Số Nhân Sự Khớp Lọc', 'Filtered Headcount')}</div>
                <div class="metric-value">{sub_total_staff:,} {tr('người', 'staff')}</div>
                <div class="metric-subtitle">{tr('Chiếm', 'Represents')} {(sub_total_staff/len(df_emp_all)*100) if len(df_emp_all) else 0:.1f}% {tr('toàn công ty', 'of company')}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{tr('Tổng Chi Phí Nhóm', 'Filtered Subtotal')}</div>
                <div class="metric-value">{sub_total_cost:,.0f} đ</div>
                <div class="metric-subtitle">{tr('Chiếm', 'Represents')} {(sub_total_cost/total_budget*100) if total_budget else 0:.1f}% {tr('ngân sách', 'of budget')}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{tr('Bình Quân / Nhân Sự', 'Average / Staff')}</div>
                <div class="metric-value" style="color: #059669;">{sub_avg_cost:,.0f} đ</div>
                <div class="metric-subtitle">~{sub_avg_cost/22:,.0f} {tr('đ/ngày công', 'VND/workday')}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{tr('Tỷ Lệ Ăn Món Âu', 'Western Meal Share')}</div>
                <div class="metric-value" style="color: #d97706;">{sub_au_pct:.1f}%</div>
                <div class="metric-subtitle">{sub_au_count:,} {tr('người chọn món Âu', 'staff eat Western')}</div>
            </div>
            """, unsafe_allow_html=True)

        # Biểu đồ phân bổ
        col_fig_hist, col_fig_scatter = st.columns([1, 1])

        with col_fig_hist:
            st.subheader(f"📊 {tr('Phân Phối Mức Chi Phí Nhân Sự', 'Employee Cost Distribution')}")
            if not filtered_df.empty:
                fig_hist = px.histogram(
                    filtered_df,
                    x="pred_cost",
                    nbins=25,
                    title=tr("Phổ Chi Tiêu Suất Ăn Của Nhân Viên", "Employee Meal Spending Histogram"),
                    labels={"pred_cost": tr("Chi Phí Dự Báo (VNĐ)", "Projected Cost (VND)"), "count": tr("Số Lượng Nhân Sự", "Headcount")},
                    color_discrete_sequence=["#3b82f6"]
                )
                fig_hist.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), yaxis_title=tr("Số Nhân Sự", "Headcount"))
                st.plotly_chart(fig_hist, use_container_width=True)

        with col_fig_scatter:
            st.subheader(f"🎯 {tr('Tương Quan: Suất Việt (44k) vs Suất Âu (70k)', 'Correlation: VN/Veg (44k) vs Western (70k)')}")
            if not filtered_df.empty:
                fig_scatter = px.scatter(
                    filtered_df,
                    x="pred_N44",
                    y="pred_N70",
                    color="khau_vi",
                    hover_name="employee_id",
                    hover_data=["department", "pred_cost"],
                    title=tr("Cơ Cấu Suất Ăn Của Từng Nhân Sự", "Individual Meal Breakdown"),
                    labels={"pred_N44": tr("Số Suất Việt / Chay", "VN / Veg Meals"), "pred_N70": tr("Số Suất Âu", "Western Meals"), "khau_vi": tr("Khẩu Vị", "Preference")},
                    color_discrete_map={
                        tr("Chỉ ăn Việt/Chay (44k)", "Only VN/Veg (44k)"): "#3b82f6",
                        tr("Có ăn món Âu (70k)", "Eats Western (70k)"): "#f59e0b",
                        tr("Không ăn", "No meal"): "#94a3b8"
                    }
                )
                fig_scatter.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")

        # Bảng dữ liệu nhân viên
        st.subheader(f"📋 {tr(f'Bảng Danh Sách Nhân Viên ({sub_total_staff:,} nhân sự)', f'Employee Directory ({sub_total_staff:,} staff)')}")

        table_emp_display = filtered_df[[
            "employee_id", "department", "pred_N44", "pred_N70",
            "tong_suat", "pred_cost", "phan_khuc", "source"
        ]].copy()
        table_emp_display.columns = [
            tr("Mã NV", "Emp ID"),
            tr("Phòng Ban / Bộ Phận", "Department"),
            tr("Suất Việt (44k)", "VN Meals (44k)"),
            tr("Suất Âu (70k)", "Western Meals (70k)"),
            tr("Tổng Suất", "Total Meals"),
            tr("Chi Phí Dự Báo (VNĐ)", "Projected Cost (VND)"),
            tr("Phân Khúc Tiêu Dùng", "Consumption Segment"),
            tr("Nguồn Mô Hình", "Model Source")
        ]

        st.dataframe(
            table_emp_display.style.format({
                table_emp_display.columns[2]: "{:,}",
                table_emp_display.columns[3]: "{:,}",
                table_emp_display.columns[4]: "{:,}",
                table_emp_display.columns[5]: "{:,.0f} đ",
            }).highlight_max(subset=[table_emp_display.columns[5]], color="#fee2e2"),
            use_container_width=True,
            height=350,
        )

        csv_emp_export = table_emp_display.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label=f"📥 {tr(f'Tải Báo Cáo Định Mức {sub_total_staff:,} Nhân Viên (CSV)', f'Download {sub_total_staff:,} Employee Quotas (CSV)')}",
            data=csv_emp_export,
            file_name=f"Employee_Meal_Quotas_{forecast_month}.csv",
            mime="text/csv",
        )

        st.markdown("---")

        # 3 nhóm ngoại lệ
        st.subheader(f"⚠️ {tr('3 Nhóm Nhân Sự Cần Lưu Ý', '3 Groups of Personnel to Pay Attention To')}")
        audit_tab1, audit_tab2, audit_tab3 = st.tabs([
            tr("🚫 Nhân sự dự báo 0 suất", "🚫 Staff forecast at zero meals"),
            tr("🥩 Nhân sự có dự báo suất Âu", "🥩 Staff forecast with Western meals"),
            tr("🆕 Nhân sự dùng Group-Mean", "🆕 Staff using Group-Mean")
        ])

        with audit_tab1:
            zero_df = df_emp_all[df_emp_all["pred_cost"] == 0].copy()
            st.markdown(tr(
                f"- **Ý nghĩa quản trị:** Mô hình dự báo **{len(zero_df)} nhân sự ({(len(zero_df)/len(df_emp_all)*100) if len(df_emp_all) else 0:.1f}%)** có chi phí suất ăn bằng **0 đ** trong kỳ dự báo.<br>"
                f"- **Hành động cho HR:** Xác minh với HR trước khi điều chỉnh đơn đặt suất ăn; dự báo 0 không đồng nghĩa đã xác nhận nhân viên nghỉ hoặc không ăn.",
                f"- **Management Value:** **{len(zero_df)} employees ({(len(zero_df)/len(df_emp_all)*100) if len(df_emp_all) else 0:.1f}%)** are forecast at **0 VND**.<br>"
                f"- **HR Action:** Verify with HR before changing meal orders; a zero forecast is not proof of absence or non-consumption."
            ), unsafe_allow_html=True)
            st.dataframe(
                zero_df[["employee_id", "department", "pred_cost", "source"]].rename(columns={
                    "employee_id": tr("Mã NV", "Emp ID"), "department": tr("Phòng Ban", "Department"),
                    "pred_cost": tr("Chi Phí (đ)", "Cost (VND)"), "source": tr("Nguồn Dự Báo", "Source")
                }),
                hide_index=True,
                use_container_width=True
            )

        with audit_tab2:
            au_df = df_emp_all[df_emp_all["pred_N70"] > 0].sort_values("pred_N70", ascending=False).copy()
            st.markdown(tr(
                f"- **Ý nghĩa quản trị:** Có **{len(au_df)} nhân sự ({(len(au_df)/len(df_emp_all)*100) if len(df_emp_all) else 0:.1f}%)** được dự báo có ít nhất một suất Âu (70.000 đ/suất).<br>"
                f"- **Hành động:** Dùng con số này làm dự báo nhu cầu, không phải lịch sử tiêu dùng đã xác nhận.",
                f"- **Management Value:** **{len(au_df)} staff ({(len(au_df)/len(df_emp_all)*100) if len(df_emp_all) else 0:.1f}%)** are forecast with at least one Western meal.<br>"
                f"- **Action:** Treat this as forecast demand, not verified historical behavior."
            ), unsafe_allow_html=True)
            st.dataframe(
                au_df.head(15)[["employee_id", "department", "pred_N44", "pred_N70", "pred_cost", "tien_n70"]].rename(columns={
                    "employee_id": tr("Mã NV", "Emp ID"), "department": tr("Phòng Ban", "Department"),
                    "pred_N44": tr("Suất Việt", "VN Meals"), "pred_N70": tr("Suất Âu", "West Meals"),
                    "pred_cost": tr("Tổng Tiền (đ)", "Total (VND)"), "tien_n70": tr("Tiền Món Âu (đ)", "West Cost (VND)")
                }).style.format({
                    tr("Tổng Tiền (đ)", "Total (VND)"): "{:,.0f} đ",
                    tr("Tiền Món Âu (đ)", "West Cost (VND)"): "{:,.0f} đ"
                }),
                hide_index=True,
                use_container_width=True
            )

        with audit_tab3:
            cold_df = df_emp_all[df_emp_all["source"].astype(str).str.contains("Group-Mean", case=False, na=False)].copy()
            st.markdown(tr(
                f"- **Ý nghĩa quản trị:** **{len(cold_df)} nhân sự ({(len(cold_df)/len(df_emp_all)*100) if len(df_emp_all) else 0:.1f}%)** có nguồn dự báo Group-Mean (cold-start).<br>"
                f"- **Lưu ý:** Nhãn nguồn dự báo không tự chứng minh ngày tuyển dụng hoặc số tháng lịch sử của từng người.",
                f"- **Management Value:** **{len(cold_df)} staff ({(len(cold_df)/len(df_emp_all)*100) if len(df_emp_all) else 0:.1f}%)** use Group-Mean (cold-start).<br>"
                f"- **Note:** The forecast source alone does not prove each person's hire date or history length."
            ), unsafe_allow_html=True)
            st.dataframe(
                cold_df.head(15)[["employee_id", "department", "pred_N44", "pred_N70", "pred_cost", "source"]].rename(columns={
                    "employee_id": tr("Mã NV", "Emp ID"), "department": tr("Phòng Ban", "Department"),
                    "pred_N44": tr("Suất Việt", "VN Meals"), "pred_N70": tr("Suất Âu", "West Meals"),
                    "pred_cost": tr("Tổng Tiền (đ)", "Total (VND)"), "source": tr("Phương Pháp", "Method")
                }).style.format({tr("Tổng Tiền (đ)", "Total (VND)"): "{:,.0f} đ"}),
                hide_index=True,
                use_container_width=True
            )


# ------------------------------------------------------------------------------
# TAB 4: ĐỐI SOÁT & KỸ THUẬT MÔ HÌNH (MODEL BENCHMARK)
# ------------------------------------------------------------------------------
with tab4:
    st.markdown(f"### 🔬 {tr('Báo Cáo Kỹ Thuật: Đối Soát Hiệu Năng & Kiểm Định Mô Hình', 'Technical Report: Model Benchmarking & Governance')}")
    st.caption(tr(
        "Benchmark nhân viên-tháng dùng Walk-Forward trên 3 tháng hoàn chỉnh gần nhất.",
        "Employee-month benchmark uses walk-forward validation on the latest 3 complete months."
    ))

    # Tính các chỉ số mô hình từ file metrics hiện có, không viết cứng số liệu.
    _ml_metrics = data.get("ml", pd.DataFrame())
    _dl_metrics = data.get("dl", pd.DataFrame())
    _ens_metrics = data.get("ens", pd.DataFrame())
    _cat_row = _ml_metrics[_ml_metrics.get("Model", pd.Series(dtype=str)).astype(str).str.contains("CatBoost", case=False, na=False)] if not _ml_metrics.empty else pd.DataFrame()
    _gru_row = _dl_metrics[_dl_metrics.get("Model", pd.Series(dtype=str)).astype(str).str.contains("GRU", case=False, na=False)] if not _dl_metrics.empty else pd.DataFrame()
    _s2_row = _ens_metrics[_ens_metrics.get("Method", pd.Series(dtype=str)).astype(str).str.contains("S2", case=False, na=False)] if not _ens_metrics.empty else pd.DataFrame()
    cat_mae = float(_cat_row["MAE_Cost"].iloc[0]) if not _cat_row.empty else np.nan
    gru_mae = float(_gru_row["MAE_Cost"].iloc[0]) if not _gru_row.empty else np.nan
    s2_mae = float(_s2_row["MAE_Cost"].iloc[0]) if not _s2_row.empty else np.nan
    cat_mae_label = f"{cat_mae:,.0f} đ" if pd.notna(cat_mae) else "N/A"
    gru_mae_label = f"{gru_mae:,.0f} đ" if pd.notna(gru_mae) else "N/A"
    s2_mae_label = f"{s2_mae:,.0f} đ" if pd.notna(s2_mae) else "N/A"
    defense_p1 = tr(
        f"• <b>1. Trong nhóm Machine Learning:</b> CatBoost M3 có MAE <b>{cat_mae_label}/người</b> theo file benchmark hiện tại.<br>"
        "• <b>2. So sánh công bằng:</b> Đối chiếu MAE trên cùng tập kiểm định và cùng kỳ trước khi kết luận mô hình tốt nhất.",
        f"• <b>1. Within Machine Learning:</b> CatBoost M3 has MAE <b>{cat_mae_label}/staff</b> in the current benchmark file.<br>"
        "• <b>2. Fair comparison:</b> Compare metrics on the same validation set and period before declaring a winner."
    )
    defense_p2 = tr(
        f"• <b>3. Deep Learning:</b> GRU MAE hiện tại là <b>{gru_mae_label}</b> theo file metrics.<br>"
        f"• <b>4. Ensemble S2:</b> MAE hiện tại là <b>{s2_mae_label}</b>. Baseline có thể tốt hơn; xem bảng xếp hạng bên dưới trước khi chọn production.",
        f"• <b>3. Deep Learning:</b> Current GRU MAE is <b>{gru_mae_label}</b> in the metrics file.<br>"
        f"• <b>4. Ensemble S2:</b> Current MAE is <b>{s2_mae_label}</b>. A baseline may perform better; consult the leaderboard before choosing production."
    )

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; padding: 22px; border-radius: 14px; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="background-color: #10b981; color: white; font-size: 11px; font-weight: 700; padding: 4px 12px; border-radius: 20px; text-transform: uppercase;">
                {tr('LỰA CHỌN PRODUCTION CHÍNH THỨC: CATBOOST (M3)', 'OFFICIAL PRODUCTION MODEL: CATBOOST (M3)')}
            </span>
            <span style="font-size: 13px; color: #94a3b8;">{tr('Tiêu chuẩn kiểm định: <b>Walk-Forward Backtest</b>', 'Validation Standard: <b>Walk-Forward Backtest</b>')}</span>
        </div>
        <h3 style="color: white; margin: 4px 0 10px 0; font-size: 19px; font-weight: 700;">
            {tr('Tại Sao Chọn CatBoost M3 Thay Vì Deep Learning Hay Số Bình Quân?', 'Why CatBoost M3 Over Deep Learning or Simple Baselines?')}
        </h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; font-size: 13.5px; color: #cbd5e1; line-height: 1.55;">
            <div>{defense_p1}</div>
            <div>{defense_p2}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    same_split_metrics = data.get("same_split_comparison", pd.DataFrame())
    if not same_split_metrics.empty:
        st.markdown(
            f"#### {tr('So sánh công bằng: cùng dữ liệu, cùng kỳ test', 'Fair comparison: same data and test months')}"
        )
        st.caption(tr(
            "Tất cả mô hình được đối chiếu theo tổng ngân sách công ty trong cùng 3 tháng test; dự báo ML/DL theo nhân viên được cộng lên tổng tháng. WAPE là tổng sai số tuyệt đối chia tổng chi phí thực tế.",
            "All models are compared on company-wide monthly budget over the same 3 test months; employee-level ML/DL predictions are summed to monthly totals. WAPE is total absolute error divided by total actual cost."
        ))
        st.dataframe(
            same_split_metrics,
            hide_index=True,
            use_container_width=True,
        )

    time_series_metrics = data.get("time_series", pd.DataFrame())
    if not time_series_metrics.empty:
        st.markdown(
            f"#### {tr('Benchmark chuỗi thời gian tổng suất theo ngày', 'Daily company-wide time-series benchmark')}"
        )
        st.caption(tr(
            "ARIMA, MLP và Facebook Prophet được kiểm định riêng trên tổng suất toàn công ty theo ngày. MAE/RMSE Cost ở đây là sai số chi phí toàn công ty/ngày; không so trực tiếp với metric nhân viên-tháng phía dưới.",
            "ARIMA, MLP, and Facebook Prophet are evaluated separately on daily company-wide meal totals. Cost MAE/RMSE here are company-wide daily errors; do not compare directly with the employee-month metrics below."
        ))
        display_columns = [
            column for column in (
                "Model", "MAE_Cost", "RMSE_Cost", "WAPE_Cost_%",
                "MAE_N44", "MAE_N70", "MSE_N44", "MSE_N70",
                "MAPE_N44_positive_%", "MAPE_N70_positive_%",
            )
            if column in time_series_metrics.columns
        ]
        st.dataframe(
            time_series_metrics[display_columns],
            hide_index=True,
            use_container_width=True,
        )

    # Master Leaderboard
    df_base_raw = data.get("base", pd.DataFrame()).copy()
    df_ml_raw = data.get("ml", pd.DataFrame()).copy()
    df_dl_raw = data.get("dl", pd.DataFrame()).copy()
    df_ens_raw = data.get("ens", pd.DataFrame()).copy()

    list_to_concat = []
    if not df_base_raw.empty:
        b_tmp = df_base_raw.copy()
        b_tmp["Phan_Loai"] = tr("Baseline (Mô Hình Cơ Sở)", "Baseline Models")
        b_tmp["Ten_Mo_Hinh"] = b_tmp["Model"]
        b_tmp["Do_Phuc_Tap"] = tr("Thấp (Quy tắc đơn giản)", "Low (Simple Rules)")
        list_to_concat.append(b_tmp)

    if not df_ml_raw.empty:
        m_tmp = df_ml_raw.copy()
        m_tmp["Phan_Loai"] = "Machine Learning"
        m_tmp["Ten_Mo_Hinh"] = m_tmp["Model"]
        m_tmp["Do_Phuc_Tap"] = tr("Trung bình (GBDT tối ưu)", "Medium (Optimized GBDT)")
        list_to_concat.append(m_tmp)

    if not df_dl_raw.empty:
        dl_tmp = df_dl_raw.copy()
        dl_tmp["Phan_Loai"] = tr("Deep Learning (Mạng Nơ-ron)", "Deep Learning (Neural Nets)")
        dl_tmp["Ten_Mo_Hinh"] = dl_tmp["Model"]
        dl_tmp["Do_Phuc_Tap"] = tr("Cao (PyTorch / GPU)", "High (PyTorch / GPU)")
        list_to_concat.append(dl_tmp)

    if not df_ens_raw.empty:
        ens_tmp = df_ens_raw.copy()
        model_col = "Method" if "Method" in ens_tmp.columns else "Model"
        ens_tmp["Ten_Mo_Hinh"] = ens_tmp[model_col]
        if model_col != "Model":
            ens_tmp = ens_tmp.rename(columns={model_col: "Model"})
        ens_tmp["Phan_Loai"] = tr("Ensemble (Tổ Hợp Mô Hình)", "Ensemble Models")
        ens_tmp["Do_Phuc_Tap"] = tr("Rất cao (Đa thuật toán)", "Very High (Multi-model)")
        list_to_concat.append(ens_tmp)

    if list_to_concat:
        master_leaderboard = (
            pd.concat(list_to_concat, ignore_index=True)
            .sort_values("MAE_Cost")
            .reset_index(drop=True)
        )
    else:
        # Giữ schema rỗng để giao diện không lỗi nếu thiếu file metrics.
        master_leaderboard = pd.DataFrame(columns=[
            "Model", "MAE_Cost", "RMSE_Cost", "WAPE_Cost_%", "MAE_N44", "MAE_N70",
            "Phan_Loai", "Ten_Mo_Hinh", "Do_Phuc_Tap"
        ])

    st.markdown(f"#### 🏆 {tr('Bảng Xếp Hạng Hiệu Năng Toàn Diện (Master Model Leaderboard)', 'Master Model Benchmark Leaderboard')}")

    col_filter_f, col_sort_f = st.columns([2, 2])
    with col_filter_f:
        filter_family = st.selectbox(
            tr("Lọc nhóm thuật toán:", "Filter Model Family:"),
            options=[
                tr(f"Tất cả {len(master_leaderboard)} mô hình", f"All {len(master_leaderboard)} models"),
                "Machine Learning",
                tr("Baseline (Mô Hình Cơ Sở)", "Baseline Models"),
                tr("Deep Learning (Mạng Nơ-ron)", "Deep Learning (Neural Nets)"),
                tr("Ensemble (Tổ Hợp Mô Hình)", "Ensemble Models")
            ]
        )
    with col_sort_f:
        sort_by_metric = st.selectbox(
            tr("Xếp hạng ưu tiên theo:", "Sort Priority:"),
            options=[
                tr("Sai số tiền MAE Cost (VNĐ)", "Cost MAE (VND)"),
                tr("Tỷ lệ sai số WAPE (%)", "WAPE Error Rate (%)"),
                tr("RMSE Cost (VNĐ)", "RMSE Cost (VND)"),
                tr("Sai số Suất Âu MAE N70", "Western Meal MAE N70")
            ]
        )

    display_board = master_leaderboard.copy()
    if filter_family != tr(f"Tất cả {len(master_leaderboard)} mô hình", f"All {len(master_leaderboard)} models"):
        display_board = display_board[display_board["Phan_Loai"] == filter_family]

    sort_metric_map = {
        tr("Sai số tiền MAE Cost (VNĐ)", "Cost MAE (VND)"): "MAE_Cost",
        tr("Tỷ lệ sai số WAPE (%)", "WAPE Error Rate (%)"): "WAPE_Cost_%",
        tr("RMSE Cost (VNĐ)", "RMSE Cost (VND)"): "RMSE_Cost",
        tr("Sai số Suất Âu MAE N70", "Western Meal MAE N70"): "MAE_N70"
    }
    display_board = display_board.sort_values(sort_metric_map[sort_by_metric]).reset_index(drop=True)
    display_board["Xep_Hang"] = display_board.index + 1

    board_view = display_board[[
        "Xep_Hang", "Ten_Mo_Hinh", "Phan_Loai", "MAE_Cost", "RMSE_Cost",
        "WAPE_Cost_%", "MAE_N44", "MAE_N70", "Do_Phuc_Tap"
    ]].copy()
    board_view.columns = [
        tr("Hạng", "Rank"),
        tr("Tên Thuật Toán / Mô Hình", "Algorithm / Model"),
        tr("Phân Loại", "Family"),
        tr("Sai Số MAE (đ)", "Cost MAE (VND)"),
        tr("RMSE (đ)", "RMSE (VND)"),
        tr("Tỷ Lệ WAPE (%)", "WAPE (%)"),
        tr("MAE Cơm Việt (suất)", "VN Meal MAE"),
        tr("MAE Món Âu (suất)", "Western Meal MAE"),
        tr("Độ Phức Tạp Vận Hành", "Operational Complexity")
    ]

    board_view[tr("Trạng Thái", "Status")] = np.where(
        board_view[tr("Tên Thuật Toán / Mô Hình", "Algorithm / Model")] == "M3_CatBoost",
        tr("⭐ ĐƯỢC CHỌN VẬN HÀNH", "⭐ SELECTED FOR PRODUCTION"),
        tr("Tham chiếu benchmark", "Benchmark Reference")
    )

    st.dataframe(
        board_view.style.format({
            board_view.columns[3]: "{:,.0f} đ",
            board_view.columns[4]: "{:,.0f} đ",
            board_view.columns[5]: "{:.2f}%",
            board_view.columns[6]: "{:.2f}",
            board_view.columns[7]: "{:.2f}",
        }).highlight_min(subset=[board_view.columns[3]], color="#dcfce7"),
        hide_index=True,
        use_container_width=True,
        height=380
    )

    st.markdown("---")

    col_chart_mae, col_chart_meal = st.columns([3, 2])

    with col_chart_mae:
        st.subheader(f"1. {tr('So Sánh Sai Số Chi Phí MAE (VNĐ/người/tháng)', 'Cost MAE Comparison (VND/staff/month)')}")
        if not master_leaderboard.empty:
            fig_mae = px.bar(
                master_leaderboard,
                x="Ten_Mo_Hinh",
                y="MAE_Cost",
                color="Phan_Loai",
                text="MAE_Cost",
                color_discrete_map={
                    "Machine Learning": "#2563eb",
                    tr("Baseline (Mô Hình Cơ Sở)", "Baseline Models"): "#94a3b8",
                    tr("Deep Learning (Mạng Nơ-ron)", "Deep Learning (Neural Nets)"): "#ef4444",
                    tr("Ensemble (Tổ Hợp Mô Hình)", "Ensemble Models"): "#8b5cf6"
                },
                labels={"MAE_Cost": tr("Sai Số MAE (VNĐ)", "MAE Cost (VND)"), "Ten_Mo_Hinh": tr("Mô Hình", "Model"), "Phan_Loai": tr("Họ Thuật Toán", "Family")}
            )
            fig_mae.update_traces(texttemplate="%{text:,.0f} đ", textposition="outside")
            fig_mae.update_layout(
                yaxis=dict(title=tr("MAE Cost (VNĐ)", "MAE Cost (VND)"), range=[0, master_leaderboard["MAE_Cost"].max() * 1.18]),
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_mae, use_container_width=True)

    with col_chart_meal:
        st.subheader(f"2. {tr('Sai Số Dự Báo Suất Ăn: Việt vs Âu', 'Meal Error Comparison: VN vs Western')}")
        if not master_leaderboard.empty:
            df_meal_err = master_leaderboard[master_leaderboard["Phan_Loai"].isin(["Machine Learning", tr("Baseline (Mô Hình Cơ Sở)", "Baseline Models")])].copy()
            fig_meal = px.bar(
                df_meal_err,
                x="Ten_Mo_Hinh",
                y=["MAE_N44", "MAE_N70"],
                barmode="group",
                title=tr("Độ Lệch Suất Ăn (Số bữa/tháng)", "Meal Count Deviation (Meals/month)"),
                labels={"value": tr("Số bữa lệch", "Deviation"), "Ten_Mo_Hinh": tr("Mô hình", "Model"), "variable": tr("Loại Suất", "Category")},
                color_discrete_map={"MAE_N44": "#3b82f6", "MAE_N70": "#f59e0b"}
            )
            fig_meal.update_layout(height=380, legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_meal, use_container_width=True)

    st.markdown("---")

    # Phân khúc kiểm định
    st.subheader(f"🎯 {tr('Đánh Giá Hiệu Năng Theo 4 Phân Khúc Hành Vi (Segment Evaluation)', 'Segment Evaluation Across 4 Consumption Groups')}")
    if not data["segment"].empty:
        df_seg_view = data["segment"].copy().reset_index(drop=True)
        col_seg_left, col_seg_right = st.columns([3, 2])

        with col_seg_left:
            df_seg_disp = df_seg_view[[
                "segment", "So_Luong", "Tong_Thuc", "Tong_Doan", "MAE_Cost", "RMSE_Cost", "WAPE_%"
            ]].copy()
            df_seg_disp.columns = [
                tr("Phân Khúc Tiêu Dùng", "Consumption Segment"),
                tr("Số Lượng Lượt", "Observations"),
                tr("Tổng Thực Tế (đ)", "Total Actual (VND)"),
                tr("Tổng Dự Báo (đ)", "Total Forecast (VND)"),
                tr("Sai Số MAE (đ)", "MAE (VND)"),
                tr("RMSE (đ)", "RMSE (VND)"),
                tr("Tỷ Lệ Lệch WAPE (%)", "WAPE Error (%)")
            ]
            st.dataframe(
                df_seg_disp.style.format({
                    df_seg_disp.columns[1]: "{:,}",
                    df_seg_disp.columns[2]: "{:,.0f} đ",
                    df_seg_disp.columns[3]: "{:,.0f} đ",
                    df_seg_disp.columns[4]: "{:,.0f} đ",
                    df_seg_disp.columns[5]: "{:,.0f} đ",
                    df_seg_disp.columns[6]: lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A (actual total = 0)",
                }),
                hide_index=True,
                use_container_width=True
            )

        with col_seg_right:
            st.markdown(f"""
            <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; padding: 16px; border-radius: 10px; font-size: 13.5px; line-height: 1.6; color: #166534;">
                <b>💡 {tr('Kết luận quan trọng cho Tài chính:', 'Key Financial Conclusion:')}</b><br>
                {tr(
                    "Các kết quả trong bảng được lấy trực tiếp từ segment_eval.csv. WAPE không xác định khi tổng thực tế bằng 0; hãy xem riêng tổng dự báo và MAE của phân khúc đó.",
                    "Values in the table are read from segment_eval.csv. WAPE is undefined when actual total is zero; review forecast total and MAE separately for that segment."
                )}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ==========================================================================
    # 6. MỔ XẺ NGUYÊN NHÂN 10 CA NGOẠI LỆ LỚN NHẤT (ĐÃ CHUẨN HÓA PHÒNG BAN)
    # ==========================================================================
    st.subheader(f"⚠️ {tr('Mổ Xẻ 10 Trường Hợp Ngoại Lệ Sai Lệch Cao Nhất (Root-Cause Outliers)', 'Root-Cause Diagnosis of Top 10 Prediction Outliers')}")
    st.caption(tr(
        "Phân tích nguyên nhân gốc rễ để chứng minh tính hợp lý của mô hình trước các câu hỏi chất vấn của Lãnh đạo",
        "Root-cause diagnosis to justify model predictions against executive scrutiny"
    ))

    if not data["outliers"].empty:
        df_out_10 = data["outliers"].head(10).copy().reset_index(drop=True)

        # ----------------------------------------------------------------------
        # 👉 ÁNH XẠ PHÒNG BAN CHUẨN CHO BẢNG OUTLIERS (3 LỚP AN TOÀN)
        # ----------------------------------------------------------------------
        # Lớp 1: Khớp trực tiếp từ bảng nhân sự đã map chuẩn trong data["emp"]
        if not data["emp"].empty and "employee_id" in data["emp"].columns and "department" in data["emp"].columns:
            emp_clean = data["emp"].copy()
            emp_clean["_id"] = pd.to_numeric(emp_clean["employee_id"].astype(str).str.extract(r'(\d+)')[0], errors="coerce")
            df_out_10["_id"] = pd.to_numeric(df_out_10["employee_id"].astype(str).str.extract(r'(\d+)')[0], errors="coerce")

            emp_map_rows = emp_clean.dropna(subset=["_id", "department"]).drop_duplicates(subset=["_id"], keep="last")
            emp_dept_map = dict(zip(emp_map_rows["_id"], emp_map_rows["department"]))
            df_out_10["phong_ban_chuan"] = df_out_10["_id"].map(emp_dept_map)

            # Lớp 2: Kế thừa từ đồng nghiệp cùng Team (nếu nhân viên đã nghỉ trước tháng 9)
            if "team_con" in emp_clean.columns:
                team_to_parent = (
                    emp_clean.dropna(subset=["department", "team_con"])
                    .groupby("team_con")["department"]
                    .agg(lambda s: s.mode()[0] if not s.mode().empty else np.nan)
                    .to_dict()
                )
                df_out_10["phong_ban_chuan"] = df_out_10["phong_ban_chuan"].fillna(
                    df_out_10["department"].astype(str).str.strip().map(team_to_parent)
                )

            df_out_10.drop(columns=["_id"], errors="ignore", inplace=True)
        else:
            df_out_10["phong_ban_chuan"] = df_out_10["department"]

        # Lớp 3: Chuẩn hóa các dòng NaN / Thẻ khách 41004xxx thành nhãn chuyên nghiệp
        guest_label = tr("Khách / Nhà thầu phụ", "Contractor / Guest")
        def clean_outlier_dept(row):
            val = str(row["phong_ban_chuan"]).strip()
            emp_str = str(row["employee_id"]).strip()
            if val in ["nan", "NaN", "None", ""] or emp_str.startswith("41004"):
                return guest_label
            return val

        df_out_10["department"] = df_out_10.apply(clean_outlier_dept, axis=1)

        # ----------------------------------------------------------------------
        # HIỂN THỊ BẢNG DỮ LIỆU ĐÃ CHUẨN HÓA
        # ----------------------------------------------------------------------
        display_out_table = df_out_10[[
            "employee_id", "month", "department", "actual_cost", "pred_cost", "error_cost", "segment"
        ]].copy()
        display_out_table.columns = [
            tr("Mã NV", "Emp ID"),
            tr("Tháng Test", "Test Month"),
            tr("Phòng Ban", "Department"),
            tr("Thực Tế Chi (đ)", "Actual Cost (VND)"),
            tr("Mô Hình Đoán (đ)", "Predicted Cost (VND)"),
            tr("Chênh Lệch (đ)", "Error Cost (VND)"),
            tr("Phân Nhóm Thực Tế", "Actual Segment")
        ]

        st.dataframe(
            display_out_table.style.format({
                display_out_table.columns[3]: "{:,.0f} đ",
                display_out_table.columns[4]: "{:,.0f} đ",
                display_out_table.columns[5]: "{:+,.0f} đ",
            }),
            hide_index=True,
            use_container_width=True
        )

        # ----------------------------------------------------------------------
        # 2 HỘP GIẢI TRÌNH NGUYÊN NHÂN CỤ THỂ
        # ----------------------------------------------------------------------
        col_out_exp1, col_out_exp2 = st.columns([1, 1])
        with col_out_exp1:
            st.markdown(f"""
            <div style="background-color: #fff1f2; border-left: 4px solid #f43f5e; padding: 14px; border-radius: 8px; font-size: 13px; line-height: 1.55;">
                <b style="color: #9f1239;">{tr('Trường hợp 1: Dự báo thừa ~880k - 946k đ (VD: NV #2578, #3600, #2936, thẻ 41004xxx)', 'Type 1: Over-predicted by ~880k - 946k VND (e.g., #2578, #3600, #2936, card 41004xxx)')}</b><br>
                {tr('• <i>Hiện tượng:</i> Lịch sử ăn đều ~20-21 bữa/tháng, nhưng tháng kiểm định đột ngột quẹt 0 bữa (thực tế 0 đ).<br>'
                    '• <i>Phòng ban ảnh hưởng:</i> <b>Admin, Piping, Thầu phụ...</b><br>'
                    '• <i>Nguyên nhân:</i> Nghỉ thai sản, nghỉ phép dài hạn không lương, hết hạn hợp đồng thầu phụ hoặc đi công tác.<br>'
                    '• <i>Kết luận kỹ thuật:</i> Không thuật toán nào có thể đoán trước việc nhân sự nghỉ việc/thai sản nếu HR không cấp trước danh sách.',
                    '• <i>Pattern:</i> Consistently ate ~20-21 meals/mo historically, but dropped to 0 meals (0 VND) in test month.<br>'
                    '• <i>Affected Depts:</i> <b>Admin, Piping, Contractors...</b><br>'
                    '• <i>Root Cause:</i> Maternity leave, extended unpaid leave, contractor contract expiration, or travel.<br>'
                    '• <i>Technical Note:</i> No algorithm can foresee unplanned HR leaves without advance roster integration.')}
            </div>
            """, unsafe_allow_html=True)

        with col_out_exp2:
            st.markdown(f"""
            <div style="background-color: #fff1f2; border-left: 4px solid #f43f5e; padding: 14px; border-radius: 8px; font-size: 13px; line-height: 1.55;">
                <b style="color: #9f1239;">{tr('Trường hợp 2: Dự báo thiếu ~938k đ (VD: NV #3762)', 'Type 2: Under-predicted by ~938k VND (e.g., #3762)')}</b><br>
                {tr('• <i>Hiện tượng:</i> Lịch sử chỉ ăn 0-1 bữa, mô hình đoán ~30k đ, nhưng tháng kiểm định quẹt thẻ ăn 22 bữa liên tục (968k đ).<br>'
                    '• <i>Phòng ban ảnh hưởng:</i> Khối sản xuất/dự án (<b>Production</b>).<br>'
                    '• <i>Nguyên nhân:</i> Nhân viên vừa hết thời gian thử việc hoặc được điều chuyển từ công trường về làm việc cố định tại văn phòng nhà máy.<br>'
                    '• <i>Cơ chế thích ứng:</i> Mô hình sẽ tự động cập nhật thói quen mới này ngay ở chu kỳ dự báo tiếp theo.',
                    '• <i>Pattern:</i> Historically ate 0-1 meal, predicted ~30k VND, but suddenly ate 22 meals (968k VND).<br>'
                    '• <i>Affected Depts:</i> <b>Production</b>.<br>'
                    '• <i>Root Cause:</i> Passed probation or transferred from field site to factory office.<br>'
                    '• <i>Adaptation:</i> Model automatically updates to this new pattern in the next forecasting cycle.')}
            </div>
            """, unsafe_allow_html=True)

    # MLOps Quality Gate
    st.subheader(f"🛡️ {tr('Khung Kiểm Soát Chất Lượng Mô Hình (Model Quality Gate)', 'Model Quality Gate & MLOps Governance')}")
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        st.markdown(tr("""
        **1. Walk-Forward Expanding Split**
        - Tuyệt đối **không dùng Random K-Fold Split** (tránh rò rỉ dữ liệu tương lai về quá khứ).
        - Mô hình học từ toàn bộ dữ liệu lịch sử trước tháng $T$ và chỉ dự báo cho duy nhất tháng $T$.
        """, """
        **1. Walk-Forward Expanding Split**
        - Strictly **no Random K-Fold Split** (prevents future-to-past data leakage).
        - Models train solely on historical records before month $T$ and evaluate only on month $T$.
        """))

    with q_col2:
        st.markdown(tr("""
        **2. Loại Bỏ "Tháng Cụt" Tự Động**
        - Kiểm tra nghiêm ngặt `is_complete_month`: tháng chưa kết thúc đến ngày cuối cùng sẽ bị loại bỏ khỏi tập huấn luyện.
        - Tránh tình trạng mô hình học nhầm số ngày ít thành xu hướng tiêu dùng giảm.
        """, """
        **2. Automated Incomplete Month Pruning**
        - Rigorous `is_complete_month` checks: partial calendar months are dropped from training.
        - Prevents models from misinterpreting truncated days as declining demand trends.
        """))

    with q_col3:
        st.markdown(tr("""
        **3. Cơ Chế Fallback 2 Lớp An Toàn**
        - Nhân viên đủ 3 tháng lịch sử: Chạy bằng CatBoost Machine Learning.
        - Nhân viên mới / khuyết lịch sử: Tự động gán **Group-Mean phòng ban**, không để mô hình dự báo ngẫu nhiên.
        """, """
        **3. Two-Tier Robust Fallback**
        - Staff with 3+ consecutive months: Evaluated via CatBoost Machine Learning.
        - New staff / sparse history: Assigned departmental **Group-Means** to avoid wild guesses.
        """))