# drc_rules.py
import math

# ==========================================
# 1. 基礎限制值檢查 (連動 Manager)
#作用： 這是最自動化的防線。
# 它會把 layout_parameters.json 裡面的每一個數字，
# 拿去問 ConstraintManager：「這個數字有沒有超出 constraints.json 規定的上下限？」
#特點： 未來在 JSON 加了新參數，只要 constraints.json 有定義範圍，
# 這個函式就會自動抓取並檢查，完全不用改程式碼。
# ==========================================
def check_basic_bounds(params, manager):
    errors, warnings = [], []
    for category, configs in params.items():
        if not isinstance(configs, dict): continue
        for param, value in configs.items():
            if isinstance(value, (int, float)):
                is_valid, msg = manager.validate_value(category, param, float(value))
                if not is_valid:
                    errors.append(f"[數值越界] {msg.replace('❌ ', '')}")
    return errors, warnings

# ==========================================
# 2. 幾何極限檢查
#作用： 把關微影製程極限。
#邏輯： 檢查所有的縫隙（Gap、Slit）是否大於製程能洗出來的最小尺寸（例如 1.0 μm）。
#同時檢查距離參數是否不小心被設為負值。
# ==========================================
def check_minimum_features(params, manager):
    errors, warnings = [], []
    min_resolution = 1.0  # 微影極限 (um)
    
    q_gap = params.get("qubit", {}).get("gap_size", 0)
    q_slit = params.get("qubit", {}).get("slit_width", 0)
    q_c_dis = params.get("qubit", {}).get("q_c_dis", 0)

    if 0 < q_gap < min_resolution:
        errors.append(f"[微影違規] Qubit gap_size ({q_gap}) 小於極限 {min_resolution}")
    if 0 < q_slit < min_resolution:
        errors.append(f"[微影違規] Qubit slit_width ({q_slit}) 小於極限 {min_resolution}")
    if q_c_dis <= 0:
        errors.append(f"[短路風險] Qubit 與 Coupler 的間距 q_c_dis ({q_c_dis}) 過小或為負值")
        
    return errors, warnings

# ==========================================
# 3. 縫隙與本體衝突檢查
# 作用： 防止「自我毀滅」的幾何生成。
# 邏輯： 如果是一個半徑 200 μm 的圓形 Qubit，但切割縫隙（Slit）卻設定成 500 μm，
# 這會導致在 GDS 繪製時整個圖形消失或出錯。此規則專門防範這種比例失衡的參數。
# ==========================================
def check_qubit_slit(params, manager):
    errors, warnings = [], []

    q_type = params.get(
        "toggles",
        {},
    ).get("qubit_type", "circle")

    q_cfg = params.get("qubit", {})
    slit_width = float(
        q_cfg.get("slit_width", 0)
    )

    if q_type == "circle":
        radius = float(
            q_cfg.get("radius", 0)
        )

        if slit_width >= 2.0 * radius:
            errors.append(
                "[幾何錯誤] 圓形 Qubit 的 slit_width "
                f"({slit_width}) 不得大於或等於直徑 "
                f"({2.0 * radius})。"
            )

    elif q_type == "rect":
        rect_length = float(
            q_cfg.get("rect_length", 0)
        )
        rect_width = float(
            q_cfg.get("rect_width", 0)
        )
        cut_angle = math.radians(
            float(q_cfg.get("cut_angle", 90))
        )

        # 矩形沿 slit 法向方向的投影尺寸
        normal_span = (
            abs(math.sin(cut_angle)) * rect_length
            + abs(math.cos(cut_angle)) * rect_width
        )

        if slit_width >= normal_span:
            errors.append(
                "[幾何錯誤] 矩形 Qubit 的 slit_width "
                f"({slit_width}) 不得大於或等於切割方向尺寸 "
                f"({normal_span:.3f})。"
            )

    return errors, warnings

# ==========================================
# 4. 面積規範檢查
#作用： 確保物理特性符合您簡報上的實驗室推薦規範。
# 邏輯： 針對圓形或矩形 Qubit 計算其總面積，並換算成 mm²。
# 如果不落在 0.03 ~ 0.12 mm² 這個範圍內，就會跳出錯誤，確保生成的電容值具有物理意義。
# ==========================================
def check_pad_area(params, manager):
    errors, warnings = [], []
    q_type = params.get("toggles", {}).get("qubit_type", "circle")
    
    if q_type == "rect":
        length = params.get("qubit", {}).get("rect_length", 0)
        width = params.get("qubit", {}).get("rect_width", 0)
        area = length * width
    else:
        radius = params.get("qubit", {}).get("radius", 0)
        area = math.pi * (radius ** 2)
        
    area_mm2 = area / 1_000_000
    if not (0.03 <= area_mm2 <= 0.12):
        errors.append(f"[面積違規] Qubit 面積為 {area_mm2:.4f} mm²，超出實驗規範 (0.03 ~ 0.12 mm²)。")
        
    return errors, warnings

# ==========================================
# 5. 邊界與跨度檢查
#作用： 防止元件畫到晶片外面。
#邏輯： 根據 Qubit 的尺寸、與 Coupler 的間距、以及 Coupler 本身的長度，把 X 軸的總長度加總起來。
#如果這個總長度超出了 Ground Plane 的一半（也就是超出晶片邊緣），就會報錯。
# ==========================================
def check_chip_boundaries(params, manager):
    errors, warnings = [], []
    try:
        gnd_len = params.get("global", {}).get("gnd_length", 5000)
        gnd_half = gnd_len / 2.0
        c_type = params.get("toggles", {}).get("coupler_type", "arc")
        
        # ⭐️ 精確對齊 geometry.py 的實體總長度計算
        if c_type == "t_shape":
            c_dis = params.get("t_coupler", {}).get("center_dis", 0)
            c_len = params.get("t_coupler", {}).get("head_width", 0) + params.get("t_coupler", {}).get("arm_length", 0)
        elif c_type == "h_shape":
            c_dis = params.get("h_coupler", {}).get("center_dis", 0)
            c_len = (params.get("h_coupler", {}).get("head1_width", 0) + 
                     params.get("h_coupler", {}).get("arm_length", 0) + 
                     params.get("h_coupler", {}).get("head2_width", 0))
        else: 
            c_dis = params.get("coupler", {}).get("center_dis", 0)
            c_len = params.get("coupler", {}).get("length", 0)

        # 取得 Qubit 半寬
        q_type = params.get("toggles", {}).get("qubit_type", "circle")
        qubit_half_x = params.get("qubit", {}).get("rect_length", 0) / 2.0 if q_type == "rect" else params.get("qubit", {}).get("radius", 0)

        q_c_dis = params.get("qubit", {}).get("q_c_dis", 0)
        q_gap = params.get("qubit", {}).get("gap_size", 0)
        
        # 由中心點向外推算最外側邊界
        qubit_cx = c_dis + c_len + q_c_dis + qubit_half_x
        max_x_boundary = qubit_cx + qubit_half_x + q_gap

        if max_x_boundary > gnd_half:
            errors.append(f"[邊界溢出] 元件最右側邊界 ({max_x_boundary}) 超出 GND 邊界 ({gnd_half})。")
        elif max_x_boundary > gnd_half * 0.9:
            warnings.append(f"[邊界警告] 元件最右側邊界 ({max_x_boundary}) 距離 GND 邊界 ({gnd_half}) 過近。")
            
    except Exception as e:
        errors.append(f"[參數缺失] 檢查晶片邊界時發生例外錯誤: {e}")
        
    return errors, warnings

# ==========================================
# 6. 量子物理參數與能量比值檢查 (Transmon Energy)
# ==========================================
def check_transmon_energy(params, manager):
    errors, warnings = [], []
    
    # 從 LOM 設定中抓取目標頻率 (假設單位為 GHz)
    freq_settings = params.get("lom_settings", {}).get("frequencies", {})
    w1 = freq_settings.get("w1", 0)  # Qubit 1 目標頻率
    w2 = freq_settings.get("w2", 0)  # Qubit 2 目標頻率
    
    # 假設 Q3D 萃取後的 Ec 會被自動填入 qubit 區塊中
    # 實務上雙位元系統會需要 ec1_ghz 與 ec2_ghz
    ec1_ghz = params.get("qubit", {}).get("ec1_ghz", 0)
    ec2_ghz = params.get("qubit", {}).get("ec2_ghz", 0)

    # 取得比值限制條件
    limits = manager.get_limits("qubit", "ej_ec_ratio")
    min_ratio = limits.get("min", 50) if limits else 50
    max_ratio = limits.get("max", 80) if limits else 80

    # --- 定義內部檢查副程式，方便同時驗證 Q1 與 Q2 ---
    def verify_qubit_energy(q_name, target_fq, ec_ghz):
        if target_fq <= 0 or ec_ghz <= 0:
            return  # 若無數值則跳過檢查 (可能還在純幾何繪製階段)

        # 公式反推：Ej = (fq + Ec)^2 / (8 * Ec)
        ej_ghz = ((target_fq + ec_ghz) ** 2) / (8 * ec_ghz)
        ratio = ej_ghz / ec_ghz

        if ratio < min_ratio:
            # 🌟 修改這裡：把 errors.append 改成 warnings.append，並標註為預測值
            warnings.append(f"[預測雜訊風險] {q_name} 的預估 Ej/Ec 比值為 {ratio:.2f} (低於 {min_ratio})。"
                            f"程式將繼續執行，請以 Step 4 Q3D 萃取後的真實值為準。")
        elif ratio > max_ratio:
            warnings.append(f"[預測非諧性警告] {q_name} 的預估 Ej/Ec 比值為 {ratio:.2f} (高於 {max_ratio})。")

        warnings.append(f"💡 [製程目標預測] 若要使 {q_name} 達到 {target_fq} GHz (預估 Ec = {ec_ghz} GHz)，推薦 Ej = {ej_ghz:.4f} GHz。")
    # 分別檢查兩顆 Qubit
    verify_qubit_energy("Qubit 1", w1, ec1_ghz)
    verify_qubit_energy("Qubit 2", w2, ec2_ghz)

    return errors, warnings

# ==========================================
# 7. 元件間距與挖空區重疊檢查 (Overlap & Spacing)
# ==========================================
def check_overlap_conflicts(params, manager):
    errors, warnings = [], []

    # 1. 取得 Qubit 相關參數
    q_c_dis = params.get("qubit", {}).get("q_c_dis", 0)
    q_gap = params.get("qubit", {}).get("gap_size", 0)

    # 2. 取得 Coupler 的 gap_size (需動態支援 arc, t_shape, h_shape)
    c_type = params.get("toggles", {}).get("coupler_type", "arc")
    if c_type == "t_shape":
        c_gap = params.get("t_coupler", {}).get("gap_size", 0)
    elif c_type == "h_shape":
        c_gap = params.get("h_coupler", {}).get("gap_size", 0)
    else:
        c_gap = params.get("coupler", {}).get("gap_size", 0)

    # 3. 檢查邏輯：金屬間距是否足以容納兩者的挖空區
    minimum_ground_neck = 10.0

    required_distance = (
        q_gap
        + c_gap
        + minimum_ground_neck
    )

    if q_c_dis < required_distance:
        errors.append(
            "[結構重疊] Qubit 與 Coupler 間距不足："
            f"q_c_dis={q_c_dis} µm，"
            f"至少需要 {required_distance} µm "
            f"(Qubit gap {q_gap} + Coupler gap {c_gap} "
            f"+ GND neck {minimum_ground_neck})。"
        )
    return errors, warnings

# ==========================================
# 8. 尖角效應防護 (Sharp Corner Filleting Check)
# ==========================================
def check_sharp_corners(params, manager):
    errors, warnings = [], []
    
    # 定義安全的最小圓角半徑 (μm)
    SAFE_RADIUS = 2.0

    # 1. 檢查 Qubit (僅針對矩形設計進行防護)
    q_type = params.get("toggles", {}).get("qubit_type", "circle")
    if q_type == "rect":
        # 取得 qubit 區塊的 round_radius，若找不到預設為 0
        q_radius = params.get("qubit", {}).get("round_radius", 0)
        
        if q_radius < SAFE_RADIUS:
            errors.append(f"[尖角損耗風險] 矩形 Qubit 的圓角半徑 (round_radius = {q_radius}) 小於安全值 {SAFE_RADIUS} μm。\n"
                          f"          這會導致極端電場集中與嚴重的 TLS 損耗。請在 JSON 的 qubit 區塊中加入或調大 round_radius。")

    # 2. 檢查 Coupler (動態識別不同形狀的耦合器)
    c_type = params.get("toggles", {}).get("coupler_type", "arc")
    if c_type == "t_shape":
        c_radius = params.get("t_coupler", {}).get("round_radius", 0)
        c_name = "T型耦合器 (t_coupler)"
    elif c_type == "h_shape":
        c_radius = params.get("h_coupler", {}).get("round_radius", 0)
        c_name = "H型耦合器 (h_coupler)"
    else:
        c_radius = params.get("coupler", {}).get("round_radius", 0)
        c_name = "弧形耦合器 (coupler)"

    if c_radius < SAFE_RADIUS:
        # Coupler 的尖角有時候是妥協於佈線空間，所以這裡發出警告 (warning) 而非強制錯誤 (error)
        warnings.append(f"[尖角損耗警告] {c_name} 的圓角半徑為 {c_radius} μm。\n"
                        f"          直角邊緣會增加局部介電損耗，若空間允許，建議設定大於 {SAFE_RADIUS} μm 的鈍化圓角。")

    return errors, warnings

def check_coupler_geometry(params, manager):
    errors, warnings = [], []
    c_type = params.get("toggles", {}).get("coupler_type", "arc")

    # ==========================================
    # A. 針對 Arc Coupler 的銜接檢查 (承襲原 drc_checker)
    # ==========================================
    if c_type == "arc":
        try:
            q_type = params.get("toggles", {}).get("qubit_type", "circle")
            qubit_half_x = params.get("qubit", {}).get("rect_length", 200) / 2.0 if q_type == "rect" else params.get("qubit", {}).get("radius", 0)
            
            c_dis = params.get("coupler", {}).get("center_dis", 0)
            c_len = params.get("coupler", {}).get("length", 0)
            q_c_dis = params.get("qubit", {}).get("q_c_dis", 0)
            
            qubit_cx = c_dis + c_len + q_c_dis + qubit_half_x
            arc_width = params.get("coupler", {}).get("arc_width", 0)
            r_inner = qubit_half_x + q_c_dis

            docking_x = qubit_cx - (r_inner + arc_width / 2.0)
            start_x = 0.0 + c_dis
            tbar_len = docking_x - start_x

            if tbar_len <= 0:
                errors.append(f"[幾何錯誤] Arc Coupler T-bar 計算長度為負值 ({tbar_len})。請檢查 center_dis 或 q_c_dis。")
        except KeyError as e:
            errors.append(f"[參數缺失] 檢查 Arc Coupler 時缺少參數: {e}")

    # ==========================================
    # B. 針對 T_Shape Coupler 的比例防呆
    # ==========================================
    elif c_type == "t_shape":
        t_cfg = params.get("t_coupler", {})
        arm_w = t_cfg.get("arm_width", 0)
        head_l = t_cfg.get("head_length", 0)
        
        if arm_w >= head_l:
            errors.append(f"[結構畸形] T型耦合器的手臂寬度 (arm_width = {arm_w}) 大於或等於頭部總長度 (head_length = {head_l})，這在版圖幾何上不合理。")

    # ==========================================
    # C. 針對 H_Shape Coupler 的比例防呆
    # ==========================================
    elif c_type == "h_shape":
        h_cfg = params.get("h_coupler", {})

        arm_width = float(
            h_cfg.get("arm_width", 0)
        )
        head1_length = float(
            h_cfg.get("head1_length", 0)
        )
        head2_length = float(
            h_cfg.get("head2_length", 0)
        )
        head1_width = float(
            h_cfg.get("head1_width", 0)
        )
        head2_width = float(
            h_cfg.get("head2_width", 0)
        )
        round_radius = float(
            h_cfg.get("round_radius", 0)
        )

        minimum_head_margin = 10.0

        if (
            arm_width + minimum_head_margin
            > min(head1_length, head2_length)
        ):
            errors.append(
                "[結構畸形] H Coupler 的 head 長度必須至少比 "
                f"arm_width 多 {minimum_head_margin} µm。"
            )

        narrowest_width = min(
            arm_width,
            head1_width,
            head2_width,
        )

        if 2.0 * round_radius > narrowest_width:
            errors.append(
                "[圓角錯誤] H Coupler 的 round_radius 過大："
                f"2R={2.0 * round_radius} µm，"
                f"但最窄金屬只有 {narrowest_width} µm。"
            )

    return errors, warnings


# ==========================================
# 🔧 註冊中心：隨時拔插您想要的檢查規則
# ==========================================
ACTIVE_RULES = [
    check_basic_bounds,
    check_minimum_features,
    check_qubit_slit,
    #check_pad_area,
    check_chip_boundaries,
    check_transmon_energy,
    check_overlap_conflicts,
    check_sharp_corners,
    check_coupler_geometry
    # 若某個功能不想檢查，直接在這裡加上註解 (例如 # check_pad_area) 即可拔除
]