import json
import os

def generate_layout_json(filename='layout_parameters.json', print_content=False):
    layout_data = {
        "layer_name": "QuantumChip_FloatingQCQ",
        
        #元件開關控制 (True: 生成, False: 關閉)
        "toggles": {
            "enable_gnd": True,       # 是否生成接地鋪銅 (GND)
            "enable_qubit": True,     # 是否生成量子位元 (Qubit)
            "enable_coupler": True,   # 是否生成耦合器 (Coupler)
           "coupler_type": "h_shape",# 選配 Coupler 形狀: "arc" (圓弧), "t_shape" (T型) 或 "h_shape" (H型)
            "enable_feedline": False,   # 是否生成饋線 (Feedline)

            "qubit_type": "rect"   # 選配 Qubit 形狀: "circle" (圓形) 或 "rect" (長方形)
        },
        
        # 總體參數 (單位：無因次原始數值，程式內部將乘以 scale 轉換為 um)
        "global": {
            "gnd_length": 5000,
            "gnd_width": 5000
        },

        # Qubit 參數 (圓形)
        "qubit": {
            # 圓形專用參數
            "radius": 200,
            "points": 180,
            
            # 長方形專用參數
            "rect_length": 300,        # 長方形的 X 方向長度 (通常在浮動雙位元中代表單邊寬度)
            "rect_width": 300,         # 長方形的 Y 方向寬度
            "round_radius": 10,

            # 圓形/長方形共用參數
            "gap_size": 30.0,
            "slit_width": 10.0,          # 兩個半邊之間的縫隙寬度
            "cut_angle": 90,           # 0: 水平切割, 90: 垂直切割, 可輸入任意角度
            "q_c_dis": 70,
            "q_re_dis": 30,            # 沒用

            # --- 預留給電容矩陣萃取結果的欄位 (單位: GHz) ---
            "ec1_ghz": 0.25,  # 預設給個合理值測試用 (對應 250 MHz)
            "ec2_ghz": 0.23,  # 對應 230 MHz
        },            

        # 耦合器參數 (弧形 + T-bar)
        "coupler": {
            "center_dis": 5.0,#距離中心距離
            "length": 150,
            "half_width": 40,#這只有一半
            "gap_size": 30.0,
            "arc_width": 50,
            "arc_start_angle": 110,
            "arc_stop_angle": 250,
            "safe_ext": 100,        #基本不動
            "round_radius": 10
        },

        # T型耦合器參數 (T-bar)
        "t_coupler": {
            "center_dis": 5.0,     # 距離中心距離
            "arm_length": 150,     # 橫向手臂長度
            "arm_width": 20,       # 橫向手臂寬度
            "head_length": 300,    # T字頭部總長度 (Y方向)
            "head_width": 50,      # T字頭部寬度 (X方向)
            "gap_size": 30.0,         # 挖空區大小
            "round_radius": 10      # 圓角半徑
        },

        "h_coupler": {
            "center_dis": 5.0,     # 距離中心點的 X 距離
            "arm_length": 250,     # 中間橫向橋樑的長度
            "arm_width": 10,       # 中間橫向橋樑的寬度
            "head1_length": 300,   # 左側(內側)垂直棒長度 (Y方向)
            "head1_width": 50,     # 左側(內側)垂直棒寬度/粗細 (X方向)
            "head2_length": 300,   # 右側(外側)垂直棒長度 (Y方向)
            "head2_width": 50,     # 右側(外側)垂直棒寬度/粗細 (X方向)
            "gap_size": 30.0,         # 挖空區大小
            "round_radius": 10      # 圓角半徑
        },

        # 共振腔參數 (U型 + 蛇形彎折)
        "resonator": {
            "u_length_right": 600,
            "u_length_up": 370,
            "u_length_right2": 75,
            "u_width": 10,
            "arc_width": 50,
            "arc_start_angle": -80,
            "arc_stop_angle": 80,
            "trace_width": 10,
            "pitch": 60,
            "straight_len": 300,
            "turns": 3,
            "round_radius": 10
        },

        # 饋線參數 (Feedline)
        "feedline": {
            "start_shift_y": 400,
            "width": 10,
            "len_l1": 1000,
            "len_u2": 170,
            "len_r3": 2300,
            "len_d4": 170,
            "len_r5": 1675,
            "round_radius": 30
        },
        
        # 圖層設定
        "layer_mapping": {
            "GND": 6,
            "qubit1_left": 0,
            "qubit1_right": 1,
            "qubit2_left": 4,
            "qubit2_right": 5,
            "coupler1_left": 2,
            "coupler1_right": 3,
        },

        "project_settings": {
            "project_name": "Floating_Qubit_Project",
            "design_name": "Q3D_Extraction",
            "setup_name": "Setup_5GHz",
            "aedt_version": "2021.2"
        },
        
        "simulation": {
            "adaptive_freq_ghz": 5.0
        },

        "Qubit_pra": {
           "thin_cond_thickness_nm": 200.0,
           "material_Q": "aluminum"
        },

        "substrate": {
            "name": "Silicon_Substrate",
            "material_S": "silicon",
            "thickness": 500  # 基板厚度，單位 um
        },

        "lom_settings": {
            "constants": {
                "h_bar": 1.05457180013e-34,
                "phi0": 2.06783375846e-15,
                "e": 1.602176620898e-19,
                "Z0": 50
            },
            "frequencies": {
                "w1": 6.6,
                "w2": 6.7,
                "wc": 4.0
            },
            "isSymmetric": 1,
            "sweep_settings": {
                "wc_lower_start": 4,
                "wc_lower_stop": 6,
                "wc_upper_start": 6,
                "wc_upper_stop": 8,
                "num_points": 10000
            }
        },
      }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(layout_data, f, indent=4, ensure_ascii=False)
        print(f"✅ 成功生成參數倉庫：{os.path.abspath(filename)}")
        return True
    except Exception as e:
        print(f"❌ 生成失敗：{e}")
        return False

if __name__ == "__main__":
    generate_layout_json(print_content=True)