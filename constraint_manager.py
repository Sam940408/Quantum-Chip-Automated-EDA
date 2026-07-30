import json
import os

def generate_constraints_json(filename='constraints.json'):
    """
    像 gds_json_.py 一樣，生成一份預設的約束條件字典並存檔。
    如果未來有新的參數需要限制，直接在這個字典裡面擴充即可。
    """
    constraints_data = {
        "qubit": {
                "gap_size": {"min": 30.0, "max": 50.0},
                "radius": {"min": 50.0, "max": 500.0},     # 配合極板總面積 (Pad Area)
                "slit_width": {"min": 50.0, "max": 70.0},  # 極板間距 (Gap)
                "rect_length": {"min": 50.0, "max": 1000.0}, # 配合極板總面積 (Pad Area)
                "rect_width": {"min": 50.0, "max": 1000.0},  # 配合極板總面積 (Pad Area)
                "q_c_dis": {"min": 70.0, "max": 120.0},
                #"q_re_dis": {"min": 10.0, "max": 30.0}       # 還沒使用
            },
        "coupler": {
            "arc_width": {"min": 10.0, "max": 100.0},
            "gap_size": {"min": 30.0, "max": 50.0},
            "center_dis": {"min": 5.0, "max": 15.0},     # 極板間距 (Gap)
            "length": {"min": 50.0, "max": 1000.0},
            "round_radius": {"min": 0, "max": 20.0}
        },
        "t_coupler": {
            "arm_length": {"min": 50.0, "max": 500.0},
            "arm_width": {"min": 5.0, "max": 100.0},
            "head_width": {"min": 10.0, "max": 200.0},
            "gap_size": {"min": 30.0, "max": 50.0},
            "center_dis": {"min": 5.0, "max": 15.0},
            "round_radius": {"min": 0, "max": 20.0}
        },
        "h_coupler": {
            "arm_length": {"min": 50.0, "max": 500.0},
            "head1_width": {"min": 10.0, "max": 200.0},
            "head2_width": {"min": 10.0, "max": 200.0},
            "gap_size": {"min": 30.0, "max": 50.0},
            "center_dis": {"min": 25.0, "max": 35.0},
            "round_radius": {"min": 10, "max": 10.0}
        },
        "feedline": {
            #"width": {"min": 2.0, "max": 50.0}
        },
        "global": {
            #"gnd_length": {"min": 2000.0, "max": 15000.0},
            #"gnd_width": {"min": 2000.0, "max": 15000.0}
        },
        "lom_settings": {
            "frequencies": {
                "w1": {"min": 4, "max": 8.0},
                "w2": {"min": 4, "max": 8.0}
            }
        },
        "Qubit_pra": {
            "thin_cond_thickness_nm": {"min": 50, "max": 200}# 厚度，單位 nm
        },
        
        "substrate": {
            "thickness": {"min": 500, "max": 500.0}  # 基板厚度，單位 um
        },
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(constraints_data, f, indent=4, ensure_ascii=False)
        print(f"✅ 成功生成預設限制值參數倉庫：{os.path.abspath(filename)}")
        return True
    except Exception as e:
        print(f"❌ 生成失敗：{e}")
        return False


class ConstraintManager:
    def __init__(self, filepath='constraints.json'):
        """
        初始化限制值管理員
        :param filepath: 限制值 JSON 檔案的路徑
        """
        self.filepath = filepath
        self.constraints = {}
        self._load_constraints()

    def _load_constraints(self):
        """從 JSON 檔案載入限制值，若檔案不存在則自動生成一份！"""
        if not os.path.exists(self.filepath):
            print(f"⚠️ 找不到 {self.filepath}，將自動生成預設規格書...")
            generate_constraints_json(self.filepath)
            
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                self.constraints = json.load(f)
            print(f"✅ 成功載入限制值檔案: {self.filepath}")
        except Exception as e:
            print(f"❌ 讀取檔案失敗: {e}")

    def save_constraints(self):
        """將目前的限制值寫回 JSON 檔案 (如果程式中途有動態修改的話)"""
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.constraints, f, indent=4, ensure_ascii=False)
            print(f"💾 限制值已儲存至: {self.filepath}")
        except Exception as e:
            print(f"❌ 儲存檔案失敗: {e}")

    def update_constraint(self, category: str, param: str, min_val: float = None, max_val: float = None):
        """更新或新增某個參數的限制值"""
        if category not in self.constraints:
            self.constraints[category] = {}
            
        if param not in self.constraints[category]:
            self.constraints[category][param] = {}

        if min_val is not None:
            self.constraints[category][param]['min'] = float(min_val)
        if max_val is not None:
            self.constraints[category][param]['max'] = float(max_val)
            
        self.save_constraints()

    def validate_value(self, category: str, param: str, value: float) -> tuple[bool, str]:
        """驗證數值是否符合定義的限制區間"""
        if category not in self.constraints or param not in self.constraints[category]:
            return True, f"⚠️ 未定義 {category}.{param} 的限制值，跳過檢查。"

        limits = self.constraints[category][param]
        min_limit = limits.get('min')
        max_limit = limits.get('max')

        if min_limit is not None and value < min_limit:
            return False, f"❌ {category}.{param} 的數值 ({value}) 小於最小值 ({min_limit})！"
        
        if max_limit is not None and value > max_limit:
            return False, f"❌ {category}.{param} 的數值 ({value}) 大於最大值 ({max_limit})！"

        return True, f"✅ {category}.{param} ({value}) 符合規範。"

    def get_limits(self, category: str, param: str) -> dict:
        """取得特定參數的上下限"""
        return self.constraints.get(category, {}).get(param, None)

# ==========================================
# 獨立執行測試
# ==========================================
if __name__ == "__main__":
    # 如果你直接執行 python constraint_manager.py
    # 它就會像 gds_json_.py 一樣，幫你生成/重置 constraints.json
    print("🌟 執行限制值管理員腳本...")
    generate_constraints_json('constraints.json')