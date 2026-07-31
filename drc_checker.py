# drc_checker.py
import json
import os
import sys
from constraint_manager import ConstraintManager
from drc_rules import ACTIVE_RULES  # 引入外部配置好的規則列

class QuantumChipDRC:
    def __init__(self, layout_filepath='layout_parameters.json', constraint_filepath='constraints.json'):
        self.filepath = layout_filepath
        self.constraint_manager = ConstraintManager(constraint_filepath)
        self.params = {}
        self.errors = []
        self.warnings = []

    def load_params(self):
        if not os.path.exists(self.filepath):
            self.errors.append(f"找不到參數檔案: {self.filepath}")
            return False
        with open(self.filepath, 'r', encoding='utf-8') as f:
            self.params = json.load(f)
        return True

    def run_checks(self):
        if not self.load_params():
            self.print_report()
            return False
        
        # 動態執行所有在 drc_rules 註冊的規則 (外掛拔插核心)
        for rule_func in ACTIVE_RULES:
            # 每個規則函式獨立執行，並將回報結果接回主程式
            rule_errors, rule_warnings = rule_func(self.params, self.constraint_manager)
            self.errors.extend(rule_errors)
            self.warnings.extend(rule_warnings)
        
        self.print_report()
        return len(self.errors) == 0

    def print_report(self):
        print("=" * 50)
        print("🛠️  量子晶片參數 DRC & 限制值檢查報告")
        print("=" * 50)
        
        if not self.errors and not self.warnings:
            print("✅ 檢查通過：參數符合約束範圍，且無結構衝突。")
        else:
            for w in self.warnings:
                print(f"⚠️  警告: {w}")
            for e in self.errors:
                print(f"❌ 錯誤: {e}")
            
            if self.errors:
                print("-" * 50)
                print("🛑 流程終止：請修正參數後再執行！")

# ==========================================
# 執行測試
# ==========================================
if __name__ == "__main__":
    checker = QuantumChipDRC('layout_parameters.json', 'constraints.json')
    is_safe = checker.run_checks()
    
    if not is_safe:
        sys.exit(1)
    else:
        sys.exit(0)