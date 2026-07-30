import math
import klayout.db as pya

# =========================================================================
# 共用輔助函數
# =========================================================================
def circle_points(dbu, corner_radius, region_a, region_b, points=36):
    final_region = region_a + region_b
    final_region.merge()
    radius_dbu = int(corner_radius / dbu)
    final_region = final_region.round_corners(radius_dbu, radius_dbu, points)
    final_region.merge()
    return final_region

def round_cpw_corners(dbu, region, inner_radius, trace_width, points=36):
    r_inner_dbu = int(inner_radius / dbu)
    r_outer_dbu = int((inner_radius + trace_width) / dbu) 
    final_region = region.round_corners(r_inner_dbu, r_outer_dbu, points)
    final_region.merge()
    return final_region

def draw_arc_capacitor(dbu, cx, cy, r_inner, r_outer, start_angle, stop_angle, arc_width):
    pts = []
    r_mid = r_inner + arc_width / 2.0
    r_cap = arc_width / 2.0
    
    for a in range(int(start_angle), int(stop_angle) + 1):
        r_a = a * math.pi / 180.0
        pts.append(pya.DPoint(cx + r_outer * math.cos(r_a), cy + r_outer * math.sin(r_a)))

    stop_rad = stop_angle * math.pi / 180.0
    cap_stop_x = cx + r_mid * math.cos(stop_rad)
    cap_stop_y = cy + r_mid * math.sin(stop_rad)
    for i in range(1, 18): 
        cap_rad = (stop_angle + (i * 10)) * math.pi / 180.0
        pts.append(pya.DPoint(cap_stop_x + r_cap * math.cos(cap_rad), cap_stop_y + r_cap * math.sin(cap_rad)))
        
    for a in range(int(stop_angle), int(start_angle) - 1, -1):
        r_a = a * math.pi / 180.0
        pts.append(pya.DPoint(cx + r_inner * math.cos(r_a), cy + r_inner * math.sin(r_a)))

    start_rad = start_angle * math.pi / 180.0
    cap_start_x = cx + r_mid * math.cos(start_rad)
    cap_start_y = cy + r_mid * math.sin(start_rad)
    for i in range(1, 18):
        cap_rad = ((start_angle - 180) + (i * 10)) * math.pi / 180.0
        pts.append(pya.DPoint(cap_start_x + r_cap * math.cos(cap_rad), cap_start_y + r_cap * math.sin(cap_rad)))

    return pya.Region(pya.DPolygon(pts).to_itype(dbu))

# =========================================================================
# 元件類別
# =========================================================================
class GroundPlane:
    def __init__(self, cx, cy, length, width):
        self.cx = cx
        self.cy = cy
        self.length = length
        self.width = width

    def get_metal_region(self, dbu):
        bbox = pya.DBox(self.cx - self.length/2, self.cy - self.width/2, 
                        self.cx + self.length/2, self.cy + self.width/2)
        return pya.Region(pya.DPolygon(bbox).to_itype(dbu))

class CircularQubit:
    def __init__(self, cx, cy, radius, gap_size, points):
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.gap_size = gap_size
        self.points = points

    def get_metal_region(self, dbu):
        bbox = pya.DBox(self.cx - self.radius, self.cy - self.radius, 
                        self.cx + self.radius, self.cy + self.radius)
        return pya.Region(pya.DPolygon.ellipse(bbox, self.points).to_itype(dbu))

    def get_gap_region(self, dbu):
        metal = self.get_metal_region(dbu)
        gap_dbu = int(self.gap_size / dbu)
        return metal.sized(gap_dbu) - metal

class FloatingRectQubit:
    def __init__(self, cx, cy, length, width, gap_size, slit_width, cut_angle, round_radius):
        self.cx = cx
        self.cy = cy
        self.length = length
        self.width = width
        self.gap_size = gap_size
        self.slit_width = slit_width
        self.cut_angle = cut_angle
        self.round_radius = round_radius
        self.points = 64  # 設定圓角解析度 (64邊形逼近)

    def get_metal_regions(self, dbu):
        # 1. 建立完整矩形
        bbox = pya.DBox(self.cx - self.length / 2.0, self.cy - self.width / 2.0, 
                        self.cx + self.length / 2.0, self.cy + self.width / 2.0)
        full_rect = pya.Region(pya.DPolygon(bbox).to_itype(dbu))
        
        # 2. 建立切割矩形
        cut_length = (self.length + self.width) * 2.0
        cut_box = pya.DBox(-cut_length / 2.0, -self.slit_width / 2.0,
                           cut_length / 2.0, self.slit_width / 2.0)
        cut_poly = pya.DPolygon(cut_box)
        
        # 3. 旋轉與平移 (支援任意切角)
        trans = pya.DCplxTrans(1.0, self.cut_angle, False, pya.DVector(self.cx, self.cy))
        cut_poly_transformed = cut_poly.transformed(trans)
        cut_region = pya.Region(cut_poly_transformed.to_itype(dbu))
        
        # 4. 相減得到分割後的金屬
        full_metal = (full_rect - cut_region).merge()

        # ==========================================
        # 🌟 新增：在分離左右極板前，統一進行圓角處理
        # ==========================================
        r_dbu = int(self.round_radius / dbu)
        if r_dbu > 0:
            full_metal = full_metal.round_corners(r_dbu, r_dbu, self.points).merge()

        # 5. 自動分離多邊形 (依 X 座標中心由左至右排序，區分 Left 與 Right)
        polygons = list(full_metal.each())
        polygons.sort(key=lambda p: p.bbox().center().x)
        
        if len(polygons) >= 2:
            metal_left = pya.Region(polygons[0])
            metal_right = pya.Region(polygons[1])
        else:
            metal_left = full_metal
            metal_right = pya.Region()
            
        return metal_left, metal_right

    def get_gap_region(self, dbu):
        # 呼叫金屬方法取得當前實際金屬
        metal_left, metal_right = self.get_metal_regions(dbu)
        metal = metal_left + metal_right
        
        # 建立外擴 gap 區域的基底矩形
        bbox = pya.DBox(self.cx - self.length / 2.0, self.cy - self.width / 2.0, 
                        self.cx + self.length / 2.0, self.cy + self.width / 2.0)
        full_rect = pya.Region(pya.DPolygon(bbox).to_itype(dbu))
        gap_dbu = int(self.gap_size / dbu)
        
        # 原始的方形挖空區
        total_hole = full_rect.sized(gap_dbu)
        
        # ==========================================
        # 🌟 新增：挖空區也要導圓角 (半徑必須是: 金屬圓角 + Gap寬度)
        # ==========================================
        r_dbu = int(self.round_radius / dbu)
        if r_dbu > 0:
            outer_r_dbu = r_dbu + gap_dbu
            total_hole = total_hole.round_corners(outer_r_dbu, outer_r_dbu, self.points).merge()
        
        return (total_hole - metal).merge()

class FloatingQubit:
    def __init__(self, cx, cy, radius, gap_size, slit_width, cut_angle, points):
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.gap_size = gap_size
        self.slit_width = slit_width
        self.cut_angle = cut_angle
        self.points = points

    def get_metal_regions(self, dbu):
        """
        回傳 (metal_left, metal_right) 兩個獨立的 Region
        """
        # 1. 完整圓形
        bbox = pya.DBox(self.cx - self.radius, self.cy - self.radius, 
                        self.cx + self.radius, self.cy + self.radius)
        full_circle = pya.Region(pya.DPolygon.ellipse(bbox, self.points).to_itype(dbu))
        
        # 2. 建立切割矩形
        cut_length = self.radius * 3.0
        cut_box = pya.DBox(-cut_length / 2.0, -self.slit_width / 2.0,
                           cut_length / 2.0, self.slit_width / 2.0)
        cut_poly = pya.DPolygon(cut_box)
        
        # 3. 旋轉與平移 (支援任意切角)
        trans = pya.DCplxTrans(1.0, self.cut_angle, False, pya.DVector(self.cx, self.cy))
        cut_poly_transformed = cut_poly.transformed(trans)
        cut_region = pya.Region(cut_poly_transformed.to_itype(dbu))
        
        # 4. 相減得到分割後的金屬
        full_metal = (full_circle - cut_region).merge()

        # ==========================================
        # 5. 自動分離多邊形 (不再需要外部拿遮罩硬切)
        # ==========================================
        polygons = list(full_metal.each())
        
        # 依照多邊形中心點的 X 座標進行排序 (確保左邊的在前面，右邊的在後面)
        polygons.sort(key=lambda p: p.bbox().center().x)
        
        if len(polygons) >= 2:
            metal_left = pya.Region(polygons[0])
            metal_right = pya.Region(polygons[1])
        else:
            # 防呆：如果縫隙太小沒有切斷，就全部丟給 left
            metal_left = full_metal
            metal_right = pya.Region()
            
        return metal_left, metal_right

    def get_gap_region(self, dbu):
        # 呼叫新的方法並將左右合併，用以計算整體的 gap
        metal_left, metal_right = self.get_metal_regions(dbu)
        metal = metal_left + metal_right
        
        bbox = pya.DBox(self.cx - self.radius, self.cy - self.radius, 
                        self.cx + self.radius, self.cy + self.radius)
        full_circle = pya.Region(pya.DPolygon.ellipse(bbox, self.points).to_itype(dbu))
        gap_dbu = int(self.gap_size / dbu)
        total_hole = full_circle.sized(gap_dbu)
        
        return (total_hole - metal).merge()

class ArcCoupler:
    def __init__(self, cx, cy, arc_center_x, arc_center_y, r_inner, r_outer, start_angle, stop_angle, arc_width, gap_size, coupler_2w, center_dis, safe_ext, round_radius):
        self.cx = cx
        self.cy = cy
        self.arc_cx = arc_center_x
        self.arc_cy = arc_center_y
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.start_angle = start_angle
        self.stop_angle = stop_angle
        self.arc_width = arc_width
        self.gap = gap_size
        self.c_2w = coupler_2w
        self.c_dis = center_dis
        self.safe_ext = safe_ext
        self.r_radius = round_radius
        self.points = 36

    def get_regions(self, dbu):
        # 1. 建立實體
        arc = draw_arc_capacitor(dbu, self.arc_cx, self.arc_cy, self.r_inner, self.r_outer, self.start_angle, self.stop_angle, self.arc_width)
        docking_x = self.arc_cx - (self.r_inner + self.arc_width / 2.0)
        start_x = self.cx + self.c_dis
        tbar_len = docking_x - start_x
        
        tbar_box = pya.DBox(start_x - self.safe_ext, self.cy - self.c_2w, start_x - self.safe_ext + tbar_len + self.safe_ext, self.cy + self.c_2w)
        tbar = pya.Region(pya.DPolygon(tbar_box).to_itype(dbu))
        metal_region = circle_points(dbu, self.r_radius, arc, tbar, self.points)

        # 2. 建立總佔用面積
        t_arc = draw_arc_capacitor(dbu, self.arc_cx, self.arc_cy, self.r_inner - self.gap, self.r_outer + self.gap, self.start_angle, self.stop_angle, self.arc_width + 2*self.gap)
        t_tbar_box = pya.DBox(start_x - self.gap - self.safe_ext, self.cy - self.c_2w - self.gap, start_x - self.gap - self.safe_ext + tbar_len + self.gap + self.safe_ext, self.cy + self.c_2w + self.gap)
        t_tbar = pya.Region(pya.DPolygon(t_tbar_box).to_itype(dbu))
        total_region = circle_points(dbu, self.r_radius + self.gap, t_arc, t_tbar, self.points)

        # 3. 截斷殘渣
        metal_chop = pya.Region(pya.DPolygon(pya.DBox(start_x - self.safe_ext - 10, self.cy - 500, start_x, self.cy + 500)).to_itype(dbu))
        metal_region = (metal_region - metal_chop).merge()

        total_chop = pya.Region(pya.DPolygon(pya.DBox(start_x - self.gap - self.safe_ext - 10, self.cy - 500, start_x - self.gap, self.cy + 500)).to_itype(dbu))
        total_region = (total_region - total_chop).merge()

        # 👇 就是這一行！如果這行不見了，下面就會報錯
        gap_region = (total_region - metal_region).merge()

        # ==============================================================
        # 4. 強制矩形裁減 (Clearance Box) - 解決中心 V 型凹陷
        # ==============================================================
        clear_box = pya.DBox(
            self.cx - 20,                       # 往左跨越中心線一點點
            self.cy - self.c_2w - self.gap,     # Y 軸下邊界
            start_x,                            # X 軸右邊界：貼齊金屬 T-bar 邊緣
            self.cy + self.c_2w + self.gap      # Y 軸上邊界
        )
        clear_region = pya.Region(pya.DPolygon(clear_box).to_itype(dbu))
        
        # 把這塊乾淨的矩形加進挖空區
        gap_region = (gap_region + clear_region).merge()

        return metal_region, gap_region

class TCoupler:
    def __init__(self, cx, cy, center_dis, arm_length, arm_width, head_length, head_width, gap_size, round_radius):
        self.cx = cx
        self.cy = cy
        self.c_dis = center_dis
        self.arm_len = arm_length
        self.arm_w = arm_width
        self.head_len = head_length
        self.head_w = head_width
        self.gap = gap_size
        self.r_radius = round_radius
        self.points = 64

    def get_regions(self, dbu):
        start_x = self.cx + self.c_dis
        
        # 1. 建立直角金屬本體
        head_box = pya.DBox(start_x, self.cy - self.head_len/2.0, start_x + self.head_w, self.cy + self.head_len/2.0)
        head = pya.Region(pya.DPolygon(head_box).to_itype(dbu))

        arm_start_x = start_x + self.head_w
        arm_box = pya.DBox(arm_start_x, self.cy - self.arm_w/2.0, arm_start_x + self.arm_len, self.cy + self.arm_w/2.0)
        arm = pya.Region(pya.DPolygon(arm_box).to_itype(dbu))

        metal_raw = (arm + head).merge()
        
        r_dbu = int(self.r_radius / dbu)
        gap_dbu = int(self.gap / dbu)

        if r_dbu > 0:
            metal_region = metal_raw.round_corners(r_dbu, r_dbu, self.points).merge()
            
            total_raw = metal_raw.sized(gap_dbu)
            outer_r_dbu = r_dbu + gap_dbu
            # 🌟 修改這裡：強制維持最小 r_dbu 的圓角，杜絕 GND 產生尖刺！
            inner_r_dbu = r_dbu 
            
            total_region = total_raw.round_corners(inner_r_dbu, outer_r_dbu, self.points).merge()
        else:
            metal_region = metal_raw
            total_region = metal_raw.sized(gap_dbu)

        gap_region = (total_region - metal_region).merge()

        # 3. 解決鏡像 V 型凹陷
        clear_box = pya.DBox(
            self.cx - 20,
            self.cy - self.head_len/2.0 - self.gap,
            start_x,
            self.cy + self.head_len/2.0 + self.gap
        )
        clear_region = pya.Region(pya.DPolygon(clear_box).to_itype(dbu))
        gap_region = (gap_region + clear_region).merge()

        return metal_region, gap_region


class HCoupler:
    def __init__(self, cx, cy, center_dis, arm_length, arm_width, 
                 head1_length, head1_width, head2_length, head2_width, 
                 gap_size, round_radius):
        self.cx = cx
        self.cy = cy
        self.c_dis = center_dis
        self.arm_len = arm_length
        self.arm_w = arm_width
        self.head1_len = head1_length
        self.head1_w = head1_width
        self.head2_len = head2_length
        self.head2_w = head2_width
        self.gap = gap_size
        self.r_radius = round_radius
        self.points = 64

    def get_regions(self, dbu):
        start_x = self.cx + self.c_dis
        
        # 1. 建立直角金屬本體
        head1_box = pya.DBox(start_x, self.cy - self.head1_len/2.0, 
                             start_x + self.head1_w, self.cy + self.head1_len/2.0)
        head1 = pya.Region(pya.DPolygon(head1_box).to_itype(dbu))

        arm_start_x = start_x + self.head1_w
        arm_box = pya.DBox(arm_start_x, self.cy - self.arm_w/2.0, 
                           arm_start_x + self.arm_len, self.cy + self.arm_w/2.0)
        arm = pya.Region(pya.DPolygon(arm_box).to_itype(dbu))

        head2_start_x = arm_start_x + self.arm_len
        head2_box = pya.DBox(head2_start_x, self.cy - self.head2_len/2.0, 
                             head2_start_x + self.head2_w, self.cy + self.head2_len/2.0)
        head2 = pya.Region(pya.DPolygon(head2_box).to_itype(dbu))

        metal_raw = (head1 + arm + head2).merge()
        
        r_dbu = int(self.r_radius / dbu)
        gap_dbu = int(self.gap / dbu)

        if r_dbu > 0:
            metal_region = metal_raw.round_corners(r_dbu, r_dbu, self.points).merge()
            
            total_raw = metal_raw.sized(gap_dbu)
            outer_r_dbu = r_dbu + gap_dbu
            # 🌟 修改這裡：強制維持最小 r_dbu 的圓角，杜絕 GND 產生尖刺！
            inner_r_dbu = r_dbu 
            
            total_region = total_raw.round_corners(inner_r_dbu, outer_r_dbu, self.points).merge()
        else:
            metal_region = metal_raw
            total_region = metal_raw.sized(gap_dbu)

        gap_region = (total_region - metal_region).merge()

        # 3. 解決鏡像 V 型凹陷
        clear_box = pya.DBox(
            self.cx - 20,
            self.cy - self.head1_len/2.0 - self.gap,
            start_x,
            self.cy + self.head1_len/2.0 + self.gap
        )
        clear_region = pya.Region(pya.DPolygon(clear_box).to_itype(dbu))
        gap_region = (gap_region + clear_region).merge()

        return metal_region, gap_region





class Feedline:
    def __init__(self, start_x, start_y, width, gap, l1, u2, r3, d4, r5):
        self.sx, self.sy = start_x, start_y
        self.w, self.gap = width, gap
        self.l1, self.u2, self.r3, self.d4, self.r5 = l1, u2, r3, d4, r5

    def get_regions(self, dbu, points=36):
        ext = 50
        pts = [pya.DPoint(self.sx + ext, self.sy)]
        cx, cy = self.sx + ext, self.sy
        
        moves = [(-(self.l1 + ext), 0), (0, self.u2), (self.r3, 0), (0, -self.d4), (self.r5 + ext, 0)]
        for dx, dy in moves:
            cx += dx
            cy += dy
            pts.append(pya.DPoint(cx, cy))
            
        fl_path = pya.DPath(pts, self.w)
        raw_metal = pya.Region(fl_path.polygon().to_itype(dbu))
        metal_region = round_cpw_corners(dbu, raw_metal, 30, self.w, points)

        gap_dbu = int(self.gap / dbu)
        raw_gap = metal_region.sized(gap_dbu) - metal_region
        gap_region = raw_gap.round_corners(gap_dbu, gap_dbu, points).merge()

        chop_h = self.w / 2.0 + self.gap + 20
        chop_w = ext + self.gap + 20
        end_x = self.sx - self.l1 + self.r3 + self.r5
        end_y = self.sy + self.u2 - self.d4
        
        chop = pya.Region(pya.DPolygon(pya.DBox(self.sx, self.sy - chop_h, self.sx + chop_w, self.sy + chop_h)).to_itype(dbu)) + \
               pya.Region(pya.DPolygon(pya.DBox(end_x, end_y - chop_h, end_x + chop_w, end_y + chop_h)).to_itype(dbu))

        metal_region = (metal_region - chop).merge()
        gap_region = (gap_region - chop).merge()
        
        return metal_region, gap_region


# =======================================================================
# 中央生成工廠
# =======================================================================
def build_chip_from_params(params):
    # 讀取開關設定與選配類型
    toggles = params.get("toggles", {})
    en_gnd = toggles.get("enable_gnd", True)
    en_q1 = toggles.get("enable_qubit", True)
    qubit_type = toggles.get("qubit_type", "circle")    # Qubit 形狀選配
    en_cp_global = toggles.get("enable_coupler", False) # 耦合器總開關
    coupler_type = toggles.get("coupler_type", "arc")   # 耦合器形狀選配
    en_fl = toggles.get("enable_feedline", True)

    cx = 0.0
    cy = 0.0
    
    # -------------------------------------------------------------
    # 1. 動態計算 Coupler 的總長度
    # -------------------------------------------------------------
    if coupler_type == "t_shape":
        c_dis = params["t_coupler"]["center_dis"]
        c_len = params["t_coupler"]["head_width"] + params["t_coupler"]["arm_length"]
    elif coupler_type == "h_shape":
        c_dis = params["h_coupler"]["center_dis"]
        # H-Coupler 佔用的總長度為：左棒寬 + 中橋長 + 右棒寬
        c_len = params["h_coupler"]["head1_width"] + params["h_coupler"]["arm_length"] + params["h_coupler"]["head2_width"]
    else: 
        c_dis = params["coupler"]["center_dis"]
        c_len = params["coupler"]["length"]

    # -------------------------------------------------------------
    # 2. 動態計算 Qubit 的 X 方向半寬度 (Radius 或 Length/2)
    # -------------------------------------------------------------
    if qubit_type == "rect":
        qubit_half_x = params["qubit"].get("rect_length", 200) / 2.0
    else:
        qubit_half_x = params["qubit"]["radius"]

    # -------------------------------------------------------------
    # 3. 精確推算 Qubit 中心 X 座標 (維持完美的 q_c_dis 間距)
    # -------------------------------------------------------------
    qubit_cx = cx + (c_dis + c_len + params["qubit"]["q_c_dis"] + qubit_half_x)
    
    # 準備回傳的元件字典
    components = {}

    if en_gnd:
        components["gnd"] = GroundPlane(cx, cy, params["global"]["gnd_length"], params["global"]["gnd_width"])
    
    # 根據選配類型生成 Qubit
    if en_q1:
        if qubit_type == "rect":
            components["q1"] = FloatingRectQubit(
                qubit_cx, cy, 
                length=params["qubit"].get("rect_length", 200),
                width=params["qubit"].get("rect_width", 150),
                gap_size=params["qubit"]["gap_size"], 
                slit_width=params["qubit"].get("slit_width", 10),
                cut_angle=params["qubit"].get("cut_angle", 0),
                round_radius=params["qubit"].get("round_radius", 10)
            )
        else: # circle
            components["q1"] = FloatingQubit(
                qubit_cx, cy, 
                radius=params["qubit"]["radius"], 
                gap_size=params["qubit"]["gap_size"], 
                slit_width=params["qubit"].get("slit_width", 10),
                cut_angle=params["qubit"].get("cut_angle", 0),
                points=params["qubit"]["points"]
            )
    
    # 根據選配類型生成 Coupler
    if en_cp_global:
        if coupler_type == "arc":
            components["cp"] = ArcCoupler(
                cx, cy, qubit_cx, cy, 
                r_inner=params["qubit"]["radius"] + params["qubit"]["q_c_dis"],
                r_outer=params["qubit"]["radius"] + params["qubit"]["q_c_dis"] + params["coupler"]["arc_width"],
                start_angle=params["coupler"]["arc_start_angle"],
                stop_angle=params["coupler"]["arc_stop_angle"],
                arc_width=params["coupler"]["arc_width"],
                gap_size=params["coupler"]["gap_size"], 
                coupler_2w=params["coupler"]["half_width"],
                center_dis=params["coupler"]["center_dis"],
                safe_ext=params["coupler"]["safe_ext"],
                round_radius=params["coupler"]["round_radius"]
            )
        elif coupler_type == "t_shape":
            components["t_cp"] = TCoupler(
                cx, cy, 
                center_dis=params["t_coupler"]["center_dis"],
                arm_length=params["t_coupler"]["arm_length"],
                arm_width=params["t_coupler"]["arm_width"],
                head_length=params["t_coupler"]["head_length"],
                head_width=params["t_coupler"]["head_width"],
                gap_size=params["t_coupler"]["gap_size"],
                round_radius=params["t_coupler"]["round_radius"]
            )
        elif coupler_type == "h_shape":
            # 這裡我們直接將 HCoupler 指派給 components["t_cp"]
            # 這樣主程式 gds_generator.py 會自動套用對應的 T 型渲染逻辑與鏡像圖層，不需做額外修改！
            components["t_cp"] = HCoupler(
                cx, cy, 
                center_dis=params["h_coupler"]["center_dis"],
                arm_length=params["h_coupler"]["arm_length"],
                arm_width=params["h_coupler"]["arm_width"],
                head1_length=params["h_coupler"]["head1_length"],
                head1_width=params["h_coupler"]["head1_width"],
                head2_length=params["h_coupler"]["head2_length"],
                head2_width=params["h_coupler"]["head2_width"],
                gap_size=params["h_coupler"]["gap_size"],
                round_radius=params["h_coupler"]["round_radius"]
            )

    if en_fl:
        components["fl"] = Feedline(
            -qubit_cx - 50, cy + params["feedline"]["start_shift_y"],
            params["feedline"]["width"], params["feedline"]["width"] * 0.6,
            params["feedline"]["len_l1"], params["feedline"]["len_u2"],
            params["feedline"]["len_r3"], params["feedline"]["len_d4"],
            params["feedline"]["len_r5"]
        )

    return components

