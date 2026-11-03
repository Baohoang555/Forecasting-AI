# Báo cáo EDA — Dự đoán tiền ăn nhân viên

## 1. Tổng quan
- Số nhân viên: 1,787
- Khoảng thời gian: 2025-07-03 -> 2026-09-21
- Số dòng lưới (nhân viên x ngày): 633,406
- Tỷ lệ có ăn trên toàn bộ lưới: 57.5%
- Tổng số bữa ăn hợp lệ: 363,962
- Tổng chi tiêu ghi nhận: 16,162,710,000 đ

## 2. Tỷ lệ có ăn theo thứ trong tuần
- Thứ 2: 72.1%
- Thứ 3: 73.1%
- Thứ 4: 74.3%
- Thứ 5: 72.5%
- Thứ 6: 73.7%
- Thứ 7: 36.2%
- Chủ nhật: 0.0%

![Tỷ lệ ăn theo thứ](figures\ty_le_an_theo_thu.png)

**Diễn giải:** nếu cuối tuần thấp hẳn so với ngày thường, biến `is_weekend`/`is_workday` là đặc trưng quan trọng. Nếu có thứ cụ thể (vd thứ 2 hoặc thứ 6) khác biệt rõ so với các ngày thường khác, cân nhắc thêm đặc trưng `day_of_week` thay vì chỉ dùng cờ nhị phân weekday/weekend.

## 3. Tỷ lệ chọn món
- Việt: 86.0%
- Chay: 12.4%
- Âu: 1.6%

![Tỷ lệ món tổng thể](figures\ty_le_mon_tong_the.png)

![Tỷ lệ món theo phòng ban](figures\ty_le_mon_theo_phong_ban.png)

**Diễn giải:** tỷ lệ chọn món Âu giữa các phòng ban dao động từ 0.0% đến 100.0%. Nếu khoảng cách này lớn, `department` là đặc trưng quan trọng cho việc dự đoán N70. Nếu gần như đồng đều, thói quen cá nhân (Mã chấm công) có thể quan trọng hơn phòng ban.

## 4. Giờ ăn theo ca

![Giờ ăn theo ca](figures\gio_an_theo_ca.png)

## 5. Chi tiêu theo tháng
count    2.234100e+04
mean     7.234551e+05
std      3.022054e+05
min      0.000000e+00
25%      5.280000e+05
50%      8.360000e+05
75%      9.680000e+05
max      1.750000e+06

![Phân phối tiền ăn theo tháng](figures\phan_phoi_tien_an_thang.png)

![Tổng tiền theo tháng](figures\tong_tien_theo_thang.png)

**Diễn giải:** nếu phân phối lệch phải mạnh (nhiều nhân viên chi thấp, một số ít chi rất cao), cân nhắc dùng Huber loss hoặc biến đổi log cho biến mục tiêu ở bước mô hình hóa. Nhìn đường tổng theo tháng để phát hiện tháng bất thường (thiếu dữ liệu, nghỉ Tết...) trước khi chia train/val/test.

## 6. Nhân viên có lịch sử ngắn (tenure)
- Ngưỡng coi là 'ngắn': dưới 90 ngày hoạt động
- Số nhân viên tenure ngắn: 179 / 1,787 (10.0%)
- Tỷ lệ có ăn (nhóm tenure ngắn): 45.5%
- Tỷ lệ có ăn (nhóm còn lại): 57.6%

**Diễn giải:** nhóm tenure ngắn không đủ 3 tháng lịch sử để tạo lag features (M-1, M-2, M-3) như thiết kế ban đầu. Cần quyết định: (a) loại nhóm này khỏi tập train/test chính, đánh giá riêng bằng Group-Mean baseline, hoặc (b) giữ lại nhưng dùng giá trị NaN/0 cho lag thiếu và để mô hình (đặc biệt cây quyết định) tự xử lý missing. Ghi rõ lựa chọn này vào phần giới hạn của báo cáo.

## 7. Đối chiếu giá (QA)
- Số bữa có Thành tiền KHÁC bảng giá đã biết (44k/70k): 0 (0.00%)
