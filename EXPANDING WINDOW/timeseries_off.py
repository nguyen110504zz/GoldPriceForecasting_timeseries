#%% - Import thư viện
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

#%% Đọc dữ liệu từ tệp CSV
file_path = "D:/TAI LIEU/ĐỒ ÁN/NỘP FINAL/MÃ NGUỒN, DATASET/BÀI TOÁN 3/data/GoldPrice.csv"
df = pd.read_csv(file_path)

# Hiển thị thông tin tổng quan về dữ liệu
print(df.info())
print(df.head())

#%%
# Chuyển đổi cột Date sang kiểu datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sắp xếp dữ liệu theo thời gian
df = df.sort_values(by='Date').reset_index(drop=True)

# Kiểm tra lại
print(df.info())
print(df.head())

#%%
# Điền giá trị thiếu cho Mean_12 bằng trung bình
df['Mean_12'].fillna(df['Mean_12'].mean(), inplace=True)

# Điền giá trị thiếu cho First_Diff bằng 0
df['First_Diff'].fillna(0, inplace=True)

# Kiểm tra lại xem còn giá trị thiếu không
print(df.isnull().sum())

df_log = df['Log_Price']
print(df_log)

#%% - Chia dữ liệu thành 6 cửa sổ theo Expanding Window
window_sizes = [len(df_log) // 6 * (i+1) for i in range(6)]
expanding_windows = [df_log.iloc[:size] for size in window_sizes]

# Kiểm tra số lượng dòng trong mỗi cửa sổ
print("Số dòng trong mỗi cửa sổ:", window_sizes)

#%% - Chia train - test với tỷ lệ 80% - 20% trong mỗi cửa sổ
train_test_splits = []
for window in expanding_windows:
    train_size = int(len(window) * 0.8)  # 80% train
    train_set = window.iloc[:train_size]
    test_set = window.iloc[train_size:]
    train_test_splits.append((train_set, test_set))

# Kiểm tra kích thước train và test của mỗi cửa sổ
split_sizes = [(len(train), len(test)) for train, test in train_test_splits]
print("Kích thước Train-Test trong mỗi cửa sổ:", split_sizes)

for i, (train, test) in enumerate(train_test_splits):
    print(f"Cửa sổ {i+1}: Train dtype: {train.dtypes}, Test dtype: {test.dtypes}")

#%% - Mô hình AR
# Danh sách lưu kết quả mô hình của từng cửa sổ
ar_models = []
predictions_ar = []
mse_scores_ar = []
mae_scores_ar = []
rmse_scores_ar = []
lower_bounds_ar = []
upper_bounds_ar = []

# Tạo figure với 6 subplots (2 hàng, 3 cột)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()  # Biến đổi thành danh sách để dễ truy cập

# Duyệt qua từng cửa sổ train-test
for i, (train, test) in enumerate(train_test_splits):
    print(f"\n=== Huấn luyện mô hình AR trên cửa sổ {i+1} ===")

    # Huấn luyện mô hình AutoReg trên Log_Price
    model_ar = AutoReg(train, lags=1, trend='t')
    fitted_ar = model_ar.fit()

    # Dự báo trên tập test
    fc_ar_result = fitted_ar.get_prediction(start=len(train), end=len(train) + len(test) - 1)
    fc_ar = fc_ar_result.predicted_mean
    conf_ar = fc_ar_result.conf_int(alpha=0.05)  # 95% confidence interval

    # Chuyển đổi index về tập test
    fc_ar.index = test.index
    conf_ar.index = test.index

    # Lấy khoảng tin cậy
    lower_series_ar = conf_ar.iloc[:, 0]
    upper_series_ar = conf_ar.iloc[:, 1]

    # Chuyển đổi từ log-price về giá gốc
    fc_ar_exp = np.exp(fc_ar)
    test_exp = np.exp(test)
    lower_series_ar_exp = np.exp(lower_series_ar)
    upper_series_ar_exp = np.exp(upper_series_ar)

    # Lưu mô hình và dự báo
    ar_models.append(fitted_ar)
    predictions_ar.append(fc_ar_exp)

    # Tính các chỉ số đánh giá
    mse = mean_squared_error(test, fc_ar)
    mae = mean_absolute_error(test, fc_ar)
    rmse = np.sqrt(mse)

    mse_scores_ar.append(mse)
    mae_scores_ar.append(mae)
    rmse_scores_ar.append(rmse)

    print(f"  - MSE  trên cửa sổ {i+1}: {mse:.5f}")
    print(f"  - MAE  trên cửa sổ {i+1}: {mae:.5f}")
    print(f"  - RMSE trên cửa sổ {i+1}: {rmse:.5f}")

    # === Vẽ biểu đồ dự báo trên subplot ===
    ax = axes[i]  # Chọn subplot tương ứng

    # Vẽ dữ liệu training
    ax.plot(train.index, np.exp(train), color='blue', label="Training data")

    # Vẽ giá trị thực tế
    ax.plot(test.index, test_exp, color='orange', label="Actual Price")

    # Vẽ giá trị dự báo
    ax.plot(fc_ar_exp.index, fc_ar_exp, color='red', label="Predicted Price")

    # Vẽ khoảng tin cậy
    ax.fill_between(test.index, lower_series_ar_exp, upper_series_ar_exp,
                    color='lavender', alpha=0.5, label="Confidence Interval")

    # Tùy chỉnh
    ax.set_title(f"Cửa sổ {i+1}")
    ax.set_xlabel("Time")
    ax.set_ylabel(" Price")
    ax.legend()
    ax.grid(True)

# Căn chỉnh layout tránh chồng chéo
plt.tight_layout()
plt.show()

#%% - Mô hình ARMA
# Danh sách lưu kết quả mô hình
arma_models = []
predictions_arma = []
mse_scores_arma = []
mae_scores_arma = []
rmse_scores_arma = []

# Tạo figure với 6 subplots (2 hàng, 3 cột)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()  # Chuyển thành danh sách để truy cập dễ hơn

# Duyệt qua từng cửa sổ train-test
for i, (train, test) in enumerate(train_test_splits):
    print(f"\n=== Huấn luyện mô hình ARMA trên cửa sổ {i+1} ===")

    # Huấn luyện mô hình ARMA (p, d, q) = (2,0,0)
    model_arma = ARIMA(train, order=(2,0,1), trend ='t')
    fitted_arma = model_arma.fit()

    # Dự báo trên tập test
    fc_arma = fitted_arma.get_forecast(steps=len(test))
    fc_values_arma = fc_arma.predicted_mean
    conf_arma = fc_arma.conf_int(alpha=0.05)  # 95% confidence interval

    # Chuyển index về tập test
    fc_values_arma.index = test.index
    lower_series_arma = conf_arma.iloc[:, 0]
    upper_series_arma = conf_arma.iloc[:, 1]
    lower_series_arma.index = test.index
    upper_series_arma.index = test.index

    # Chuyển đổi từ log-price về giá thực tế
    fc_arma_exp = np.exp(fc_values_arma)
    test_exp = np.exp(test)
    lower_series_arma_exp = np.exp(lower_series_arma)
    upper_series_arma_exp = np.exp(upper_series_arma)

    # Lưu mô hình và dự báo
    arma_models.append(fitted_arma)
    predictions_arma.append(fc_arma_exp)

    # Tính các chỉ số đánh giá
    mse = mean_squared_error(test, fc_values_arma)
    mae = mean_absolute_error(test, fc_values_arma)
    rmse = np.sqrt(mse)

    mse_scores_arma.append(mse)
    mae_scores_arma.append(mae)
    rmse_scores_arma.append(rmse)

    print(f"  - MSE  trên cửa sổ {i+1}: {mse:.5f}")
    print(f"  - MAE  trên cửa sổ {i+1}: {mae:.5f}")
    print(f"  - RMSE trên cửa sổ {i+1}: {rmse:.5f}")

    # === Vẽ biểu đồ dự báo trên subplot ===
    ax = axes[i]  # Chọn subplot tương ứng

    # Vẽ dữ liệu training
    ax.plot(train.index, np.exp(train), color='blue', label="Training data")

    # Vẽ giá trị thực tế
    ax.plot(test.index, test_exp, color='orange', label="Actual Price")

    # Vẽ giá trị dự báo
    ax.plot(fc_arma_exp.index, fc_arma_exp, color='red', label="Predicted Price")

    # Vẽ khoảng tin cậy
    ax.fill_between(test.index, lower_series_arma_exp, upper_series_arma_exp,
                    color='lavender', alpha=0.5, label="Confidence Interval")

    # Tùy chỉnh
    ax.set_title(f"Cửa sổ {i+1}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)

# Căn chỉnh layout tránh chồng chéo
plt.tight_layout()
plt.show()

#%% - Mô hình ARIMA
# Danh sách lưu kết quả mô hình
arima_models = []
predictions_arima = []
mse_scores_arima = []
mae_scores_arima = []
rmse_scores_arima = []

# Tạo figure với 6 subplots (2 hàng, 3 cột)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()  # Chuyển thành danh sách để truy cập dễ hơn

# Duyệt qua từng cửa sổ train-test
for i, (train, test) in enumerate(train_test_splits):
    print(f"\n=== Huấn luyện mô hình ARIMA trên cửa sổ {i+1} ===")

    # Huấn luyện mô hình ARIMA (p, d, q) = (0,1,2)
    model_arima = ARIMA(train, order=(0, 1, 2), trend = 't')
    fitted_arima = model_arima.fit()

    # Dự báo trên tập test
    fc_arima = fitted_arima.get_forecast(steps=len(test))
    fc_values_arima = fc_arima.predicted_mean
    conf_arima = fc_arima.conf_int(alpha=0.05)  # 95% confidence interval

    # Chuyển index về tập test
    fc_values_arima.index = test.index
    lower_series_arima = conf_arima.iloc[:, 0]
    upper_series_arima = conf_arima.iloc[:, 1]
    lower_series_arima.index = test.index
    upper_series_arima.index = test.index

    # Chuyển đổi từ log-price về giá thực tế
    fc_arima_exp = np.exp(fc_values_arima)
    test_exp = np.exp(test)
    lower_series_arima_exp = np.exp(lower_series_arima)
    upper_series_arima_exp = np.exp(upper_series_arima)

    # Lưu mô hình và dự báo
    arima_models.append(fitted_arima)
    predictions_arima.append(fc_arima_exp)

    # Tính các chỉ số đánh giá
    mse = mean_squared_error(test, fc_values_arima)
    mae = mean_absolute_error(test, fc_values_arima)
    rmse = np.sqrt(mse)

    mse_scores_arima.append(mse)
    mae_scores_arima.append(mae)
    rmse_scores_arima.append(rmse)

    print(f"  - MSE  trên cửa sổ {i+1}: {mse:.5f}")
    print(f"  - MAE  trên cửa sổ {i+1}: {mae:.5f}")
    print(f"  - RMSE trên cửa sổ {i+1}: {rmse:.5f}")

    # === Vẽ biểu đồ dự báo trên subplot ===
    ax = axes[i]  # Chọn subplot tương ứng

    # Vẽ dữ liệu training
    ax.plot(train.index, np.exp(train), color='blue', label="Training data")

    # Vẽ giá trị thực tế
    ax.plot(test.index, test_exp, color='orange', label="Actual Price")

    # Vẽ giá trị dự báo
    ax.plot(fc_arima_exp.index, fc_arima_exp, color='red', label="Predicted Price")

    # Vẽ khoảng tin cậy
    ax.fill_between(test.index, lower_series_arima_exp, upper_series_arima_exp,
                    color='lavender', alpha=0.5, label="Confidence Interval")

    # Tùy chỉnh
    ax.set_title(f"Cửa sổ {i+1}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)

# Căn chỉnh layout tránh chồng chéo
plt.tight_layout()
plt.show()

#%% - SARIMA
#  Danh sách lưu kết quả mô hình SARIMA
sarima_models = []
predictions_sarima = []
mse_scores_sarima = []
mae_scores_sarima = []
rmse_scores_sarima = []

# Tạo figure với 6 subplots (2 hàng, 3 cột)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Duyệt qua từng cửa sổ train-test
for i, (train, test) in enumerate(train_test_splits):
    print(f"\n=== Huấn luyện mô hình SARIMA trên cửa sổ {i+1} ===")

    # Huấn luyện mô hình SARIMA với (p,d,q) = (2,1,2) và (P,D,Q,s) = (1,1,1,12)
    model_sarima = SARIMAX(train, order=(2,1,0), seasonal_order=(0,0,1,12), trend='t')
    fitted_sarima = model_sarima.fit()

    # Dự báo trên tập test
    fc_sarima = fitted_sarima.get_forecast(steps=len(test))
    fc_values_sarima = fc_sarima.predicted_mean
    conf_sarima = fc_sarima.conf_int(alpha=0.05)  # 95% confidence interval

    # Chuyển index về tập test
    fc_values_sarima.index = test.index
    lower_series_sarima = conf_sarima.iloc[:, 0]
    upper_series_sarima = conf_sarima.iloc[:, 1]
    lower_series_sarima.index = test.index
    upper_series_sarima.index = test.index

    # Chuyển đổi từ log-price về giá thực tế
    fc_sarima_exp = np.exp(fc_values_sarima)
    test_exp = np.exp(test)
    lower_series_sarima_exp = np.exp(lower_series_sarima)
    upper_series_sarima_exp = np.exp(upper_series_sarima)

    # Lưu mô hình và dự báo
    sarima_models.append(fitted_sarima)
    predictions_sarima.append(fc_sarima_exp)

    # Tính các chỉ số đánh giá
    mse = mean_squared_error(test, fc_values_sarima)
    mae = mean_absolute_error(test, fc_values_sarima)
    rmse = np.sqrt(mse)

    mse_scores_sarima.append(mse)
    mae_scores_sarima.append(mae)
    rmse_scores_sarima.append(rmse)

    print(f"  - MSE  trên cửa sổ {i+1}: {mse:.5f}")
    print(f"  - MAE  trên cửa sổ {i+1}: {mae:.5f}")
    print(f"  - RMSE trên cửa sổ {i+1}: {rmse:.5f}")

    # === Vẽ biểu đồ dự báo trên subplot ===
    ax = axes[i]

    # Vẽ dữ liệu training
    ax.plot(train.index, np.exp(train), color='blue', label="Training data")

    # Vẽ giá trị thực tế
    ax.plot(test.index, test_exp, color='orange', label="Actual Stock Price")

    # Vẽ giá trị dự báo
    ax.plot(fc_sarima_exp.index, fc_sarima_exp, color='red', label="Predicted Stock Price")

    # Vẽ khoảng tin cậy
    ax.fill_between(test.index, lower_series_sarima_exp, upper_series_sarima_exp,
                    color='lavender', alpha=0.5, label="Confidence Interval")

    # Tùy chỉnh
    ax.set_title(f"Cửa sổ {i+1}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.legend()
    ax.grid(True)

# Căn chỉnh layout tránh chồng chéo
plt.tight_layout()
plt.show()

#%% - Holt-Winters
#  Danh sách lưu kết quả mô hình
hw_models = []
predictions_hw = []
mse_scores_hw = []
mae_scores_hw = []
rmse_scores_hw = []

# Tạo figure với 6 subplots (2 hàng, 3 cột)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Duyệt qua từng cửa sổ train-test
for i, (train, test) in enumerate(train_test_splits):
    print(f"\n=== Huấn luyện mô hình Holt-Winters trên cửa sổ {i+1} ===")

    # Huấn luyện mô hình Holt-Winters
    model_hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12)
    fitted_hw = model_hw.fit(smoothing_level=0.8, smoothing_slope=0.2, smoothing_seasonal=0.2)

    # Dự báo trên tập test
    fc_hw = fitted_hw.forecast(steps=len(test))

    # Chuyển đổi từ log-price về giá thực tế
    fc_hw_exp = np.exp(fc_hw)
    test_exp = np.exp(test)

    # Tính sai số dự báo
    forecast_error = test - fc_hw
    std_dev = np.std(forecast_error)  # Độ lệch chuẩn của sai số
    confidence_interval_log = 1.96 * std_dev  # Khoảng tin cậy 95%

    # Chuyển khoảng tin cậy sang giá trị thực tế
    lower_bound = np.exp(fc_hw - confidence_interval_log)
    upper_bound = np.exp(fc_hw + confidence_interval_log)

    # Lưu mô hình và dự báo
    hw_models.append(fitted_hw)
    predictions_hw.append(fc_hw_exp)

    # Tính các chỉ số đánh giá
    mse = mean_squared_error(test, fc_hw)
    mae = mean_absolute_error(test, fc_hw)
    rmse = np.sqrt(mse)

    mse_scores_hw.append(mse)
    mae_scores_hw.append(mae)
    rmse_scores_hw.append(rmse)

    print(f"  - MSE  trên cửa sổ {i+1}: {mse:.5f}")
    print(f"  - MAE  trên cửa sổ {i+1}: {mae:.5f}")
    print(f"  - RMSE trên cửa sổ {i+1}: {rmse:.5f}")

    # === Vẽ biểu đồ dự báo trên subplot ===
    ax = axes[i]

    # Vẽ dữ liệu training
    ax.plot(train.index, np.exp(train), color='blue', label="Training data")

    # Vẽ giá trị thực tế
    ax.plot(test.index, test_exp, color='orange', label="Actual Price")

    # Vẽ giá trị dự báo
    ax.plot(fc_hw_exp.index, fc_hw_exp, color='red', label="Predicted Price")

    # Vẽ khoảng tin cậy 95%
    ax.fill_between(fc_hw_exp.index,
                    lower_bound,
                    upper_bound,
                    color='lavender', alpha=0.5, label="Confidence Interval")

    # Tùy chỉnh
    ax.set_title(f"Cửa sổ {i+1}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)

# Căn chỉnh layout tránh chồng chéo
plt.tight_layout()
plt.show()

# ====== So sánh với Baseline cho mỗi mô hình ======
#%% AR Model
print("\n=== So sánh AR Model với Baseline ===")
baseline_predictions_ar = []
baseline_rmse_ar = []

for i, (train, test) in enumerate(train_test_splits):
    baseline_pred = np.full_like(test, train.mean())
    baseline_rmse = np.sqrt(mean_squared_error(test, baseline_pred))
    baseline_predictions_ar.append(baseline_pred)
    baseline_rmse_ar.append(baseline_rmse)
    print(f"Cửa sổ {i+1}:")
    print(f"  AR Model RMSE: {rmse_scores_ar[i]:.5f}")
    print(f"  Baseline RMSE: {baseline_rmse:.5f}")

# Vẽ biểu đồ so sánh
plt.figure(figsize=(16, 10))
x = np.arange(len(train_test_splits))
width = 0.35
plt.bar(x - width/2, rmse_scores_ar, width, label='AR Model', color='blue')
plt.bar(x + width/2, baseline_rmse_ar, width, label='Baseline', color='green')
plt.xlabel('Cửa sổ')
plt.ylabel('RMSE')
plt.title('So sánh RMSE giữa AR Model và Baseline')
plt.xticks(x, [f'Cửa sổ {i+1}' for i in range(len(train_test_splits))])
plt.legend()
plt.grid(True)
plt.show()

#%% ARMA
print("\n=== So sánh ARMA Model với Baseline ===")
baseline_predictions_arma = []
baseline_rmse_arma = []

for i, (train, test) in enumerate(train_test_splits):
    baseline_pred = np.full_like(test, train.mean())
    baseline_rmse = np.sqrt(mean_squared_error(test, baseline_pred))
    baseline_predictions_arma.append(baseline_pred)
    baseline_rmse_arma.append(baseline_rmse)
    print(f"Cửa sổ {i+1}:")
    print(f"  ARMA Model RMSE: {rmse_scores_arma[i]:.5f}")
    print(f"  Baseline RMSE: {baseline_rmse:.5f}")

# Vẽ biểu đồ so sánh
plt.figure(figsize=(16, 10))
x = np.arange(len(train_test_splits))
width = 0.35
plt.bar(x - width/2, rmse_scores_arma, width, label='ARMA Model', color='blue')
plt.bar(x + width/2, baseline_rmse_arma, width, label='Baseline', color='green')
plt.xlabel('Cửa sổ')
plt.ylabel('RMSE')
plt.title('So sánh RMSE giữa ARMA Model và Baseline')
plt.xticks(x, [f'Cửa sổ {i+1}' for i in range(len(train_test_splits))])
plt.legend()
plt.grid(True)
plt.show()

#%% ARIMA
print("\n=== So sánh ARIMA Model với Baseline ===")
baseline_predictions_arima = []
baseline_rmse_arima = []

for i, (train, test) in enumerate(train_test_splits):
    baseline_pred = np.full_like(test, train.mean())
    baseline_rmse = np.sqrt(mean_squared_error(test, baseline_pred))
    baseline_predictions_arima.append(baseline_pred)
    baseline_rmse_arima.append(baseline_rmse)
    print(f"Cửa sổ {i+1}:")
    print(f"  ARIMA Model RMSE: {rmse_scores_arima[i]:.5f}")
    print(f"  Baseline RMSE: {baseline_rmse:.5f}")

# Vẽ biểu đồ so sánh
plt.figure(figsize=(16, 10))
x = np.arange(len(train_test_splits))
width = 0.35
plt.bar(x - width/2, rmse_scores_arima, width, label='ARIMA Model', color='blue')
plt.bar(x + width/2, baseline_rmse_arima, width, label='Baseline', color='green')
plt.xlabel('Cửa sổ')
plt.ylabel('RMSE')
plt.title('So sánh RMSE giữa ARIMA Model và Baseline')
plt.xticks(x, [f'Cửa sổ {i+1}' for i in range(len(train_test_splits))])
plt.legend()
plt.grid(True)
plt.show()

# SARIMA
print("\n=== So sánh SARIMA Model với Baseline ===")
baseline_predictions_sarima = []
baseline_rmse_sarima = []

for i, (train, test) in enumerate(train_test_splits):
    baseline_pred = np.full_like(test, train.mean())
    baseline_rmse = np.sqrt(mean_squared_error(test, baseline_pred))
    baseline_predictions_sarima.append(baseline_pred)
    baseline_rmse_sarima.append(baseline_rmse)
    print(f"Cửa sổ {i+1}:")
    print(f"  SARIMA Model RMSE: {rmse_scores_sarima[i]:.5f}")
    print(f"  Baseline RMSE: {baseline_rmse:.5f}")

# Vẽ biểu đồ so sánh
plt.figure(figsize=(16, 10))
x = np.arange(len(train_test_splits))
width = 0.35
plt.bar(x - width/2, rmse_scores_sarima, width, label='SARIMA Model', color='blue')
plt.bar(x + width/2, baseline_rmse_sarima, width, label='Baseline', color='green')
plt.xlabel('Cửa sổ')
plt.ylabel('RMSE')
plt.title('So sánh RMSE giữa SARIMA Model và Baseline')
plt.xticks(x, [f'Cửa sổ {i+1}' for i in range(len(train_test_splits))])
plt.legend()
plt.grid(True)
plt.show()

#%% Holt-Winters
print("\n=== So sánh Holt-Winters Model với Baseline ===")
baseline_predictions_hw = []
baseline_rmse_hw = []

for i, (train, test) in enumerate(train_test_splits):
    baseline_pred = np.full_like(test, train.mean())
    baseline_rmse = np.sqrt(mean_squared_error(test, baseline_pred))
    baseline_predictions_hw.append(baseline_pred)
    baseline_rmse_hw.append(baseline_rmse)
    print(f"Cửa sổ {i+1}:")
    print(f"  Holt-Winters Model RMSE: {rmse_scores_hw[i]:.5f}")
    print(f"  Baseline RMSE: {baseline_rmse:.5f}")

# Vẽ biểu đồ so sánh
plt.figure(figsize=(16, 10))
x = np.arange(len(train_test_splits))
width = 0.35
plt.bar(x - width/2, rmse_scores_hw, width, label='Holt-Winters Model', color='blue')
plt.bar(x + width/2, baseline_rmse_hw, width, label='Baseline', color='green')
plt.xlabel('Cửa sổ')
plt.ylabel('RMSE')
plt.title('So sánh RMSE giữa Holt-Winters Model và Baseline')
plt.xticks(x, [f'Cửa sổ {i+1}' for i in range(len(train_test_splits))])
plt.legend()
plt.grid(True)
plt.show()

#%%
# Số lượng cửa sổ (splits)
x = np.arange(len(train_test_splits))
width = 0.15  # Độ rộng của mỗi cột

plt.figure(figsize=(18, 10))

# Vẽ biểu đồ cột (bar chart) cho từng mô hình
plt.bar(x - 2*width, rmse_scores_ar, width, label='AR Model', color='blue')
plt.bar(x - width, rmse_scores_arma, width, label='ARMA Model', color='red')
plt.bar(x, rmse_scores_arima, width, label='ARIMA Model', color='purple')
plt.bar(x + width, rmse_scores_sarima, width, label='SARIMA Model', color='orange')
plt.bar(x + 2*width, rmse_scores_hw, width, label='Holt-Winters Model', color='cyan')

# Vẽ baseline (màu xanh lá)
plt.plot(x, baseline_rmse_ar, marker='o', linestyle='dashed', color='green', label='Baseline')

# Định dạng biểu đồ
plt.xlabel('Cửa sổ')
plt.ylabel('RMSE')
plt.title('So sánh RMSE giữa 5 mô hình và Baseline')
plt.xticks(x, [f'Cửa sổ {i+1}' for i in range(len(train_test_splits))])
plt.legend()
# plt.grid(True)

# Hiển thị biểu đồ
plt.show()