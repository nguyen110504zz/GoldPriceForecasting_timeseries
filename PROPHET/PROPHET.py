#%% Nhập thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet

#%% Tải dữ liệu
file_path = "D:/TAI LIEU/ĐỒ ÁN/NỘP FINAL/MÃ NGUỒN, DATASET/BÀI TOÁN 3/data/GoldPrice.csv"
df = pd.read_csv(file_path)

#%% Plot dữ liệu
df.plot()

#%% Chuyển đổi sang dạng datetime để vẽ biểu đồ Time Series
df["Date"] = pd.to_datetime(df["Date"])

#%% Sắp xếp dữ liệu theo Date
df = df.sort_values(by="Date")

#%% Thể hiện giá vàng theo thời gian
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Price"], label="Gold Price", color="gold")

#%% Định dạng và show lên biểu đồ
plt.xlabel("Date")
plt.ylabel("Gold Price")
plt.title("Gold Price Over Time")
plt.legend()
plt.grid()
plt.show()

#%% - Xác định số lượng mẫu cho tập train (95%)
train_size = int(len(df) * 0.95)

#%% Chia tập dữ liệu thành train (95%) và test (5%)
train_df = df.iloc[:train_size]  # 95% dữ liệu đầu tiên
test_df = df.iloc[train_size:]   # 5% dữ liệu còn lại

#%% Chuẩn bị dữ liệu cho Prophet
df_train_prophet = train_df[['Date', 'Price']].rename(columns={'Date': 'ds', 'Price': 'y'})

#%% Khởi tạo và huấn luyện mô hình Prophet trên 95% dữ liệu
model = Prophet(seasonality_mode="multiplicative")
model.add_seasonality(name="yearly", period=365, fourier_order=10)  # Thêm seasonality mạnh hơn
model.fit(df_train_prophet)

#%% Dự báo dạng In-Sample Forecast chỉ trên tập train
future_in_sample = model.make_future_dataframe(periods=0)
forecast_in_sample = model.predict(future_in_sample)

#%% Vẽ biểu đồ
plt.figure(figsize=(12, 6))

#%% Vẽ dữ liệu thực tế (100%)
plt.plot(df["Date"], df["Price"], label="Actual Price", color="gold")

# Vẽ đường dự báo nhưng chỉ trên 95% dữ liệu train
plt.plot(forecast_in_sample["ds"], forecast_in_sample["yhat"], label="Predicted Price (Train)", color="blue")

# Vẽ vùng dự báo (confidence interval)
plt.fill_between(forecast_in_sample["ds"], forecast_in_sample["yhat_lower"], forecast_in_sample["yhat_upper"],
                 color="blue", alpha=0.2)

#%% Định dạng biểu đồ
plt.xlabel("Date")
plt.ylabel("Gold Price")
plt.title("In-Sample Forecast vs Actual Price (Train Data)")
plt.legend()
plt.grid()
plt.show()

#%% Chia dữ liệu thành 95% train và 5% test
train_size = int(len(df) * 0.95)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

#%% Huấn luyện Prophet trên tập train
model = Prophet()
model.fit(train_df.rename(columns={"Date": "ds", "Price": "y"}))

#%% Dự báo trên tập train (95%)
future_train = model.make_future_dataframe(periods=0)
forecast_train = model.predict(future_train)

#%% Dự báo trên tập test (5%)
future_test = test_df[["Date"]].rename(columns={"Date": "ds"})  # Chỉ dự báo đúng phần test
forecast_test = model.predict(future_test)

#%% Vẽ biểu đồ
plt.figure(figsize=(12, 6))

# Hiển thị dữ liệu thực tế
plt.plot(df["Date"], df["Price"], label="Actual Price", color="gold")

# Đường dự báo trên tập train
plt.plot(forecast_train["ds"], forecast_train["yhat"], label="Predicted Price (Train)", color="blue")
plt.fill_between(forecast_train["ds"], forecast_train["yhat_lower"], forecast_train["yhat_upper"],
                 color="blue", alpha=0.2)

# Đường dự báo trên tập test (chỉ trong 5% cuối)
plt.plot(forecast_test["ds"], forecast_test["yhat"], label="Predicted Price (Test)", color="red")
plt.fill_between(forecast_test["ds"], forecast_test["yhat_lower"], forecast_test["yhat_upper"],
                 color="red", alpha=0.2)

# Định dạng biểu đồ
plt.xlabel("Date")
plt.ylabel("Gold Price")
plt.title("Gold Price Prediction with Train and Test Forecasts")
plt.legend()
plt.grid()
plt.show()

#%% Các thành phần của mô hình dự báo
fig2 = model.plot_components(forecast_train)
plt.show()

#%% Tính các chỉ số đánh giá mô hình trên tập test
# Lấy dữ liệu thực tế từ tập test
y_true = test_df['Price'].values

# Lấy dữ liệu dự báo từ Prophet trên tập test
y_pred = forecast_test['yhat'].values

# Tính toán các chỉ số đánh giá
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

# In kết quả
print("Đánh giá mô hình Prophet trên tập test:")
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"MSE (Mean Squared Error): {mse:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")