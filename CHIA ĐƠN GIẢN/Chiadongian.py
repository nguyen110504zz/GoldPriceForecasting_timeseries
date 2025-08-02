import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# %% Đọc dữ liệu
file_path = "D:/TAI LIEU/ĐỒ ÁN/NỘP FINAL/MÃ NGUỒN, DATASET/BÀI TOÁN 3/data/GoldPrice.csv"
df = pd.read_csv(file_path)

# %% Chuyển cột 'Date' thành kiểu datetime
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# %% Chia dữ liệu theo tuần
df_weekly = df.resample('W').mean()

# %% Chia dữ liệu thành tập train và validation
split_ratio = 0.8
split_index = int(len(df_weekly) * split_ratio)
train_data = df_weekly.iloc[:split_index]
validation_data = df_weekly.iloc[split_index:]
forecast_steps = len(validation_data)


# %% Hàm huấn luyện và dự báo mô hình
def train_and_forecast(model_order, model_name, seasonal_order=None):
    if model_name == "SARIMA":
        model = SARIMAX(train_data['Log_Price'], order=model_order, seasonal_order=seasonal_order)
    elif model_name == "Holt-Winters":
        model = ExponentialSmoothing(train_data['Log_Price'], trend='add', seasonal=None)
    else:  # AR, ARMA, ARIMA
        model = sm.tsa.ARIMA(train_data['Log_Price'], order=model_order, trend='t')

    model_fit = model.fit()
    forecast_log = model_fit.forecast(steps=forecast_steps)
    forecast = np.exp(forecast_log)  # Chuyển đổi ngược về giá thực tế

    # Tính MAE và RMSE theo Log_Price
    mae_log = mean_absolute_error(validation_data['Log_Price'], forecast_log)
    rmse_log = np.sqrt(mean_squared_error(validation_data['Log_Price'], forecast_log))

    print(f"Summary for {model_name}:")
    print(model_fit.summary())

    return forecast, mae_log, rmse_log, model_fit


# %% Huấn luyện và dự báo với các mô hình
models = {
    "AR": (1, 0, 0),
    "ARMA": (2, 0, 1),
    "ARIMA": (0, 1, 2),
}

results = {}
for model_name, order in models.items():
    forecast, mae_log, rmse_log, model_fit = train_and_forecast(order, model_name)
    results[model_name] = {"forecast": forecast, "MAE_Log": mae_log, "RMSE_Log": rmse_log,
                           "summary": model_fit.summary()}

# Chạy riêng SARIMA với seasonal_order
forecast_sarima, mae_sarima, rmse_sarima, model_fit_sarima = train_and_forecast((2, 1, 0), "SARIMA", (0, 0, 1, 12))
results["SARIMA"] = {"forecast": forecast_sarima, "MAE_Log": mae_sarima, "RMSE_Log": rmse_sarima,
                     "summary": model_fit_sarima.summary()}

# Holt-Winters
forecast_hw, mae_hw, rmse_hw, model_fit_hw = train_and_forecast(None, "Holt-Winters")
results["Holt-Winters"] = {"forecast": forecast_hw, "MAE_Log": mae_hw, "RMSE_Log": rmse_hw,
                           "summary": model_fit_hw.summary()}

# %% Xuất kết quả MAE và RMSE của từng mô hình
print("\nKẾT QUẢ ĐÁNH GIÁ CÁC MÔ HÌNH (Đơn vị Log_Price)")
for model_name, result in results.items():
    print(f"\n {model_name}:")
    print(f"   - MAE (Log_Price)  = {result['MAE_Log']:.4f}")
    print(f"   - RMSE (Log_Price) = {result['RMSE_Log']:.4f}")

# %% Vẽ biểu đồ riêng cho từng mô hình
for model_name, result in results.items():
    plt.figure(figsize=(10, 5))
    plt.plot(train_data.index, np.exp(train_data['Log_Price']), label="Train data", color="orange")
    plt.plot(validation_data.index, np.exp(validation_data['Log_Price']), label="Validation data", color="green")
    plt.plot(validation_data.index, result["forecast"], label="Prediction data", color="red", linestyle="dashed")
    plt.fill_between(validation_data.index, np.exp(validation_data['Log_Price']), result["forecast"], color='blue',
                     alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel("Gold Price")
    plt.title(f"Gold Price Prediction with {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

# %% Tính baseline
baseline_pred = np.full_like(validation_data['Log_Price'], fill_value=np.exp(train_data['Log_Price'].mean()))
baseline_rmse_log = np.sqrt(mean_squared_error(validation_data['Log_Price'], np.log(baseline_pred)))

# %% Vẽ biểu đồ cột so sánh baseline và RMSE của từng mô hình
model_names = list(results.keys()) + ["Baseline"]
rmse_values_log = [results[m]["RMSE_Log"] for m in results.keys()] + [baseline_rmse_log]

plt.figure(figsize=(10, 6))
plt.bar(model_names, rmse_values_log, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel("Models")
plt.ylabel("RMSE (Log_Price)")
plt.title("Comparison of RMSE (Log_Price) between Models and Baseline")
plt.xticks(rotation=45)
plt.show()
