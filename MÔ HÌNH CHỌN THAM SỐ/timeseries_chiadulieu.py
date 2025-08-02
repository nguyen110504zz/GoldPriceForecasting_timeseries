#%%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from itertools import product
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima.arima import auto_arima
import warnings
warnings.filterwarnings('ignore')

#%%
df = pd.read_csv('D:/TAI LIEU/ĐỒ ÁN/NỘP FINAL/MÃ NGUỒN, DATASET/BÀI TOÁN 3/data/GoldPrice.csv')
print(df.head())

#%%
# Chuyển đổi cột Date sang kiểu datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sắp xếp dữ liệu theo thời gian
df = df.sort_values(by='Date').reset_index(drop=True)

# Kiểm tra lại
print(df.info())
print(df.head())

df_price = df['Log_Price']
diff = df['First_Diff']

#%% - Phân tích tính mùa vụ của dữ liệu
decomposition = seasonal_decompose(df_price, period=30, model='additive')

# Vẽ các thành phần
fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
decomposition.observed.plot(ax=axes[0], title='Observed')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonality')
decomposition.resid.plot(ax=axes[3], title='Residuals')
plt.tight_layout()
plt.show()

#%% - Vẽ ACF, PACF
# Chuyển dữ liệu về dạng Series để tránh lỗi ndim
data_series = df["Log_Price"]

# Vẽ lại ACF và PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_acf(data_series, ax=axes[0], lags=40)
axes[0].set_title("Autocorrelation Function (ACF)")

plot_pacf(data_series, ax=axes[1], lags=40)
axes[1].set_title("Partial Autocorrelation Function (PACF)")

plt.show()
#%%
# Chạy auto_arima chỉ với thành phần AR
stepwise_fit_AR = auto_arima(df_price,
                             start_p=0, max_p=5,
                             d=0,
                             start_q=0, max_q=0,
                             trace=True,
                             suppress_warnings=True)

print(stepwise_fit_AR.summary())
stepwise_fit_AR.plot_diagnostics(figsize=(15, 8))
plt.show()

#%% - ARMA
stepwise_fit_ARMA = auto_arima(df_price,
                               start_p=0, max_p=5,  # Tìm p tối ưu (AR)
                               d=0,                 # Không lấy sai phân
                               start_q=0, max_q=5,  # Tìm q tối ưu (MA)
                               trace=True,
                               suppress_warnings=True)

print(stepwise_fit_ARMA.summary())
stepwise_fit_ARMA.plot_diagnostics(figsize=(15, 8))
plt.show()

#%% - Xác định tham số p, d, q cho mô hình ARIMA
stepwise_fit_ARIMA = auto_arima(df_price, trace=True, suppress_warnings=True)
print(stepwise_fit_ARIMA.summary())
stepwise_fit_ARIMA.plot_diagnostics(figsize=(15, 8))
plt.show()

#%% - Mô hình SARIMA
# Chạy auto_arima cho mô hình SARIMA
stepwise_fit_SARIMA = auto_arima(df_price,
                                 start_p=0, max_p=5,  # Tìm p tối ưu (AR)
                                 d=None,              # Tự động xác định d
                                 start_q=0, max_q=5,  # Tìm q tối ưu (MA)
                                 seasonal=True,       # Bật chế độ tìm tham số thời vụ
                                 start_P=0, max_P=2,  # Tìm P tối ưu (AR theo mùa)
                                 D=None,              # Tự động xác định D (sai phân theo mùa)
                                 start_Q=0, max_Q=2,  # Tìm Q tối ưu (MA theo mùa)
                                 m=12,                # Chu kỳ mùa vụ (nếu dữ liệu theo tháng)
                                 trace=True,
                                 suppress_warnings=True)

print(stepwise_fit_SARIMA.summary())
stepwise_fit_SARIMA.plot_diagnostics(figsize=(15, 8))
plt.show()

model = ExponentialSmoothing(df_price, trend="add", seasonal="add", seasonal_periods=12)
fit = model.fit()
print(fit.summary())