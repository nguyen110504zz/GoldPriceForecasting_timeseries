#%%Import Lib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from pandas.plotting import lag_plot
from statsmodels.stats.diagnostic import acorr_ljungbox

#%%Data
df = pd.read_csv("D:/TAI LIEU/ĐỒ ÁN/NỘP FINAL/MÃ NGUỒN, DATASET/BÀI TOÁN 3/data/datagold.csv")
df.info()
#%%
# Chuyển đổi cột Date sang kiểu datetime
df['Date'] = pd.to_datetime(df['Date'])
# Sắp xếp dữ liệu theo thời gian
df = df.sort_values(by='Date').reset_index(drop=True)
#%% trực quan hóa
plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["Price"], linestyle="-", color="b")

# Chỉ hiển thị năm trên trục X
years = df["Date"].dt.year.unique()
plt.xticks(pd.to_datetime(years, format="%Y"), years)
plt.xlabel("Year")
plt.ylabel("Price")
plt.title("Gold Price Over Time")
plt.grid(True)
plt.show()
#%% Missing check
missing_values = df.isnull().sum()
print(missing_values)
#%% Duplicate check
duplicate_rows = df.duplicated().sum()
print(f"Số lượng dòng trùng lặp: {duplicate_rows}")
#%% Trans log Price
df["Log_Price"] = np.log(df["Price"])
print(df[["Date", "Price", "Log_Price"]])
#%%
plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["Log_Price"], linestyle="-", color="b")
years = df["Date"].dt.year.unique()
plt.xticks(pd.to_datetime(years, format="%Y"), years)
plt.xlabel("Year")
plt.ylabel("Log_Price")
plt.title("Gold Price Over Time")
plt.grid(True)
plt.show()
#%%Mô hình nhân
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
result = seasonal_decompose(df["Log_Price"], model="multiplicative", period=365)
plt.figure(figsize=(10, 8))
#LogPrice
plt.subplot(411)
plt.plot(df["Log_Price"], label="Log Price", color="blue")
plt.legend()
#Trend
plt.subplot(412)
plt.plot(result.trend, label="Trend", color="red")
plt.legend()
#Seasonality
plt.subplot(413)
plt.plot(result.seasonal, label="Seasonality", color="green")
plt.legend()
#Residual
plt.subplot(414)
plt.plot(result.resid, label="Residual", color="black")
plt.legend()

plt.tight_layout()
plt.show()

#%% So sánh
# Tính giá trị trung bình của 12 kỳ trước đó
df["Mean_12"] = df["Log_Price"].rolling(window=12).mean()
# Display
plt.figure(figsize=(12, 6))
plt.plot(df["Log_Price"], label="Giá đóng cửa", color="blue")
plt.plot(df["Mean_12"], label="Trung bình 12 kỳ", color="red", linestyle="dashed")
plt.xlabel("Năm")
plt.ylabel("Giá vàng")
plt.title("So sánh giá đóng cửa với giá trị trung bình 12 kỳ trước")
plt.legend()
plt.show()

#%% Def ADF KPSS
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print("ADF Test:")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"P-value: {result[1]:.4f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value:.4f}")
    if result[1] < 0.05:
        print("Kết luận: Bác bỏ H0 - Chuỗi dừng.")
    else:
        print("Kết luận: Không thể bác bỏ H0 - Chuỗi không dừng.")

def kpss_test(series):
    result = kpss(series, regression='c', nlags="auto")
    print("\nKPSS Test:")
    print(f"KPSS Statistic: {result[0]:.4f}")
    print(f"P-value: {result[1]:.4f}")
    print("Critical Values:")
    for key, value in result[3].items():
        print(f"   {key}: {value:.4f}")
    if result[1] < 0.05:
        print("Kết luận: Bác bỏ H0 - Chuỗi không dừng.")
    else:
        print("Kết luận: Không thể bác bỏ H0 - Chuỗi dừng.")
#%% Run ADF, KPSS
adf_test(df[['Log_Price']])
kpss_test(df[['Log_Price']])
#%% def tương quan tương lai quá khứ
def plot_lag_close_price(df, column="Log_Price", lag=1):
    plt.figure(figsize=(8, 6))
    lag_plot(df[column], lag=lag)
    plt.title("Lag plot of Close Price")
    plt.xlabel("y(t)")
    plt.ylabel(f"y(t + {lag})")
    plt.grid(True)
    plt.show()
plot_lag_close_price(df)
#%% Thực hiện sai phân bậc 1
def first_order_difference(df, column="Log_Price"):
    df["First_Diff"] = df[column].diff()
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["First_Diff"], label="First order difference", color="tab:blue")
    plt.title("First Order Difference")
    plt.xlabel("Date")
    plt.legend()
    plt.show()
first_order_difference(df)
#%% Fill miss check
df['Mean_12'] = df['Mean_12'].fillna(df['Mean_12'].mean())
df['First_Diff'] = df['First_Diff'].fillna(0)
# Kiểm tra lại xem còn giá trị thiếu không
print(df.isnull().sum())
#%%Kiểm định chuỗi nhiễu trắng
lags_list = [5, 10, 15, 20]
ljung_box_result = acorr_ljungbox(df["First_Diff"], lags=lags_list, return_df=True)
print("\nKết quả kiểm định Ljung-Box:\n", ljung_box_result)

p_values = ljung_box_result['lb_pvalue'].values
if np.all(p_values > 0.05):
    print("Chuỗi có thể là nhiễu trắng → Không nên dùng mô hình dự báo.")
else:
    print("Chuỗi không phải là nhiễu trắng → Có thể tiếp tục mô hình dự báo.")
#%%
df.reset_index(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df.to_csv("D:/TAI LIEU/ĐỒ ÁN/NỘP FINAL/MÃ NGUỒN, DATASET/BÀI TOÁN 3/data/GoldPrice.csv", index=False)
df.info()



