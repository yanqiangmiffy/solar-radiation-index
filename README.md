# solar-radiation-index
太阳能

## 特征工程

- 时间特征

```text
1. 因为本次比赛方是需要使用当天数据来预测当天的结果，所以不能构建时序特征
```

```text
2. 构建11,14,17的离散特征
```

- 空间特征

```
1. 与方向独立的特征：温度、湿度，气压
```

```
2. 基于方向的特征：风向，风向的频率，基于风向的平均风速
```

- 统计特征

```
日照时数（辐照度） 、平均大气压 、平均风速 、平均气温、气温日较差 、相对湿度、日总辐射量
```
## 参考文献
1. 基于BP神经网络的太阳辐射预测——以兰州市为例
2. [【kdd 2018 cup-github资料】](https://github.com/search?l=Jupyter+Notebook&q=kdd+cup+2018&type=Repositories)