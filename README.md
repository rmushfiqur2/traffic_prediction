# Traffic prediction (eNodeB wise)

Traffic prediction of eNodeBs using clustered set. Data of eNodeBs of same cluster are trained together.

4 Prediction algorithms are compared: LSTM, ARIMA, multi-variate LSTM, Facebook Prophet

## Clustering

Clustering is based on 3 months traffic usage pattern.

![clustering img](https://github.com/rmushfiqur2/traffic_prediction/blob/main/img/clustering.jpg?raw=true)

## Prediction result

![kats](https://github.com/rmushfiqur2/traffic_prediction/blob/main/img/kats_clus_1.jpg?raw=true)

![kats](https://github.com/rmushfiqur2/traffic_prediction/blob/main/img/kats_clus_2.jpg?raw=true)

![kats](https://github.com/rmushfiqur2/traffic_prediction/blob/main/img/kats_clus_3.jpg?raw=true)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
