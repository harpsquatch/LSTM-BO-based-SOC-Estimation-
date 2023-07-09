# Battery State of Charge (SOC) Estimation using LSTM and Bayesian Hyperparameter Optimization
This project focuses on estimating the state of charge (SOC) using a Stacked Long Short-Term Memory (LSTM) Neural Network. The SOC estimation is a critical aspect of battery management systems and plays a crucial role in optimizing battery performance and lifespan.

## Dataset

The dataset used in this project was obtained from [Mendeley](https://data.mendeley.com/datasets/cp3473x7xv/3). It contains battery performance data that was collected during tests conducted at McMaster University in Hamilton, Ontario, Canada. In this project 25C temperature data is being utilized.

## Data Preprocessing
The dataset was preprocessed to extract the relevant information for voltage, current, temperature, and SOC. The preprocessing script loads the dataset and selects the necessary columns. Outliers in the SOC data were detected and removed using the Local Outlier Factor (LOF) algorithm. The cleaned dataset is then ready for further analysis and model training.

## SOC Estimation
The SOC estimation is performed using LSTM neural networks with hyperparameter optimization using Bayesian methods. The LSTM model is built and trained to learn the patterns and relationships between voltage, current, and temperature variables to predict the SOC of the battery. Bayesian hyperparameter optimization is employed to find the optimal configuration for the LSTM model, resulting in improved performance.The trained model is then used to estimate the SOC of the battery. The input data is preprocessed by normalizing the features, and the LSTM model is applied to make predictions. The accuracy of the SOC predictions is evaluated using metrics such as Mean Squared Error (MSE) and R-squared.

Please refer to the code files for detailed implementation and further information on the data preprocessing, cleaning, model training, and evaluation. If you have any questions or need further assistance, please don't hesitate to reach out.
