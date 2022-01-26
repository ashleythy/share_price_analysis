## Pearson Correlation and Regression Analysis of Stock Market Prices


Python scripts to query, consolidate and analyse stock market prices using correlation and regression analysis

----

## General Info

This project does the following:

1) Query stock prices from Polygon.io - Stock Market Data
2) Runs data consolidation, analysis and visualisation on `analysis_notebook.ipynb` using functions defined in `analysis.py` 

----

## Setup

Install the required dependencies: 

`pip install -r requirements.txt`

---
## Functions

**1) import_dataset(path)**
- Specify path to folder containing queried stock market data for each stock
- Function imports all stock datasets in path directory and combines them into a single dataframe for subsequent analysis
- Function ensures that only prices of the same dates and times are used for the analysis

**2) correlate(df, price, start_ymd, end_ymd, interval)**
- Calculates the correlation scores of prices for all possible combinations of stock pairs (without replacement and where order does not matter) using dataset consolidated from the first step
    - For example, if the dataset contains prices for 3 stocks - bitcoin, eth, coin base -, 3 correlations scores will be calculated for bitcoin & eth, bitcoin & coin base, and eth & coin base respectively
- If paramater `interval` is not specified ('-'), then correlation scores will only be calculated for prices within the specified time frame - `start_ymd` and `end_ymd` 
    - Example output: 

![](/images/correlate_image1.png)

- Specifying parameter `interval` breaks down the correlation scores of prices for all stock pairs across the indicated intervals
    - For example, specifying `interval = ('m', 1)` calculates correlation scores of prices for every 1 month within `start_ymd` and `end_ymd`
    - Example output:

![](/images/correlate_image2.png)

- Function also plots the correlation scores on a line graph
    - Example plot:

![](/images/correlate_image3.png)

<u> Interpretation of Pearson's Correlation Coefficients/ Scores </u>

- It is a statistical measure of the relationship or association between 2 continuous variables (in this case, prices between 2 stocks)
- It gives us information about the direction and strength of the relationship
    - Direction:
        - Positive coefficients (+) implies that when prices of stock A increase, prices of stock B increase too
        - Negative coefficients (-) implies that when prices of stock A increase, prices of stock B decrease (or vice versa)
    - Strength:
        - The closer the coefficient is to +1 or -1, the stronger the relationship is between the variables
- Assumption: Linear relationship

**3) ols_regressor(df, price, start_ymd, end_ymd)**
- Calculates the regression coefficients that describe the relationship between each independent variable and the dependent variable
    - In our case it is the prices of different stocks within the specified time frame - `start_ymd` and `end_ymd`
- Calling this function will display all stock options that the user may choose as independent or dependent variables
    - There can only be ONE dependent variable, but there may be multiple independent variables
    - Example input:

![](/images/ols_regressor_image1.png)

- Using `statsmodel` to fit an OLS regression model allows us to produce a summary table containing the regression results
    - Example summary table

![](/images/ols_regressor_image2.png)

<u> Interpretation of Regression Coefficients </u>

- In the example summary table above, we may construct the mathematical relationship between the independent and dependent variables as such:
    - Y = -0.6957 + (0.1814 * X1)+ (0.1845 * X2) + (0.1702 * X3) + (0.1913 * X4) + (0.1954 * X5) 


- Pending!