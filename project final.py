import numpy as np
import pandas as pd
import copy
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from operator import itemgetter
import matplotlib.pyplot as plt

RISK_FREE_RATE = 0.025
CONFIDENCE_LEVEL = 0.95
SLIPPAGE = 0.00025
TRADING_COST = 0.0002
N = [5,6,7,8,9]
R = [2,3,4]
PA = [(x, y) for x in N for y in R]

class PortfolioConstruction:
    
    def __init__(self,stock_data):
        self.stock_data = stock_data
        self.stock_returns = pd.DataFrame()
        self.portfolio_periodic_returns = pd.DataFrame()
    
    def returns_calc(self):
        stock_data_df = self.stock_data.copy()
        for col in stock_data_df.columns[0:]:
            self.stock_returns[col] = (stock_data_df[col].pct_change())
        return self.stock_returns.dropna()
    
    def portfolio(self,n,r):
        ret_df = self.stock_returns.copy()
        portfolio = []
        periodic_return = []
        
        for i in range(len(ret_df)):
            if len(portfolio) > 0:
                periodic_return.append(ret_df[portfolio].iloc[i, :].mean())
                bad_stocks = ret_df[portfolio].iloc[i, :].sort_values(ascending=True)[:r].index.values.tolist()
                portfolio = [stock for stock in portfolio if stock not in bad_stocks]
            
            number_of_replacements = n - len(portfolio)
            new_add = ret_df.iloc[i, :].sort_values(ascending=False)[:number_of_replacements].index.values.tolist()
            portfolio = portfolio + new_add
            
        periodic_return_adjusted = [adj - ((2 * r) + (SLIPPAGE + TRADING_COST)) for adj in periodic_return]
        
        self.portfolio_periodic_returns["Period"] = np.arange(1, len(ret_df))
        self.portfolio_periodic_returns["Period Returns"] = np.array(periodic_return_adjusted)
        self.portfolio_periodic_returns.dropna(inplace=True)
        self.portfolio_periodic_returns.set_index("Period",inplace = True)

        return self.portfolio_periodic_returns

    def market_portfolio_plot(self, data, title):
        x, y = self.portfolio_periodic_returns.index, self.portfolio_periodic_returns['Period Returns']

        plt.plot(x, (y * 100), c='b', label='Rebalancing Returns')
        plt.plot(data.index, (data["Market Returns"] * 100), c='m', label='Market Returns')
        plt.plot(x, np.zeros(len(self.portfolio_periodic_returns)), linestyle = "--", alpha = 0.8)
        plt.xlabel("Period")
        plt.ylabel("Percent Returns")
        plt.title(title)
        plt.legend(loc='upper left')
        plt.show()

    def returns_histogram(self, data):
        plt.hist(data['Market Returns'], bins=15)
        plt.xlabel("Percent Returns")
        plt.ylabel("Count")
        plt.title("Histogram of Returns")
        plt.show()

class KPI:
    
    def __init__(self,portfolio_returns,market_returns,interval):
        assert interval == "1mo" or interval == "3mo", "Intervals can only be 1mo(for monthly) or 3mo(for trimonthly)"
        self.portfolio_returns = portfolio_returns
        self.market_returns = market_returns
        if interval == "1mo":
            self.interval = 12
        elif interval == "3mo":
            self.interval = 4
        
    def cagr(self):
        portfolio_returns_df = self.portfolio_returns.copy()
        portfolio_returns_df["Cumulative Return"] = (1 + portfolio_returns_df["Period Returns"]).cumprod()
        t = len(portfolio_returns_df)/self.interval
        cagr = (portfolio_returns_df["Cumulative Return"].iloc[-1] ** (1/t)) - 1
        return cagr
    
    def market_cagr(self):
        portfolio_returns_df = self.market_returns.copy()
        portfolio_returns_df["Cumulative Return"] = (1 + portfolio_returns_df["Market Returns"]).cumprod()
        t = len(portfolio_returns_df)/self.interval
        cagr = (portfolio_returns_df["Cumulative Return"].iloc[-1] ** (1/t)) - 1
        return cagr
      
    def volatility(self):
        portfolio_returns_df = self.portfolio_returns.copy()
        vol = portfolio_returns_df.loc[:,"Period Returns"].std()
        return vol * np.sqrt(self.interval)
    
    def market_vol(self):
        portfolio_returns_df = self.market_returns.copy()
        vol = portfolio_returns_df.loc[:,"Market Returns"].std()
        return vol * np.sqrt(self.interval)
    
    def sharpe_ratio(self):
        risk_premium = KPI.cagr(self) - RISK_FREE_RATE
        return risk_premium/KPI.volatility(self)
    
    def sortino_ratio(self):
        portfolio_returns_df = self.portfolio_returns.copy()
        risk_premium = KPI.cagr(self) - RISK_FREE_RATE
        negative_returns = portfolio_returns_df[portfolio_returns_df["Period Returns"] < 0]
        downward_volatility = negative_returns["Period Returns"].std() * np.sqrt(self.interval)
        return risk_premium / downward_volatility
    
    def var(self):
        portfolio_mean = np.mean(self.portfolio_returns['Period Returns'])
        portfolio_stdeviation = np.std(self.portfolio_returns['Period Returns'])
        alpha = norm.ppf(1 - CONFIDENCE_LEVEL)
        var = (portfolio_mean - portfolio_stdeviation) * alpha 
        return var
    
    def maximum_drawdown(self):
        port_return_df = self.portfolio_returns.copy()
        port_return_df["Cumulative Return"] = (1 + port_return_df["Period Returns"]).cumprod()
        port_return_df["Rolling Max Cumulative"] = port_return_df["Cumulative Return"].cummax()
        port_return_df["Drawdown"] = port_return_df["Rolling Max Cumulative"] - port_return_df["Cumulative Return"]
        port_return_df["Drawdown Percentage"] = port_return_df["Drawdown"]/port_return_df["Rolling Max Cumulative"]
        return port_return_df["Drawdown Percentage"].max()
    
    def calamar_ratio(self):
        return KPI.cagr(self) / KPI.maximum_drawdown(self)
    
    def capm(self):
        y = np.array(self.portfolio_returns["Period Returns"] - (RISK_FREE_RATE / self.interval))
        x = np.array((self.market_returns["Market Returns"] - (RISK_FREE_RATE / self.interval))).reshape(-1,1)
        capm = LinearRegression().fit(x, y)
        beta = capm.coef_[0]
        alpha = capm.intercept_
        return beta,(alpha * self.interval)
       
    def beta(self):
        return KPI.capm(self)[0]
    
    def alpha(self):
        return KPI.capm(self)[1]
    
    def capm_expected_return(self):
        risk_prem =  (self.market_returns["Market Returns"].mean() * self.interval) - (RISK_FREE_RATE / self.interval)
        expected_return = (RISK_FREE_RATE / self.interval) + (KPI.beta(self) * risk_prem)
        return expected_return

class PortfolioSelection:
    
    def __init__(self,stock_data,market_returns,interval):
        assert interval == "1mo" or interval == "3mo", "Intervals can only be 1mo(for monthly) or 3mo(for trimonthly)"
        self.stock_data = stock_data
        self.market_returns = market_returns
        self.interval = interval
        
    def max_optimizer(self,dict):
        return max(dict.items(), key=itemgetter(1))
    
    def min_optimizer(self,dict):
        return min(dict.items(), key=itemgetter(1))

    def optimal_kpi_combinations_calculator(self):
        
        cagr = {}
        volatility = {}
        sharpe = {}
        sortino = {}
        var = {}
        alpha = {}
        max_drawdown = {}
        calamar_ratio = {}
        results = pd.DataFrame()
        
        pfolio = PortfolioConstruction(self.stock_data)
        pfolio.returns_calc()
        
        for combinations in PA:
            
            pfolio_returns = pfolio.portfolio(combinations[0],combinations[1])
            pfolio_kpi = KPI(pfolio_returns,self.market_returns,self.interval)
            
            cagr[combinations] = pfolio_kpi.cagr().round(3)
            volatility[combinations] = pfolio_kpi.volatility().round(3)
            sharpe[combinations] = pfolio_kpi.sharpe_ratio().round(3)
            sortino[combinations] = pfolio_kpi.sortino_ratio().round(3)
            var[combinations] = pfolio_kpi.var().round(3)
            alpha[combinations] = pfolio_kpi.alpha().round(3)
            max_drawdown[combinations] = pfolio_kpi.maximum_drawdown()
            calamar_ratio[combinations] = pfolio_kpi.calamar_ratio()
            
        results['KPI'] = (
        'Max CAGR', 'Min Volatility', 'Max Sharpe Ratio', ' Max Sortino Ratio', 'Min VAR', 'Max Alpha',
        'Minimum Max Drawdown', 'Max Calamr Ratio')

        results['Combinations'] = (
        PortfolioSelection.max_optimizer(self, cagr)[0], PortfolioSelection.min_optimizer(self, volatility)[0],
        PortfolioSelection.max_optimizer(self, sharpe)[0], PortfolioSelection.max_optimizer(self, sortino)[0],
        PortfolioSelection.min_optimizer(self, var)[0], PortfolioSelection.max_optimizer(self, alpha)[0],
        PortfolioSelection.min_optimizer(self, max_drawdown)[0],
        PortfolioSelection.max_optimizer(self, calamar_ratio)[0])

        results['Value'] = (
        PortfolioSelection.max_optimizer(self, cagr)[1], PortfolioSelection.min_optimizer(self, volatility)[1],
        PortfolioSelection.max_optimizer(self, sharpe)[1], PortfolioSelection.max_optimizer(self, sortino)[1],
        PortfolioSelection.min_optimizer(self, var)[1], PortfolioSelection.max_optimizer(self, alpha)[1],
        PortfolioSelection.min_optimizer(self, max_drawdown)[1],
        PortfolioSelection.max_optimizer(self, calamar_ratio)[1])
        return results

class ComparativeVisualization:

    def __init__(self, portfolio_monthly_KPI_results, portfolio_trimonthly_KPI_results, one_mo_combo, three_mo_combo):
        self.mo_rebalance = portfolio_monthly_KPI_results
        self.three_mo_rebalnce = portfolio_trimonthly_KPI_results
        self.one_mo_combo = one_mo_combo
        self.three_mo_combo = three_mo_combo


    def rebalancing_comparision_visualization(self):
        barwidth = 0.4
        xaxis = ['CAGR', 'Vol.', 'Sharpe', 'Sort.', 'VAR', 'Alpha', 'Beta', 'CAPM', 'Draw', 'Calamar']
        r1 = np.arange(len(self.mo_rebalance['KPI']))
        r2 = [x + barwidth for x in r1]

        plt.bar(r1, self.mo_rebalance['Values'], label="1 Month Rebalancing", fill="b", width=barwidth)
        plt.bar(r2, self.three_mo_rebalnce['Values'], label='3 Month Rebalancing', fill='m', width=barwidth)
        plt.title("KPI Comparisons for Optimal Portfolios ")
        plt.xlabel("KPI")
        plt.xticks([r + barwidth for r in range(len(self.mo_rebalance["KPI"]))], xaxis)
        plt.ylabel("Values")
        plt.autoscale(tight=True)
        plt.legend(loc='upper left')
        plt.show()

    def KPI_comparision_visualization(self):
        barwidth = 0.4
        r1 = np.arange(len(self.one_mo_combo['KPI']))
        r2 = [x + barwidth for x in r1]
        xaxis = ['CAGR', 'Vol.', 'Sharpe', 'Sort.', 'VAR', 'Alpha', 'Draw', 'Calamar']

        plt.bar(r1, self.one_mo_combo['Value'], label="1 Month Rebalancing", fill="b", width=barwidth)
        plt.bar(r2, self.three_mo_combo['Value'], label='3 Month Rebalancing', fill='m', width=barwidth)
        plt.title("Best Possible KPIs based on Portfolio Rebalancing ")
        plt.xlabel("KPI")
        plt.xticks([r + barwidth for r in range(len(self.one_mo_combo["KPI"]))], xaxis)
        plt.ylabel("Values")
        plt.autoscale(tight=True)
        plt.legend(loc='upper left')
        plt.show()

if __name__ == '__main__':

    # importing monthly adjusted closing price of DJI index
    dow_jones_monthly = pd.read_csv('/Users/colecrescas/downloads/DJI_monthly.csv')
    market_returns_monthly = dow_jones_monthly.copy()
    market_returns_monthly["Market Returns"] = market_returns_monthly['^DJI'].pct_change()
    market_returns_monthly = market_returns_monthly.drop(columns = '^DJI').dropna()

    # importing monthly adjusted closing price of constituent stocks in DJI index
    stock_data_monthly = pd.read_csv('/Users/colecrescas/downloads/dow_stock_monthly.csv', index_col=0)
    stock_data_monthly = stock_data_monthly.set_index('Date').astype(float).round(2)

    # importing quarterly adjusted closing price of DJI index
    dow_jones_quarterly = pd.read_csv('/Users/colecrescas/downloads/DJI_trimonthly.csv')
    market_returns_quarterly = dow_jones_quarterly.copy()
    market_returns_quarterly["Market Returns"] = market_returns_quarterly['^DJI'].pct_change()
    market_returns_quarterly = market_returns_quarterly.drop(columns = '^DJI').dropna()

    # importing quarterly adjusted closing price of constituent stocks in DJI index
    stock_data_quarterly = pd.read_csv('/Users/colecrescas/downloads/dow_stock_trimonthly.csv', index_col=0)
    stock_data_quarterly = stock_data_quarterly.set_index('Date').astype(float).round(2)

    # finding optimal (N,R) combinations for various KPIs for monthly portfolio rebalancing
    portfolio_selection_monthly = PortfolioSelection(stock_data_monthly,market_returns_monthly,"1mo")
    one_mo_combo = portfolio_selection_monthly.optimal_kpi_combinations_calculator()
    print(one_mo_combo)

    # finding optimal (N,R) combinations for various KPIs for quarterly portfolio rebalancing
    portfolio_selection_quarterly = PortfolioSelection(stock_data_quarterly,market_returns_quarterly,"3mo")
    three_mo_combo = portfolio_selection_quarterly.optimal_kpi_combinations_calculator()
    three_mo_combo

    # optimal (N,R) combination for monthly rebalancing: 9,3
    # calculating portfolio returns for 9,3 monthly rebalancing
    portfolio_monthly = PortfolioConstruction(stock_data_monthly)
    portfolio_monthly.returns_calc()
    portfolio_returns_monthly = portfolio_monthly.portfolio(9, 3)
    print(portfolio_returns_monthly)

    # checking returns for normality
    portfolio_monthly.returns_histogram(market_returns_monthly)

    # plotting portfolio returns against market returns
    portfolio_monthly.market_portfolio_plot(market_returns_monthly,"Monthly Returns, Combinations: (9, 3)")

    # calculating KPIs for 9,3 monthly rebalancing
    portfolio_monthly_KPI = KPI(portfolio_returns_monthly,market_returns_monthly,"1mo")
    portfolio_monthly_KPI_results = pd.DataFrame()
    portfolio_monthly_KPI_results['KPI'] = (
        'CAGR', 'Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Value at Risk', 'Alpha', 'Beta',
        'CAPM Expected Return', 'Max Drawdown', 'Calamr Ratio')
    portfolio_monthly_KPI_results['Values'] = (
        portfolio_monthly_KPI.cagr(), portfolio_monthly_KPI.volatility(), portfolio_monthly_KPI.sharpe_ratio(),
        portfolio_monthly_KPI.sortino_ratio(), portfolio_monthly_KPI.var(), portfolio_monthly_KPI.alpha(),
        portfolio_monthly_KPI.beta(), portfolio_monthly_KPI.capm_expected_return(),
        portfolio_monthly_KPI.maximum_drawdown(), portfolio_monthly_KPI.calamar_ratio())
    print(portfolio_monthly_KPI_results)

    # optimal (N,R) combination for monthly rebalancing: 5,2
    # calculating portfolio returns for 5,2 quarterly rebalancing
    portfolio_quarterly = PortfolioConstruction(stock_data_quarterly)
    portfolio_quarterly.returns_calc()
    portfolio_returns_quarterly = portfolio_quarterly.portfolio(5,2)
    print(portfolio_returns_quarterly)

    # plotting portfolio returns against market returns
    portfolio_quarterly.market_portfolio_plot(market_returns_quarterly, "Quarterly Returns, Combinations: (5, 2)")

    portfolio_quarterly_KPI = KPI(portfolio_returns_quarterly,market_returns_quarterly,"3mo")
    portfolio_quarterly_KPI_results = pd.DataFrame()
    portfolio_quarterly_KPI_results['KPI'] = (
        'CAGR', 'Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Value at Risk', 'Alpha', 'Beta',
        'CAPM Expected Return', 'Max Drawdown', 'Calamr Ratio')
    portfolio_quarterly_KPI_results['Values'] = (
        portfolio_quarterly_KPI.cagr(), portfolio_quarterly_KPI.volatility(), portfolio_quarterly_KPI.sharpe_ratio(),
        portfolio_quarterly_KPI.sortino_ratio(), portfolio_quarterly_KPI.var(), portfolio_quarterly_KPI.alpha(),
        portfolio_quarterly_KPI.beta(), portfolio_quarterly_KPI.capm_expected_return(),
        portfolio_quarterly_KPI.maximum_drawdown(), portfolio_quarterly_KPI.calamar_ratio())
    print(portfolio_quarterly_KPI_results)

    # comparative visualization of KPIs based on monthly and quarterly rebalancing
    graphs = ComparativeVisualization(
        portfolio_monthly_KPI_results, portfolio_quarterly_KPI_results, one_mo_combo, three_mo_combo)
    graphs.rebalancing_comparision_visualization()
    graphs.KPI_comparision_visualization()

    # Results of the backtest in terms of compartive returns
    Index = ["Market","Monthly Portfolio Rebalancing CAGR","Monthly Portfolio Rebalancing Alpha",
             "Quarterly Portfolio Rebalancing", "Quarterly Portfolio Rebalancing Alpha"]
    returns = {
    "Returns (%)" : [
        portfolio_monthly_KPI.market_cagr().round(4)*100, portfolio_monthly_KPI.cagr().round(4)*100,
        portfolio_monthly_KPI.alpha().round(4)*100,portfolio_quarterly_KPI.cagr().round(4)*100,
        portfolio_quarterly_KPI.alpha().round(4)*100]
    }
    rdf = pd.DataFrame(returns,index = Index)
    print(rdf)

    

    
    

    

    

    


    

    

    

    

    

    

    

    

    



