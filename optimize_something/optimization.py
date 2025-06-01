""""""  		  	   		 	 	 			  		 			 	 	 		 		 	
"""MC1-P2: Optimize a portfolio.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			 	 	 		 		 	
or edited.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Student Name: Santosha Spickard
GT User ID: sspickard3
GT ID: 902422399
"""  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
import datetime as dt  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
import numpy as np  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
import matplotlib.pyplot as plt  		  	   		 	 	 			  		 			 	 	 		 		 	
import pandas as pd  		  	   		 	 	 			  		 			 	 	 		 		 	
from util import get_data, plot_data
import scipy.optimize as spo
import matplotlib.dates as mdates

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "sspickard3"  # replace tb34 with your Georgia Tech username.

def study_group():
    """
    : return A comma separated string of GT_Name of each member of your study group

    :rtype: str
    """
    return "gburdell3"

def sum_to_one_constraint(allocations):
    """
    Makes sure that the values sum to 1
    Inputs: allocations(list)
    """
    return np.sum(allocations) - 1.0


def objective_error(allocation, prices, risk_free_rate = 0.0):
    """
    Computes the sharpe ratio for the input prices and the allocation provided.
    Inputs:
    - prices (pd.DataFrame): historical prices of the stocks in the portfolio
    - allocation (list): list of the allocations, sums to 1
    - risk_free_rate (float): amount being used to calculate the risk free rate
    Outputs:
    float of the sharpe ratio

    Source:
        https://www.codingfinance.com/post/2018-04-05-portfolio-returns-py/
    """
    # get the rate of return for each stock
    sharpe_ratio = portfolio_stats(
        prices=prices,
        allocation=allocation,
        risk_free_rate=risk_free_rate
    )['sharpe_ratio']
    # multiply it by negative 1 to get the error so we can minimize
    error = sharpe_ratio * -1
    return error

def portfolio_stats(prices, allocation, risk_free_rate=0):
    # calculate the daily returns
    daily_returns = prices.pct_change()[1:]
    weighted_daily_returns = daily_returns * allocation
    portfolio_returns  = weighted_daily_returns.sum(axis = 1)
    # get the summary stats
    portfolio_mean = portfolio_returns.mean()
    portfolio_std = portfolio_returns.std()
    # calculate the sharpe ratio
    sharpe_ratio = np.sqrt(252) * (portfolio_mean - risk_free_rate) / portfolio_std
    # calculate the cumulative return
    cumulative_return = calc_cumulative_return(
        prices = prices,
        allocation = allocation
        )
    res = {
        'sharpe_ratio': sharpe_ratio,
        'average_daily_return': portfolio_mean,
        'std_daily_return': portfolio_std,
        'cumulative_return':cumulative_return
    }
    return res

def calc_cumulative_return(prices, allocation):
    """
    Calculates the cumulatives return of the portfolio
    Inputs:
    prices (pd.DataFrame): prices over time
    allocation: list of the allocations

    Returns:
    float: cumulative retrurn as a percent
    """
    prices_weighted = prices * allocation
    prices_weighted['total'] = prices_weighted.sum(axis = 1)
    first_value = prices_weighted['total'][0]
    last_value = prices_weighted['total'][-1]
    cumulative_return = (last_value - first_value) / first_value
    return cumulative_return
  		  	   		 	 	 			  		 			 	 	 		 		 	
# This is the function that will be tested by the autograder  		  	   		 	 	 			  		 			 	 	 		 		 	
# The student must update this code to properly implement the functionality  		  	   		 	 	 			  		 			 	 	 		 		 	
def optimize_portfolio(  		  	   		 	 	 			  		 			 	 	 		 		 	
    sd=dt.datetime(2008, 1, 1),  		  	   		 	 	 			  		 			 	 	 		 		 	
    ed=dt.datetime(2009, 1, 1),  		  	   		 	 	 			  		 			 	 	 		 		 	
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		 	 	 			  		 			 	 	 		 		 	
    gen_plot=False,  		  	   		 	 	 			  		 			 	 	 		 		 	
):  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		 	 	 			  		 			 	 	 		 		 	
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		 	 	 			  		 			 	 	 		 		 	
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		 	 	 			  		 			 	 	 		 		 	
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		 	 	 			  		 			 	 	 		 		 	
    statistics.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type sd: datetime  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type ed: datetime  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		 	 	 			  		 			 	 	 		 		 	
        symbol in the data directory)  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type syms: list  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		 	 	 			  		 			 	 	 		 		 	
        code with gen_plot = False.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type gen_plot: bool  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		 	 	 			  		 			 	 	 		 		 	
        standard deviation of daily returns, and Sharpe ratio  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: tuple  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Read in adjusted closing prices for given symbols, date range  		  	   		 	 	 			  		 			 	 	 		 		 	
    dates = pd.date_range(sd, ed)  		  	   		 	 	 			  		 			 	 	 		 		 	
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		 	 	 			  		 			 	 	 		 		 	
    prices = prices_all[syms]  # only portfolio symbols
    prices.sort_index(inplace=True)
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # find the allocations for the optimal portfolio
    #=======
    # get the first guess
    first_allocation = list(np.ones(len(prices.columns)))

    # call the optimizer to minimize the error function
    result = spo.minimize(
        objective_error,
        first_allocation,
        args=(prices,),
        method='SLSQP',
        options={'disp': True},
        bounds = [(0,1) for _ in syms],
        constraints = ({
            'type':'eq',
            'fun': sum_to_one_constraint
        })
    )
    final_allocation = result.x
    # get the stats with the final allocation
    final_stats = portfolio_stats(
        prices=prices,
        allocation=final_allocation
    )
    final_cumulative_return = final_stats['cumulative_return']
    final_average_daily_return = final_stats['average_daily_return']
    final_std_daily_return = final_stats['std_daily_return']
    final_sharpe_ratio = final_stats['sharpe_ratio']

    final_allocation_np = np.array(final_allocation)

    if gen_plot:
        make_plot(
            prices=prices_all,
            comparative_stock_name='SPY',
            symbols=syms,
            allocation=final_allocation
        )
  		  	   		 	 	 			  		 			 	 	 		 		 	
    return final_allocation_np, final_cumulative_return, final_average_daily_return, final_std_daily_return, final_sharpe_ratio

def make_plot(prices, comparative_stock_name,symbols, allocation):
    """
    Save a plot which has the date as the x axis (formatted to Mar 2024),
    y axis is Prices and 2 lines - one for the comparative stock and one
    for the portfolio. They should be normalized so that they both start at $1.
    A legend should be shown.
    Inputs:
    - prices (pd.DataFrame) of all of the prices
    - comparative_stock_name (string): stock name of the comparative stock code to add to the plot
    - symbols (list(str)): list of strings of the stocks in the portfolio
    - allocation (list(float)): the stock allocations for each of the stocks in the portfolio
    """
    # get portfolio values
    portfolio_data =prices[symbols]
    portfolio_data = portfolio_data * allocation
    portfolio_data['total'] = portfolio_data.sum(axis=1)
    portfolio_data['Portfolio'] = portfolio_data['total'] / portfolio_data['total'].values[0]

    # get the comparative stock
    comp_data = prices[[comparative_stock_name]]
    comp_data_first_value = comp_data[comparative_stock_name].values[0]
    comp_data = comp_data/comp_data_first_value

    # graph the data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(portfolio_data.index, portfolio_data['Portfolio'], label='Portfolio')
    ax.plot(comp_data.index, comp_data[comparative_stock_name], label=comparative_stock_name)
    # format the axis
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalize Price')
    ax.legend()
    ax.grid(True)
    ax.set_title('Normalized Portfolio Value vs ' + comparative_stock_name)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/figure1.png')
    plt.close()


def test_code():  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    This function WILL NOT be called by the auto grader.  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    start_date = dt.datetime(2009, 1, 1)  		  	   		 	 	 			  		 			 	 	 		 		 	
    end_date = dt.datetime(2010, 1, 1)  		  	   		 	 	 			  		 			 	 	 		 		 	
    symbols = ["GOOG", "AAPL", "GLD", "XOM", "IBM"]
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Assess the portfolio  		  	   		 	 	 			  		 			 	 	 		 		 	
    allocs, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )
  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Print statistics  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Start Date: {start_date}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"End Date: {end_date}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Symbols: {symbols}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Allocations:{allocs}")
    print(f"Sharpe Ratio: {sr}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Average Daily Return: {adr}")  		  	   		 	 	 			  		 			 	 	 		 		 	
    print(f"Cumulative Return: {cr}")


    # Test the second case
    # start_date = dt.datetime(2008, 6, 1)
    # end_date = dt.datetime(2009, 6, 1)
    # symbols=['IBM', 'X', 'GLD', 'JPM']
    # allocs, cr, adr, sddr, sr = optimize_portfolio(
    # sd=start_date,
    # ed=end_date,
    # syms=symbols,
    # gen_plot=True)

    # # Print statistics
    # print(f"Start Date: {start_date}")
    # print(f"End Date: {end_date}")
    # print(f"Symbols: {symbols}")
    # print(f"Allocations:{allocs}")
    # print(f"Sharpe Ratio: {sr}")
    # print(f"Volatility (stdev of daily returns): {sddr}")
    # print(f"Average Daily Return: {adr}")
    # print(f"Cumulative Return: {cr}")

  		  	   		 	 	 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		 	 	 			  		 			 	 	 		 		 	
    # This code WILL NOT be called by the auto grader  		  	   		 	 	 			  		 			 	 	 		 		 	
    # Do not assume that it will be called  		  	   		 	 	 			  		 			 	 	 		 		 	
    test_code()  		  	   		 	 	 			  		 			 	 	 		 		 	
