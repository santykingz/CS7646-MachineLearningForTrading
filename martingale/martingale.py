""""""  		  	   		 	 	 			  		 			 	 	 		 		 	
"""Assess a betting strategy.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
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
  		  	   		 	 	 			  		 			 	 	 		 		 	
Student Name: Tucker Balch (replace with your name)  		  	   		 	 	 			  		 			 	 	 		 		 	
GT User ID: tb34 (replace with your User ID)  		  	   		 	 	 			  		 			 	 	 		 		 	
GT ID: 900897987 (replace with your GT ID)  		  	   		 	 	 			  		 			 	 	 		 		 	
"""  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
import numpy as np  		  	   		 	 	 			  		 			 	 	 		 		 	
import pandas as pd
import matplotlib.pyplot as plt

  		  	   		 	 	 			  		 			 	 	 		 		 	
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
  		  	   		 	 	 			  		 			 	 	 		 		 	
def gtid():  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: The GT ID of the student  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: int  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    return 902422399  # replace with your GT ID number
  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
def get_spin_result(win_prob):  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
    :param win_prob: The probability of winning  		  	   		 	 	 			  		 			 	 	 		 		 	
    :type win_prob: float  		  	   		 	 	 			  		 			 	 	 		 		 	
    :return: The result of the spin.  		  	   		 	 	 			  		 			 	 	 		 		 	
    :rtype: bool  		  	   		 	 	 			  		 			 	 	 		 		 	
    """  		  	   		 	 	 			  		 			 	 	 		 		 	
    result = False  		  	   		 	 	 			  		 			 	 	 		 		 	
    if np.random.random() <= win_prob:  		  	   		 	 	 			  		 			 	 	 		 		 	
        result = True  		  	   		 	 	 			  		 			 	 	 		 		 	
    return result  		  	   		 	 	 			  		 			 	 	 		 		 	

def play_game(win_prob, bankroll = float('inf')):
    """
    1 episode is 1000 iterations, or forward fill the last value

    returns list
    """
    episode_winnings = 0
    winnings = [0]
    counter = 0
    while (episode_winnings < 80 and episode_winnings > -1 * bankroll) and counter < 1000:
        won = False
        counter = counter + 1
        bet_amount = 1
        while not won and episode_winnings > (-1 * bankroll):
            # wager bet_amount on black
            won = get_spin_result(win_prob)
            if won == True:
                episode_winnings = episode_winnings + bet_amount
            else:
                episode_winnings = episode_winnings - bet_amount
                bet_amount = bet_amount * 2
                if bet_amount > (bankroll + episode_winnings):
                    bet_amount = min(bet_amount, bankroll + episode_winnings)
            # record the winnings
            winnings.append(episode_winnings)
    winnings = forward_fill_list(winnings, 1000)
    return np.array(winnings[0:1000])

def forward_fill_list(list_in, length_needed):
    # pad the winnings to the correct length
    last_val = list_in[-1]
    while len(list_in) <  length_needed:
        list_in.append(last_val)
    return list_in

def create_figure_1(win_prob):
    """
    Run your simple simulator 10 episodes and track the winnings, starting
    from 0 each time.
    Plot all 10 episodes on one chart using Matplotlib functions.
    The horizontal (X) axis must range from 0 to 300,
    the vertical (Y) axis must range from –256 to +100.
    We will not be surprised if some of the plot lines are not visible
    because they exceed the vertical or horizontal scales.
    """
    for i in range(0,10):
        episode_history = play_game(win_prob)
        plt.plot(range(0,300), episode_history[0:300], label = 'episode ' + str(i))
    plt.title('Winnings Over Time for 10 Martingale Betting Episodes')
    plt.xlim((0,300))
    plt.legend()
    plt.ylim((-256,100))
    plt.grid(True)
    plt.xlabel('rounds')
    plt.ylabel('winnings')

    # Save the chart as a PNG file
    plt.savefig("./images/figure1.png")
    plt.close()

def experiment_1(win_prob):
    episodes = []
    for i in range(0,1000):
        episode_history = play_game(win_prob)
        episodes.append(episode_history)
    # save the episodes into a csv
    df = pd.DataFrame(episodes).T
    df.to_csv('experiment1_1000_episodes.csv', index=False)
    # calculate the stats
    mean_vals = np.mean(episodes,axis=0)[0:300]
    med_vals = np.median(episodes, axis = 0)[0:300]
    std_vals = np.std(episodes, axis = 0, ddof=0)[0:300]
    return mean_vals, med_vals, std_vals

def create_figure_2_3(win_prob):
    """
    Run your simple simulator 1000 episodes.
    (Remember that 1000 successive bets are one episode.)
    Plot the mean value of winnings for each spin round using the same axis bounds as Figure 1.
    For example, you should take the mean of the first spin of all 1000 episodes.
    Add an additional line above and below the mean,
    at mean plus standard deviation,
    and mean minus standard deviation of the winnings at each point.
    """
    # call the experiment 1
    mean_vals, med_vals, std_vals = experiment_1(win_prob)

    # make the chart for figure 2
    plt.plot(range(0,300), mean_vals, label = 'mean')
    plt.plot(range(0,300), mean_vals + std_vals, label = 'mean + std')
    plt.plot(range(0,300), mean_vals - std_vals, label = 'mean - std')
    plt.legend()
    plt.title('Mean Winnings and Std Dev Across 1000 Episodes')
    plt.xlim((0,300))
    plt.ylim((-256,100))
    plt.grid(True)
    plt.xlabel('rounds')
    plt.ylabel('winnings')

    # Save the chart as a PNG file
    plt.savefig("./images/figure2.png")
    plt.close()

    # make the figure 3
    plt.plot(range(0,300), med_vals, label = 'median')
    plt.plot(range(0,300), med_vals + std_vals, label = 'median + std')
    plt.plot(range(0,300), med_vals - std_vals, label = 'median - std')
    plt.legend()
    plt.title('Median Winnings and Std Dev Across 1000 Episodes')
    plt.xlim((0,300))
    plt.ylim((-256,100))
    plt.grid(True)
    plt.xlabel('rounds')
    plt.ylabel('winnings')

    # Save the chart as a PNG file
    plt.savefig("./images/figure3.png")
    plt.close()

def experiment_2(win_prob):
    episodes = []
    for i in range(0,1000):
        episode_history = play_game(win_prob = win_prob, bankroll=256)
        episodes.append(episode_history)
    mean_vals = np.mean(episodes,axis=0)[0:300]
    med_vals = np.median(episodes,axis=0)[0:300]
    std_vals = np.std(episodes, axis = 0, ddof=0)[0:300]
    # save the episodes into a csv
    df = pd.DataFrame(episodes).T
    df.to_csv('experiment2_1000_episodes.csv', index=False)
    return mean_vals, med_vals, std_vals

def create_figure_4_5(win_prob):
    """
    Run your realistic simulator 1000 episodes and track the winnings,
    starting from 0 each time. Plot the mean value of winnings for each spin
    using the same axis bounds as Figure 1.
    Add an additional line above and below the mean at mean plus
    tandard deviation and mean minus standard deviation of the winnings
    at each point.
    """
    mean_vals, med_vals, std_vals = experiment_2(win_prob)

    # plot figure 4
    plt.plot(range(0,300), mean_vals, label = 'mean')
    plt.plot(range(0,300), mean_vals + std_vals, label = 'mean + std')
    plt.plot(range(0,300), mean_vals - std_vals, label = 'mean - std')
    plt.legend()
    plt.title('Mean Winnings With Bankroll Constraint Across 1000 Episodes')
    plt.xlim((0,300))
    plt.ylim((-256,100))
    plt.grid(True)
    plt.xlabel('rounds')
    plt.ylabel('winnings')

    # Save the chart as a PNG file
    plt.savefig("./images/figure4.png")
    plt.close()

    # plot figure 5
    plt.plot(range(0,300), med_vals, label = 'median')
    plt.plot(range(0,300), med_vals + std_vals, label = 'median + std')
    plt.plot(range(0,300), med_vals - std_vals, label = 'median - std')
    plt.legend()
    plt.title('Median Winnings With Bankroll Constraint Across 1000 Episodes')
    plt.xlim((0,300))
    plt.ylim((-256,100))
    plt.grid(True)
    plt.xlabel('rounds')
    plt.ylabel('winnings')

    # Save the chart as a PNG file
    plt.savefig("./images/figure5.png")
    plt.close()

def answer_questions():
    answers = pd.DataFrame(columns = ['Question','Answer'])
    # Question 1
    # calculate and provide the estimated probability of winning $80 within 1000 sequential bets.
    data = pd.read_csv('experiment1_1000_episodes.csv')
    last_row = data.iloc[-1].values
    count_80 = len([x for x in last_row if x == 80])
    prob_80 = count_80/1000
    answers = pd.concat([answers,
                         pd.DataFrame({
                            'Question':1,
                            'Answer': str({'prob_80':prob_80})
                         },index=[0])], ignore_index=True)

    # Question 4
    # the estimated probability of winning $80 within 1000 sequential bets.
    del data, last_row, count_80, prob_80
    data = pd.read_csv('experiment2_1000_episodes.csv')
    last_row = data.iloc[-1].values
    count_80 = len([x for x in last_row if x == 80])
    prob_80 = count_80/1000
    answers = pd.concat([answers,
                         pd.DataFrame({
                            'Question':2,
                            'Answer': str({
                                'prob_80':prob_80,
                                'count_80':count_80
                            })
                         },index=[0])], ignore_index=True)

    # standard deviation convergence
    data = pd.read_csv('experiment2_1000_episodes.csv')

    # save answers to csv
    answers.to_csv('answers.csv')

def test_code():
    """
    Method to test your code
    """
    win_prob = 18/38  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once
    print(get_spin_result(win_prob))  # test the roulette spin
    # add your code here to implement the experiments
    create_figure_1(win_prob)
    create_figure_2_3(win_prob)
    create_figure_4_5(win_prob)
    answer_questions()

if __name__ == "__main__":
    test_code()
