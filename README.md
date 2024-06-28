# Predicting the NBA champion with a Linear Regression Machine Learning Model

I learned about linear regression machine learning models recently and I wanted to practice that knowledge on a subject that I care a lot about, the NBA. Every year since middle school, I always stayed up to date on the latest NBA news, watching NBA games, and most relevantly, looking at team/player statistics. I was a big NBA stathead growing up. Big, impactful numbers always caught my attention in the NBA. I'd always look at the NBA website to see the different updates in player/team statistics. Due to my fascination and knowledge on NBA statistics, I thought that I could best apply that knowledge into this machine learning model, to predict who would win the NBA championship.

How it was made:
- I acquired regular season data from all 30 teams in 15 random seasons from 1999 to 2024. 15 was an arbitrarily chosen number. I didn't want to use all the seasons in NBA history because the model could potentially memorize the winner from each season. I also tried to stick to more recent seasons for data because the data during the 2000s is far more reliable than the data from earlier in NBA history. The data I acquired for each team was:
  - Average Age: Really young teams typically did not end up winning the championship, while incredibly old teams were far too burnt out to make any noise most of the time
  - Offensive Rating: How good a team's offense was (higher rating meant better offense)
  - Defensive Rating: How good a team's defense was (lower rating meant better defense)
  - Net Rating: How good were teams overall (higher rating meant a better team overall)
  - Margin of Victory: How much was a team beating its opponents by (higher margin meant they won by more points)
  - Simple Rating System: How much a team was better than the other teams (higher SRS the better a team was than the league)
 I chose these data points because I believe they were some of the most effective measurements of how good a team was relative to the rest of the league that season (at least during the regular season).

- Putting the data into excel, I then converted the data into a ranking rather than the pure numbers. I did this for a few reasons:
  1. To account for different eraas of the NBA (early 2000s had far worse offenses but far better defenses statistically than the late 2010s + early 2020s)
  2. Better reflected how much better a team was than the rest of the league (a team being first in net rating says a lot more than just the net rating itself)
 I converted the data into rankings with excel functions. (This was much harder than anticipated) I also added a championship column where 0 means a team did not win while 1 means a team did win

- I made two different datasets. One for training the model (this was made with the 15 random seasons). Another dataset was amde for testing the model (I picked another 6 arbitrary seasons and did the same thing from the previous steps) These datasets were saved as .csv files

- After that I created a python file importing:
  - TensorFlow (allowed to convert data into tensors to use in the models that they provide)
  - Numpy (allows for vectors to be represented and manipulated)
  - Pandas (data analytics tool, used to actually upload the .csv files)
  - PyPlot (plot the data into a graph for visual representation)

- I then uploaded the data from the .csv files into python with pandas. I used .head() to make sure the data was uploaded proparly. I popped out the championship column for both datasets because they are labels (outputs). 

- I then specified categorical columns (non-numerical data) and numerical columns (numerical data). Categorical data cannot be analyzed by a linear regression model, so encoding the data is necessary to let the model use it. Taking the encoded categorical columns and the numerical columns, they were added to a feature columns list. Feature columns is how the model receives input data.

- I then created a make input function function lol. This allows for different input functions to be made for different use cases. An input function takes in feature dataset (input), label datasest (output), num_of_epochs(essentially re-running the data through the model), whether to shuffle (give the model new ways to look at the data), and batch size (best to use batches for really large datasets so the model does not get overwhelmed). I used the function to make an input function for training the model and another one to test the model. 

- I then created the linear regression model using the feature columns list made earlier with TensoFlow's LinearClassifier module.

- Afterwards, I trained it using the training data. The training went relatively fast because there really was not much data (only 450 different teams, 15 batches of 30 teams each).

Results:
- The model had a supposed 97% accuracy after testing it with the testing dataset. However I believe this lacks context. The model was very good at telling teams who could not win the championship, which was the vast majority of teams inputted. Thus, the accuracy isn't really a reflection on the model's ability to tell who would win the championship, but more on its ability to tell you who would not.
- I ran some personal tests with the prediction method on the dataset, and I personally got a 50% hit rate on the model's prediction for the championship winner. To determine the model's choice for a winner, I chose the team who had the best probability of winning the championship according to the model.

Reflection:
I am very pleased with this project so far. I did not come into programming this model with the expectation that my model was going to be a revolutionary, game changing model. I learned how to apply linear regression machine learning to a subject I am passionate about, and I am very satisfied with that. This was an incredibly difficult but fun experience overall. However, I do want to improve upon the model's odds on picking the championship winner in the future (for fun and other reasons...). Some things I could improve upon include:
- Using more NBA seasons for data, more seasons allow for the model to better predict future winners
- Researching other advanced metrics that reflect a team's ability to win a championship (all-stars, balance, playoff experience, etc.)
- Cutting down teams inputted from each season to the 16 who made the playoffs. The data got very bloated with hundreds of very obviously bad teams who had no chance of even winning the championship (because they were not in the running in the first place)
- Possibly including more playoff results in general (first round exist, second round exist, finals appearance, etc)
- Cooperating with friends so data acquisition does not take ages...
- Fine tuning the epochs and shuffling of data

Overall, I am happy to have this model be a representation in my progression in learning about machine learning. Also, its almost impossible to 100% say who can win a championship in any season with just data. There are some things about a team that just can't be put into numbers (the 2023 Miami Heat is a prime example of this). However, this does not stop NBA statheads like myself from trying to do so anyways. Artificial intelligence has vast potential to impact human life overall, and I want to tap into that. This project is just the beginning of my overall journey in artificial intelligence.
