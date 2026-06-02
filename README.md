### Project Title

**Isaiah Hayes**

#### Executive summary
Startup founders, investors, and business analysts have limited information about an early-stage startup's future success, yet they're forced to make critical decisions at this stage. This project explores whether early-stage business indicators can be used to predict startup outcomes like failure, acquisition, or an initial public offering (IPO). Using a dataset of 100,000 startup companies, exploratory data analysis and machine learning techniques were applied to identify patterns and build predictive models aimed at achieving said goal. This project is aimed at assisting stakeholders in understanding which factors are most strongly associated with startup success, and whether they can be used to reliably predict a future outcome.

#### Rationale
Forbes presented a statistic stating that 90% of startups fail in the first 10 years. That makes it incredibly challenging for stakeholders to identify the promising opportunities. However, defining which early-stage factors are most associated with long-term success will support better investment decisions, reduce risk exposure, and help founders focus on what actually drives sustainable growth.

#### Research Question
Can a startup's future outcome (failure, acquisition, or IPO) be reliably predicting by using a small set of early-stage business indicators like revenue, product traction, funding activity, market size, founder experience, and others?

#### Data Sources
These models use a dataset from Kaggle, covering key business and operational factors for startups which include market size, founder experience and background, industry, revenue, burn rate, and more. The target variable includes the following classification outcomes: Failure, Acquisition, or IPO. Features span numerical and categorical types.

#### Methodology
The project follows the CRISP-DM framework. This framework will cover Business Understanding, Exploratory Data Analysis (EDA), Data Preparation, Predictive Modeling, and Evaluation.

Analysis techniques will include:
* Data science libraries including pandas, NumPy, matplotlib, seaborn, and scikit-learn
* ML algorithms and models such as feature encoding, scaling, log transformations, Logistic Regression, Regularization, and Decision Trees
* Python programming language
* Jupyter Notebook dev environment

Categorical features were one-hot encoded. The target variable was label encoded for classification.

#### Results
Pair plots revealed a strong relationship between product traction and revenue across startup outcomes. Additionally, companies that reached IPO status generally exhibited the highest user traction and highest revenues. Contrary, failed startups typically had the lowest revenue and user count.

Furthermore, burn rate revealed a strong relationship with revenue and product traction. The data suggests that startups with higher burn rates had lower overall revenue and product traction.

Since revenue also demonstrates a strong relationship with startup outcomes, these features may indirectly contribute to predicting outcome classifications.

After encoding categorical features and initializing a baseline logistic regression model, analysis showed an accuracy result of 64% on the test data.
However, after scaling the data, the second logistic regession model resulted in an accuracy score of 74%.

#### Next steps
Next, I will reframe the problem as a binary classification task, combining "Acquisitions" and "IPO" outcome classes into a single "Success" category. I excpect this to create a more balanced target variable (outcome), which may lead to improved predictive performance.

#### Outline of project

- [Link to notebook 1](https://github.com/IsaiahWHayes/startup-success-prediction)
- [Link to notebook 2](https://github.com/IsaiahWHayes/startup-success-prediction)
- [Link to notebook 3](https://github.com/IsaiahWHayes/startup-success-prediction-capstone)


##### Contact and Further Information
For questions and additional information regarding this project, please contact: Isaiah Hayes
**LinkedIn**: https://www.linkedin.com/in/isaiahwhayes
**Kaggle Dataset**: https://www.kaggle.com/datasets/dhrubangtalukdar/startup-funding-and-outcome-dataset?resource=download
**Forbes Resource**: https://www.forbes.com/councils/forbescoachescouncil/2024/09/10/90-of-startups-fail-how-to-secure-your-place-in-the-10/
