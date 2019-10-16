# S2DSproject
Project developed during Science to Data Science 2019 (London), in collaboration with MADE.com

The aim of the project was to estimate the potential value of customers visiting the MADE.com website over a period of 90 days using tracking data relative to the first visit of the website. To this end, we designed a two-step model (Catboost_step1_step2.ipynb): first, we aimed at correctly classifying visitors that converted to customers in the time period of interest. Second, we developed a regression model to estimate the potential revenue for the converted customers. The model performance evaluation (model_performance.ipynb) shows that with this approach we can improve the precision by 65 % (compared to a naive model based on averaging of historical data), or, equivalently, reach the same precision with 9x less data. 

The GeoCoding folders containes routines to convert Google Analytics city information to geolocation (accounting for language differences, multiple cities with the same name, etc).

