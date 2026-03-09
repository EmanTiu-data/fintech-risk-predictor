# fintech-risk-predictor
This is a Full-Stack Data Science application designed to analyze and predict credit risk in the Fintech sector. The project combines a robust SQL backend with a Machine Learning engine to provide real-time decision-making support for loan approvals.
​Goal: To bridge the gap between raw financial data and actionable business insights through an interactive, production-ready dashboard.
​# Key Features
​Predictive Engine: Powered by a Random Forest Classifier to assess the probability of default based on user profiles.
​Interactive Simulation: A dedicated "Credit Simulator" sidebar that allows users to input data and receive instant risk scores.
​SQL Integration: Data is managed via a relational SQLite database, ensuring data integrity and efficient querying.
​Advanced Analytics: Interactive visualizations built with Plotly to track KPIs such as Default Rate, Average Ticket, and Risk Distribution.
​Aesthetic UI: A custom-styled Streamlit interface focusing on user experience (UX) and clean data storytelling.
​# Tech Stack
​Language: Python 3.x
​Dashboard: Streamlit
​Machine Learning: Scikit-Learn (Random Forest, Label Encoding)
​Database: SQLite3
​Visualization: Plotly Express / Pandas
​DevOps: GitHub & Streamlit Community Cloud
# project structure
├── app.py              # Main Streamlit application & ML Logic

├── fintech_risk.db     # SQLite Database with 1,000+ credit records

├── requirements.txt    # Project dependencies

└── README.md           # Documentation

# Run the app
streamlit run app.py

# Why this matters?
​In the modern Fintech landscape, distinguishing between "Good" and "Bad" risk is vital for profitability. This project demonstrates the ability to build end-to-end solutions: from data storage and cleaning to predictive modeling and cloud deployment.

​Developed by Emanuel | Data Science Student & Aspiring FinTech Engineer
