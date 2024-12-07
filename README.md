# Instagram Reach Prediction Model  

## Overview  
This project uses Python, data science techniques, and machine learning to analyze and predict Instagram reach based on various post metrics. It provides insights into how impressions, likes, shares, saves, and other factors correlate to engagement, helping users optimize their content for maximum visibility.  

---

## Features  
- **Data Analysis**:  
  - Visualization of impressions from different sources (home, hashtags, explore, etc.).  
  - WordClouds to analyze frequently used captions and hashtags.  
  - Scatter plots to explore relationships between impressions and engagement metrics like likes, comments, and shares.  

- **Correlation Analysis**:  
  - Heatmap of correlations between metrics such as impressions, likes, shares, and profile visits.  
  - Conversion rate calculation to analyze follower growth based on profile visits.  

- **Machine Learning**:  
  - Developed a predictive model using the **Passive Aggressive Regressor** to forecast Instagram post reach based on input features.  

---

## Dependencies  
This project requires the following Python libraries:  
- `pandas`  
- `numpy`  
- `matplotlib`  
- `seaborn`  
- `plotly`  
- `wordcloud`  
- `sklearn`  

Install dependencies using:  
```bash
pip install pandas numpy matplotlib seaborn plotly wordcloud scikit-learn
