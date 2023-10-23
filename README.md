# Customer-segmentation-and-product-recommendation-using-k-means-model
The data, content, and final delivery I shared on GitHub utilize mock-up data due to confidentiality. This work sample aims to demonstrate my analytical skills, problem-solving approach, and the lessons I learned from the case.

## 1. Introduction
In the contemporary insurance market, understanding customer behavior and preferences has become paramount for ensuring effective marketing and sales strategies. As a data analyst at Applica Solutions, I took on a project for a prominent insurance company that aimed to delve deep into their customer data. The project sought to segment customers based on a variety of parameters and analyze the efficacy of different segmentation methods, to optimize the company's marketing campaigns.

![image](https://github.com/Sarashang1/Customer-segmentation-and-product-recommendation-using-k-means-model/assets/115900641/cfbe3bad-c3cf-4171-999a-025c07564a19)

## 2. Objective
The primary objective of this project was:

- To uncover insights from the existing customer data.
- To effectively segment the company's customers to facilitate targeted marketing.
- Evaluate the performance of different segmentation models through A/B testing.
- Analyze and optimize the insurance company's email marketing campaigns using relevant KPIs.
- Integrate the derived insights back into the Zoho CRM application for enhanced real-time targeting.

## 3. Data Source and Preprocessing

### Data Sources:
- Zoho CRM Application: This was the primary source of the customer demographic data and past purchase records. The data was extracted using API calls facilitated through Python.
- Company Database: This internal database stored information related to customer interactions on the website and their responses to email advertisements. SQL queries were employed to fetch this data.

### Preprocessing Steps:
- Cleaning and Integration: The data from the different sources was integrated into a unified structure. Rigorous cleaning processes were applied which involved:
- Handling missing values and outliers.
- Removing duplicate records.
- Ensuring consistent data formats across sources.
- Selecting the most pertinent features for modeling.
  
## 4. Action

### Customer Segmentation: The segmentation was carried out in two distinct ways:
- RFM Model: This traditional method involved segmenting customers based on three metrics: Recency (when was the last purchase), Frequency (how often do they purchase), and Monetary (how much do they spend). Each customer was assigned scores for each of these metrics, enabling distinct customer groups to emerge.
- K-means Clustering: This is an AI-driven unsupervised learning method. It segments the customers by identifying centroids of data clusters and assigns data points (customers) to the nearest centroid. The number of centroids (or clusters) was determined using methods like the Elbow method.
- A/B Testing: With these two models, we launched targeted marketing campaigns and evaluated their performances by comparing key metrics, such as conversion rates and click-through rates.

### Email Campaign Analysis: An interactive dashboard was created using Power BI to display the following KPIs:
- Clickthrough Rate: Measures how many people clicked on the content links.
- Conversion Rate: Assesses how many clicks turned into actual sales or sign-ups.
- Bounce Rate: Shows the percentage of visitors who leave after viewing just one page.
- List Growth Rate: Monitors the growth of the email subscription list.
- Email Sharing/Forwarding Rate: Evaluates the virality of the emails.
- Overall ROI: Calculates the return on investment for the email campaigns.
- Open Rate: Measures the percentage of recipients who opened the email.
- Unsubscribe Rate: Reflects the percentage of recipients who chose to opt out of receiving future emails.
- CRM Application Enhancement: The K-means model predictions were used to segment new clients based on their likelihood to belong to a certain segment. Recommendations for the top three insurance products likely to be purchased by each segment were also provided. This data was integrated back into the Zoho CRM system in collaboration with the development team, enabling marketing and sales teams to access real-time recommendations.

## 5. Result
Upon comparing the two customer segmentation models, the following observations were made:
- RFM Model: While the RFM model provided a good initial insight into customer behaviors based on their transactional history, it had certain limitations. The model was primarily deterministic and did not consider other potential behavioral or interaction-based features. However, for campaigns that targeted customer loyalty and purchase intent, this model was considerably effective.
- K-means Clustering: The K-means model, being data-driven, offered a more holistic view of the customer base. It considered not just transactional data but also interaction data, leading to nuanced segments. This model was particularly efficient in uncovering hidden patterns and behaviors which the RFM model might have overlooked.
- In terms of actual campaign performance, targeted campaigns based on K-means showed a higher conversion rate, suggesting that this method was more precise in its customer segmentation. However, for more traditional campaign strategies, especially those revolving around loyalty programs, the RFM model remained a reliable tool.

Combining the strengths of both models, we achieved a comprehensive understanding of the customer base. Furthermore, the integration of customer segmentation and product recommendations into the CRM system led to a marked improvement in the efficiency of targeted campaigns, with a significant 26% boost. Through this project, we not only segmented the customer base effectively but also provided actionable insights that the insurance company could leverage for improved marketing outcomes.

The integration of customer segmentation and product recommendations into the CRM system led to a marked improvement in the efficiency of targeted campaigns, with a significant 26% boost. Through this project, we not only segmented the customer base effectively but also provided actionable insights that the insurance company could leverage for improved marketing outcomes.
