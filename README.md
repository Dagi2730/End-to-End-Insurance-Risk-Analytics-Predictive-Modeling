# Task 2: Statistical Hypothesis Testing - Insurance Risk Analysis

## 📄 Overview
This task performs statistical hypothesis testing on an insurance dataset of **10,000 records** to identify significant differences in **risk profiles** and **profit margins** across provinces, zip codes, and gender. The results guide feature engineering for predictive modeling.

**Dataset Features:**  
- PolicyID  
- Province  
- ZipCode  
- Gender  
- VehicleType  
- VehicleIntroDate  
- CustomValueEstimate  
- TotalPremium  
- TotalClaims  
- TransactionMonth  

---

## 🧪 Hypotheses Tested

| Hypothesis | Description | Test | Result | Effect Size | Business Impact |
|------------|-------------|------|--------|------------|----------------|
| H1 | Provincial Risk Differences | Kruskal-Wallis | ✅ Significant | Negligible (0.0009) | HIGH – province-specific adjustments recommended |
| H2 | Zip Code Risk Differences | Kruskal-Wallis | ❌ Not Significant | N/A | LOW – current approach adequate |
| H3 | Zip Code Profit Margins | Kruskal-Wallis | ❌ Not Significant | N/A | LOW – current approach adequate |
| H4 | Gender Risk Differences | Mann-Whitney U | ❌ Not Significant | Negligible (-0.0005) | LOW – current approach adequate |

---

## 🔍 Key Findings

1. **Provincial Risk Differences (H1)**  
   - Statistically significant differences detected across provinces.  
   - Effect size is negligible but highlights the need for **province-specific risk adjustments**.  
   - Significant post-hoc pairs:  
     - KwaZulu-Natal vs Eastern Cape ✅  
     - Eastern Cape vs Western Cape ✅  

2. **Zip Code Differences (H2 & H3)**  
   - Insufficient data for meaningful analysis (H2: risk, H3: profit margins).  
   - Not significant; minimal impact on modeling decisions.  

3. **Gender Differences (H4)**  
   - No significant differences detected.  
   - Effect size is negligible; gender may not be a critical feature.  

---

## 📊 Executive Summary

- **Total hypotheses tested:** 4  
- **Significant hypotheses:** 1/4 (25%)  
- **Overall implication:** Geographic (province-level) differences should be considered for pricing and risk modeling. Other factors (zip code, gender) are less influential in this dataset.

---

## 💼 Strategic Recommendations

1. **Geographic Pricing:** Implement **province-specific risk adjustments**.  
2. **Regional Analysis:** Conduct further analysis on provincial markets to optimize risk prediction.  
3. **Feature Engineering:** Include **province** as a key predictive feature in modeling.  
4. **Zip Code & Gender:** Consider for exploratory feature engineering but not critical.  

---

## ⚠️ Statistical Notes

- **Sample Size:** 10,000 (high statistical power)  
- **Significance Level:** α = 0.05  
- **Multiple Comparisons:** Bonferroni corrections applied  
- **Assumptions:** Non-parametric tests used when normality or equal variance assumptions were violated  
- **Effect Sizes:** Reported for practical significance assessment  

---

## 💾 Results Storage

All hypothesis testing results are saved as JSON:  

