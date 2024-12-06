"""
# Practice Questions and Answers

## 1. How do you assess the statistical significance of an insight?
- **Perform Hypothesis Testing**: Formulate null and alternative hypotheses, then use statistical tests (e.g., t-test, chi-square test).
- **Compute p-value**: A p-value indicates the probability of observing the data if the null hypothesis is true. A p-value less than a significance level (e.g., 0.05) suggests rejecting the null hypothesis.
- **Check Effect Size**: Statistical significance alone isn't enough; evaluate the practical significance using effect size measures.
- **Cross-validate Insights**: Use different subsets of the data or replicate experiments to ensure robustness.

## 2. What is the Central Limit Theorem? Explain it. Why is it important?
- **Definition**: The Central Limit Theorem (CLT) states that the sampling distribution of the sample mean approaches a normal distribution as the sample size grows, regardless of the population distribution, provided the samples are independent and identically distributed (i.i.d.).
- **Why It's Important**:
  - Enables **parametric statistical tests** like t-tests and confidence intervals by justifying the assumption of normality.
  - Supports the use of sample means for inference about population parameters.
  - Helps simplify problems involving sampling and estimation.

## 3. What is the statistical power?
- **Definition**: Statistical power is the probability that a test will correctly reject the null hypothesis when the alternative hypothesis is true (i.e., avoiding a Type II error).
- **Key Factors Influencing Power**:
  - Sample size: Larger samples increase power.
  - Effect size: Larger effects are easier to detect.
  - Significance level (α): A higher α increases power but risks false positives.
  - Variability: Lower variability in data increases power.

## 4. How do you control for biases?
- **Randomization**: Randomly assign subjects to groups or treatments.
- **Blinding**: Use single or double-blind study designs to prevent bias from expectations.
- **Stratification**: Ensure balanced representation of key variables across groups.
- **Matching**: Pair subjects with similar characteristics to control for confounding factors.
- **Adjustments in Analysis**: Use statistical methods like regression or propensity score matching to account for potential biases.
- **Clear Protocols**: Standardize data collection and handling procedures.

## 5. What are confounding variables?
- **Definition**: Confounding variables are extraneous variables that are related to both the independent variable (predictor) and the dependent variable (outcome). They distort the true relationship between the variables of interest.
- **Example**: Studying the relationship between exercise and weight loss, but not accounting for diet, which affects both exercise habits and weight loss.

## 6. What is A/B testing?
- **Definition**: A/B testing is a statistical method to compare two variants (A and B) to determine which performs better for a specific metric.
- **Key Steps**:
  - Split the audience randomly into two groups.
  - Expose each group to a different variant.
  - Measure a predefined metric (e.g., click-through rate, conversion rate).
  - Use hypothesis testing to determine whether the observed differences are statistically significant.
- **Applications**: Marketing campaigns, UI/UX design, product feature testing.

## 7. What are confidence intervals?
- **Definition**: A confidence interval (CI) is a range of values, derived from the sample, that is likely to contain the population parameter (e.g., mean) with a certain level of confidence (e.g., 95%).
- **Importance**:
  - Provides an estimate of uncertainty around the sample statistic.
  - Offers more information than a point estimate by quantifying the precision of the estimate.
- **Interpretation**: A 95% CI means that if the study were repeated many times, 95% of the intervals would contain the true parameter value.
"""
