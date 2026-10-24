# Factors Affecting Performance in Swimming

Group research project for module **327** at North-West University. The project asks which factors influence a swimmer's performance score. It answers the question with exploratory data analysis, hypothesis tests and multiple linear regression on a sports-performance dataset, and presents the results in a slide deck.

**Team (NazriFC):** Tafara Kanyemba, Unathi Manana, Neo Zandamela

## Research question and method

The data is `sport_perform.csv`. According to the presentation it has 2,000 observations and 33 variables. The analysis keeps only the swimming athletes, about 200 rows, and models the `perform` score against physical, technical and psychological measures. These are numbered columns such as `x11` (endurance), `x13` (fitness), `x16` (lactate threshold) and `x23` (psychological factors). `code.py` is organised into the project's phases:

1. **Phase 1: Exploratory analysis**
   - Encodes gender as binary and builds a correlation matrix.
   - Drops `x22` (physical fitness) because it is identical to `x13` (fitness).
   - Bins several measures into three-level categories and plots performance against endurance, fitness, lactate threshold, maximal oxygen uptake and muscle strength. The plots include scatter and regression plots, boxplots, count plots and bubble plots.
2. **Phase 2: Hypothesis testing**
   - Runs a chi-square test of independence between performance level and psychological level (`x23`).
   - Compares male and female performance. Normality is checked with a Shapiro-Wilk test on the standardised, pooled data and equal variances with Levene's test, followed by a two-sample t-test.
3. **Phase 3: Regression model**
   - Fits OLS with all predictors, then a reduced model: `perform ~ gender + x07 + x09 + x11 + x16 + x20 + x21 + x28` (body alignment and posture, decision-making, endurance, lactate threshold, muscle strength, pattern recognition, technical skills). Also fits separate male and female models.
   - Diagnostics:
     - multicollinearity: variance inflation factors (VIF)
     - residual normality: Q-Q plot, histogram with KDE, Shapiro-Wilk test
     - influential observations: Cook's distance, leverage, DFFITS, studentised residuals
     - heteroskedasticity: residual plots, Breusch-Pagan and White tests
   - Refits the model without the influential rows and uses it to predict performance for an example swimmer.
4. **Phase 4: Discussion.** This phase is in the presentation only, where the significant variables are related to the swimming literature. The Phase 4 heading in `code.py` is empty.

## Project structure

```
code.py                          # full analysis script (Phases 1-3)
Python code.docx                 # Word copy of the same code, with a team header
327_presentation_final_1_.pptx   # final presentation: method, plots, test results, discussion
requirements.txt                 # Python dependencies
```

## Getting started

Prerequisites: Python 3 and a copy of `sport_perform.csv`. **The dataset is not in this repository.**

```bash
pip install -r requirements.txt
```

1. Edit line 20 of `code.py` so it points to your copy of the data. The path is currently hard-coded to a Windows location:
   ```python
   data = pd.read_csv("C:/pythonproj/sport_perform.csv")
   ```
2. Run the script in an interactive environment such as Spyder, VS Code/Jupyter cells or IPython. It was written to be run section by section. Several results are bare expressions, such as `mod.summary()` and `chi2_contingency(tab)`, which display only in an interactive console. The script also opens more than 30 plot windows one after another.

   ```bash
   python code.py   # works, but each plot blocks until closed and the bare-expression results are not printed
   ```

## Results

The plots, test results and final model are in [`327_presentation_final_1_.pptx`](327_presentation_final_1_.pptx). In summary, the slides report:

- no significant relationship between psychological factors and performance (chi-square, p = 0.502)
- no significant difference in performance between male and female swimmers (t-test, p = 0.485)

The final regression model explains performance using gender, body alignment and posture, decision-making, endurance, lactate threshold, muscle strength, pattern recognition and technical skills.
