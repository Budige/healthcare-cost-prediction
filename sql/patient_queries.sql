-- Healthcare Cost Analysis Queries
-- Author: Rakesh Budige

-- Patient demographics summary
SELECT 
    CASE 
        WHEN age < 45 THEN '18-44'
        WHEN age < 65 THEN '45-64'
        WHEN age < 75 THEN '65-74'
        ELSE '75+'
    END AS age_group,
    COUNT(*) AS patient_count,
    AVG(healthcare_cost) AS avg_cost,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY healthcare_cost) AS median_cost
FROM patients
GROUP BY age_group
ORDER BY age_group;

-- High-risk patients (top 10%)
-- TODO: Add comorbidity score calculation
WITH cost_percentile AS (
    SELECT 
        patient_id,
        healthcare_cost,
        NTILE(10) OVER (ORDER BY healthcare_cost) AS decile
    FROM patients
)
SELECT * 
FROM cost_percentile 
WHERE decile = 10
ORDER BY healthcare_cost DESC;

-- Cost by chronic conditions
SELECT 
    diabetes,
    heart_disease,
    COUNT(*) AS patient_count,
    ROUND(AVG(healthcare_cost), 2) AS avg_cost
FROM patients
GROUP BY diabetes, heart_disease
ORDER BY avg_cost DESC;
