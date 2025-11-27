## Notes – NHS Wales Hackathon (Challenge 3: Vaccine Activity)

These are my personal notes from the NHS Wales Modelling Collaborative Hackathon (Nov 2025). I didn’t write the SPC engine code, but I spent time reading through the material shared with the group and trying to understand how the analysis was put together.

## What I Looked At
- How the Autumn 2024 and Spring 2025 datasets were structured  
- How weekly vaccination rates were calculated from the data  
- How the baseline (mean + sigma) was created across regions  
- How z-scores and ±3σ control limits were used  
- How simple SPC run rules highlight unusual behaviour  
- The logistic curve fitting that was used for visual comparisons  
- Differences between Autumn and Spring uptake patterns  

## What I Learned
I mainly focused on understanding how the data was cleaned and reorganised into a Region–Group–Cohort–Week format. Following the baseline calculation (mean + sigma across regions) helped me see how expected behaviour can be estimated using historical campaigns.

Going through the z-score logic, control limits and run rules made it clearer how an SPC-style approach can flag changes in weekly uptake without using a complex model. Overall, this helped me understand the general workflow behind the analysis.

## Summary
These notes reflect what I personally followed and understood during the hackathon. They are not a complete analysis, just my own record of what I managed to learn from the material.
