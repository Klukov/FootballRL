# FootballRL
My master thesis project  
based on: https://github.com/google-research/football
<br/><br/>

**Reward types:**  
- scoring
- scoring,checkpoints -> default

<br/>

**Env Representation:**
  - pixels
  - pixels_gray
  - extracted – 4 layer minimap -> default
  - simple115 / simple115v2

<br/>

**Stacked:**
- True -> default <br/>
Only for: pixels, pixels_gray, extracted
- False  


<br/>

**Possible scenarios:<br/>
(check if model is trained for specific scenario)**
1. 11_vs_11_competition
2. 11_vs_11_stochastic
3. academy_corner
4. academy_empty_goal
5. academy_run_to_score_with_keeper
6. 11_vs_11_easy_stochastic
7. 1_vs_1_easy
8. academy_counterattack_easy
9. academy_pass_and_shoot_with_keeper
10. academy_single_goal_versus_lazy
11. 11_vs_11_hard_stochastic
12. 5_vs_5
13. academy_counterattack_hard
14. academy_run_pass_and_shoot_with_keeper
15. 11_vs_11_kaggle
16. academy_3_vs_1_with_keeper
17. academy_empty_goal_close -> default
18. academy_run_to_score
19. test_example_multiagent

<br/>
