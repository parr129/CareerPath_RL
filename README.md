# CareerPath_RL_Meta_Hackathon
CareerPath-RL: The Tier-5 Socio-Economic Mobility Simulator

Meta PyTorch OpenEnv Hackathon (Round 1 Submission)

Project Vision

In the Indian educational landscape, the concept of “college tier” plays a significant role in shaping career opportunities. Students from lower-tier institutions often face structural disadvantages when competing for high-paying jobs.

CareerPath-RL is designed to model this reality using Reinforcement Learning. It simulates the career trajectory of a Tier-5 graduate and captures the “prestige penalty”—the additional effort required in terms of skill-building, networking, and financial risk-taking to achieve outcomes comparable to graduates from Tier-1 institutions.

Technical Stack
Framework: PyTorch (DQN Architecture)
Environment: OpenEnv / Gymnasium (Custom MDP)
Language: Python 3.10+
Libraries: torch, gymnasium, numpy, matplotlib
Environment Design (The “Rules of India”)

The custom environment, CareerMobilityEnv, models a career as a 60-month Markov Decision Process.

State Space: Tier, Skill Level, Savings, Debt (INR), Experience, Collar Level (0–4)
Reward Logic: Reward shaping is used to balance short-term survival with long-term growth. Actions like “Working” provide immediate income, while “Upskilling” improves future career prospects.
Constraint: A bankruptcy threshold of -₹200,000 forces the agent to carefully manage financial risk while investing in growth.
Performance Analysis: The “Underdog” Strategy

In simulation, the Tier-5 agent converges toward a strategy that reflects realistic career behavior under constraints.

Phase 1: Human Capital Investment (Month 1–2)
The agent prioritizes upskilling despite limited savings (₹5,000), recognizing that without skills, upward mobility is unlikely.

Phase 2: Social Capital Building (Month 3–5)
The agent shifts focus to networking, leveraging referrals to counteract institutional bias.

Phase 3: The Grind and Breakthrough (Month 6–19)
After multiple failed attempts (representing real-world rejections), the agent stabilizes financially through work and eventually transitions to Collar Level 1 around Month 19.

Phase 4: Risk-Aware Survival (Month 20–39)
The agent alternates between earning and applying for better opportunities, managing debt while continuing upward movement. By Month 39, it reaches ₹-183,000 in savings—close to the bankruptcy limit, but still viable.

Key Insight
The agent learns that low-wage work alone is insufficient. Meaningful progress requires calculated risks, particularly investing in skills and networks even at the cost of short-term financial stability.

Learning Progress

Training results show high variance during early episodes due to exploration. Over time, the policy stabilizes and consistently prioritizes early upskilling.

Outcome:
Despite starting with a Tier-5 disadvantage, the agent successfully transitions from unemployment to a specialized role. This demonstrates how reinforcement learning can uncover effective strategies in complex socio-economic systems.

How to Run
Open CareerPath_RL.ipynb in Google Colab
Run the environment setup and training cells
Review the final strategy log to observe the agent’s month-by-month career decisions
Significance for Meta OpenEnv
Agentic AI: The system models long-term decision-making rather than simple prediction
Social Relevance: It highlights structural challenges faced by students from lower-tier institutions
Technical Strength: Implements a clean and modular DQN architecture aligned with modern PyTorch and OpenEnv practices

Developed by: TEAM ECHO
Category: Reinforcement Learning / AI Agents
Location: India (Targeted for the Bangalore Finale)
