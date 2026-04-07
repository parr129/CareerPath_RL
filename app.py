import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import plotly.express as px

class CareerMobilityEnv:
    def __init__(self, tier=5, init_savings=50.0, init_debt=200.0):
        self.state = np.array([float(tier), 0.0, float(init_savings), float(init_debt), 0.0, 0.0])
        self.months = 0

    def _get_salary(self, tier, collar):
        base = {0: 18, 1: 30, 2: 55, 3: 110, 4: 300}
        multiplier = {1: 2.2, 2: 1.6, 3: 1.3, 4: 1.1, 5: 0.9}
        return base[int(collar)] * multiplier[int(tier)]

    def step(self, action):
        tier, skills, savings, debt, exp, collar = self.state
        self.months += 1
        savings -= 15

        if action == 0:
            savings += self._get_salary(tier, collar)
            exp += 1
        elif action == 1:
            savings -= 30
            skills = min(100, skills + 12)
        elif action == 2:
            savings -= 5
            skills = min(100, skills + 3)
        elif action == 3:
            prob = (skills / 100) * (exp / 12 + 1) * (1 / tier) * 1.8
            if np.random.rand() < prob:
                collar = min(4, collar + 1)

        debt += (debt * 0.007)

        if savings > 30:
            pay = min(savings - 20, debt * 0.1)
            savings -= pay
            debt -= pay

        self.state = np.array([tier, skills, savings, debt, exp, collar])
        done = True if (savings < -200 or collar >= 3 or self.months >= 60) else False
        return self.state, done


class DQNAgent(nn.Module):
    def __init__(self):
        super(DQNAgent, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.network(x)


st.set_page_config(page_title="CareerPath-RL Dashboard", layout="wide")

st.title("CareerPath-RL: Career Mobility Simulator")
st.markdown(
    "This application uses reinforcement learning to simulate career growth decisions "
    "for students across different college tiers, balancing income, skills, and debt."
)

st.sidebar.header("Student Profile")
tier = st.sidebar.selectbox("College Tier", [1, 2, 3, 4, 5], index=4)
start_savings = st.sidebar.slider("Starting Savings (₹k)", 0, 500, 50)
start_debt = st.sidebar.slider("Education Loan (₹k)", 0, 1000, 200)

agent = DQNAgent()
agent.eval()

if st.button("Generate Career Roadmap"):
    env = CareerMobilityEnv(tier=tier, init_savings=start_savings, init_debt=start_debt)
    history = []
    state = env.state
    done = False

    with st.spinner("Calculating optimal strategy..."):
        while not done:
            with torch.no_grad():
                action = agent(torch.FloatTensor(state)).argmax().item()

            act_map = {0: "Work", 1: "Upskill", 2: "Network", 3: "Apply"}

            history.append({
                "Month": env.months + 1,
                "Action": act_map[action],
                "Skills": state[1],
                "Savings (₹k)": state[2],
                "Debt (₹k)": state[3],
                "Collar Level": int(state[5])
            })

            state, done = env.step(action)
            if env.months >= 60:
                break

    df = pd.DataFrame(history)

    col1, col2, col3, col4 = st.columns(4)
    final_row = df.iloc[-1]

    col1.metric("Final Status", "White Collar" if final_row["Collar Level"] >= 3 else "Unsuccessful")
    col2.metric("Time Taken", f"{len(df)} Months")
    col3.metric("Final Savings", f"₹{int(final_row['Savings (₹k)'])}k")
    col4.metric("Max Skills", f"{int(df['Skills'].max())}%")

    st.subheader("Trajectory")

    fig = px.line(
        df,
        x="Month",
        y=["Savings (₹k)", "Debt (₹k)"],
        title="Financial Progress"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.area(
        df,
        x="Month",
        y="Skills",
        title="Skill Growth"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Monthly Roadmap")
    st.table(df)

    st.info(
        f"For this Tier-{tier} profile, the agent primarily focused on "
        f"{df['Action'].value_counts().idxmax()} to achieve career progression."
    )
