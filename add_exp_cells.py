import json

with open('mvp.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = [
    # Cell: markdown - Experiment description
    {
        "cell_type": "markdown",
        "id": "exp_md_1",
        "metadata": {},
        "source": [
            "## 模块4：行为差异验证\n",
            "\n",
            "### 核心假设\n",
            "玩家通过自定义奖励函数，能塑造出行为差异化的智能体。\n",
            "\n",
            "### 实验设计\n",
            "用两组极端奖励权重分别训练，对比行为指标。\n",
            "\n",
            "| 配置 | damage_dealt | damage_taken | survival | distance | kill |\n",
            "|------|-------------|-------------|----------|----------|------|\n",
            "| 激进型 | +2.0 | -0.1 | 0 | 0 | +10.0 |\n",
            "| 保守型 | +0.5 | -5.0 | +1.0 | -0.5 | +2.0 |\n",
            "\n",
            "### 验证指标\n",
            "- 平均每局存活步数\n",
            "- 平均每局造成/受到伤害\n",
            "- 平均与敌方距离\n",
            "- 平均每局攻击次数\n",
            "- 胜率\n",
            "- 训练曲线收敛性\n",
            "\n",
            "### 通过标准\n",
            "两组在关键指标上有统计显著差异（p < 0.05），行为模式肉眼可辨识。\n"
        ]
    },
    # Cell: code - Quick experiment
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "exp_code_1",
        "metadata": {},
        "outputs": [],
        "source": [
            "from ppo_trainer import (\n",
            "    run_training, evaluate_agent, compare_agents,\n",
            "    load_actor_for_eval, run_behavior_experiment,\n",
            ")\n",
            "\n",
            "# 快速验证：少量步数训练两组并对比\n",
            "agg_summary, con_summary, p_vals = run_behavior_experiment(\n",
            "    ArenaEnv,\n",
            "    total_steps=20480,    # 快速测试 ~10 iterations\n",
            "    hidden_dim=128,\n",
            "    eval_episodes=50,     # 50 局评估\n",
            ")"
        ]
    },
    # Cell: code - Full experiment (commented)
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "exp_code_2",
        "metadata": {},
        "outputs": [],
        "source": [
            "# 完整实验：100K 步训练 + 100 局评估\n",
            "# agg_summary, con_summary, p_vals = run_behavior_experiment(\n",
            "#     ArenaEnv,\n",
            "#     total_steps=100_000,\n",
            "#     hidden_dim=128,\n",
            "#     eval_episodes=100,\n",
            "# )"
        ]
    },
]

nb['cells'].extend(new_cells)

with open('mvp.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Added {len(new_cells)} experiment cells')
