import json
from .storage import PROJECT

CATALOG = [
    dict(
        id="production",
        title="生产计划 · 整数决策",
        category="MILP",
        description="两种产品共享两类资源，决定整数产量，使总收益最大。",
        problem_text="生产 x、y 两种产品，每件收益分别为 3、5。两类资源总量均为 4；每件 x 消耗 2、1，每件 y 消耗 1、2。产量为非负整数，最大化总收益。",
        hint="比较整数模型与连续松弛的结果，观察变量类型如何影响可行域。",
    ),
    dict(
        id="production_lp",
        title="生产计划 · 连续松弛",
        category="LP",
        description="允许产量连续变化，观察线性规划与整数规划的差别。",
        problem_text="最大化 3x+5y，满足 2x+y≤4、x+2y≤4，x、y 为非负连续变量。",
        hint="本例允许分数产量；是否符合实际题意需要自行判断。",
    ),
    dict(
        id="assignment",
        title="3 × 3 任务指派",
        category="MILP · 索引集合",
        description="将三位执行者分配到三个任务，一人一项，最小化总成本。",
        problem_text="执行者 A、B、C 与任务 P、Q、R 一一指派。成本矩阵为 [[9,2,7],[6,4,3],[5,8,1]]，求总成本最小的指派方案。",
        hint="练习字符串索引、二元变量、参数矩阵及嵌套求和。",
    ),
    dict(
        id="facility_location",
        title="仓库设施选址 (Facility Location)",
        category="MILP · 选址与固定成本",
        description="决定启用哪些候选仓库及各仓库向客户的供货量，最小化固定建设费与运输总成本。",
        problem_text="有 3 个候选仓库 W1..W3 与 4 个客户 C1..C4。各仓库有建设固定成本 f[w] 与最大供应能力 cap[w]，运输单价为 c[w,k]，客户需求为 d[k]。决定建哪些仓库及如何配送，最小化总成本。",
        hint="参考 Gurobi 经典设施选址模型：二元决策变量 y[w] 控制产能上限 cap[w]*y[w]。",
    ),
    dict(
        id="blending_oil",
        title="原油配料混合优化 (Blending)",
        category="LP · 成分比例",
        description="混合三种不同品质与价格的原料油，在满足产量、辛烷值与含硫量指标下最小化成本。",
        problem_text="采购 Crude1、Crude2、Crude3 三种原料油混合生产 1000 吨成品油。要求加权平均辛烷值不低于 90、含硫量不超过 1.5%，且不超过各自供应上限，最小化总采购成本。",
        hint="参考 Gurobi 经典配料模型：学习如何把质量比例约束线性化为线性规划。",
    ),
    dict(
        id="workforce_shift",
        title="员工周班次排班 (Workforce)",
        category="MILP · 整数排班",
        description="在“做五休二”的 7 种轮班制度下，确定各班次雇佣人数以满足每天最低在岗需求。",
        problem_text="一周 7 天每日最低在岗需求分别为 14, 13, 15, 16, 19, 18, 11 人。员工可选择 7 种连续工作 5 天、休息 2 天的班次，各班次周薪不同，求总人力成本最小的整数排班方案。",
        hint="参考 Gurobi 经典排班模型：使用 0-1 覆盖矩阵 a[s,d] 表达班次与在岗日期的对应关系。",
    ),
    dict(
        id="lot_sizing",
        title="多周期生产与库存规划 (Lot Sizing)",
        category="MILP · 动态库存",
        description="在 4 个生产周期内权衡开机准备成本、生产成本与库存持有费，满足各期需求。",
        problem_text="计划期包含 T1..T4 四个周期，各期需求分别为 40, 60, 90, 50 件（其中 T3 产能仅 80 件）。每期开机需固定准备费 500 元，单位生产成本与库存持有费已知，求总成本最低的生产与库存方案。",
        hint="参考 Gurobi 多周期批量问题：结合 Big-M 启停约束与跨期库存结转平衡。",
    ),
    dict(
        id="free_variable",
        title="自由变量与负值解",
        category="LP · 变量范围",
        description="理解显式自由变量：最优值可以为负数。",
        problem_text="最小化 x，约束 x≥−3。x 为连续变量，声明范围为双向无界。",
        hint="比较显式自由变量与人为增加 x≥0 后的不同模型。",
    ),
    dict(
        id="infeasible",
        title="识别不可行模型",
        category="LP · 状态",
        description="两个互相矛盾的约束，没有共同可行解。",
        problem_text="最小化 x，满足 x≥2 与 x≤1。x 为连续自由变量。",
        hint="不可行是该模型的数学结果；平台不会调换不等号。",
    ),
    dict(
        id="unbounded",
        title="理解无界结果",
        category="LP · 状态",
        description="缺少限制目标改善的约束时，可能不存在有限最优值。",
        problem_text="最小化连续自由变量 x，不添加任何约束。",
        hint="“无界”不等于“未运行”，也不等于目标值为零。",
    ),
]


def example(id):
    item = next((e for e in CATALOG if e["id"] == id), None)
    if not item:
        raise KeyError(id)
    model = json.loads((PROJECT / "fixtures" / f"{id}.json").read_text(encoding="utf-8"))
    data = (
        json.loads(
            (PROJECT / "fixtures" / "assignment_data.json").read_text(encoding="utf-8")
        )
        if id == "assignment"
        else {}
    )
    return dict(**item, model=model, data=data)
