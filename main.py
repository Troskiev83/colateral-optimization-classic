import json
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpContinuous, PULP_CBC_CMD, LpStatusOptimal, value

def run(input_data, solver_params=None, extra_arguments=None):
    """
    Optimize the allocation of financial assets to accounts to minimize collateral cost.

    Parameters:
    - input_data (dict): Contains assets, accounts, haircut_matrix, and limit_matrix.
    - solver_params (dict, optional): Parameters for the solver.
    - extra_arguments (dict, optional): Additional arguments.

    Returns:
    - dict: JSON-compatible dictionary with allocation_matrix and total_collateral_cost.
    """

    # Initialize solver_params and extra_arguments if not provided
    if solver_params is None:
        solver_params = {}
    if extra_arguments is None:
        extra_arguments = {}

    # =============================
    # Parse Input Data
    # =============================
    
    # Assets
    assets = input_data.get("assets", [])
    I = [asset["asset_id"] for asset in assets]
    Q_i = {asset["asset_id"]: asset["available_quantity"] for asset in assets}
    V_i = {asset["asset_id"]: asset["market_value"] for asset in assets}
    omega_i = {asset["asset_id"]: asset["tier_rating"] for asset in assets}
    
    # Accounts
    accounts = input_data.get("accounts", [])
    J = [account["account_id"] for account in accounts]
    C_j = {account["account_id"]: account["collateral_requirement"] for account in accounts}
    
    # Haircut Matrix
    haircut_matrix_data = input_data.get("haircut_matrix", [])
    H_ij = {}
    for entry in haircut_matrix_data:
        asset_id = entry["asset_id"]
        account_id = entry["account_id"]
        haircut = entry["haircut"]
        H_ij[(asset_id, account_id)] = haircut
    
    # Limit Matrix
    limit_matrix_data = input_data.get("limit_matrix", [])
    L_ij = {}
    for entry in limit_matrix_data:
        asset_id = entry["asset_id"]
        account_id = entry["account_id"]
        max_allocation = entry["max_allocation"]
        L_ij[(asset_id, account_id)] = max_allocation

    # =============================
    # Define the MILP Problem
    # =============================
    
    # Initialize the problem
    prob = LpProblem("Collateral_Optimization", LpMinimize)
    
    # Decision Variables: Q_ij >=0
    Q_vars = LpVariable.dicts("Q",
                               [(i, j) for i in I for j in J],
                               lowBound=0,
                               cat=LpContinuous)
    
    # Objective: Minimize total collateral cost
    prob += lpSum(omega_i[i] * Q_vars[(i, j)] * V_i[i] * H_ij.get((i, j), 1) for i in I for j in J), "Total_Collateral_Cost"
    
    # Constraints:
    
    # 1. Collateral Requirements
    for j in J:
        prob += lpSum(Q_vars[(i, j)] * V_i[i] * H_ij.get((i, j), 1) for i in I) >= C_j[j], f"Collateral_Requirement_Account_{j}"
    
    # 2. Asset Quantity Limits
    for i in I:
        prob += lpSum(Q_vars[(i, j)] for j in J) <= Q_i[i], f"Asset_Quantity_Limit_Asset_{i}"
    
    # 3. Allocation Limits
    for i in I:
        for j in J:
            prob += Q_vars[(i, j)] * V_i[i] <= L_ij.get((i, j), float('inf')), f"Allocation_Limit_Asset_{i}_Account_{j}"
    
    # 4. Non-negativity is already handled by variable definition
    
    # Solve the problem
    solver = solver_params.get("solver", PULP_CBC_CMD())
    prob.solve(solver)
    
    # Check if the solution is optimal
    if prob.status != LpStatusOptimal:
        return {
            "output": {
                "allocation_matrix": [],
                "total_collateral_cost": None,
                "status": "No optimal solution found."
            }
        }
    
    # =============================
    # Extract Results
    # =============================
    
    allocation_matrix = []
    for i in I:
        for j in J:
            allocation_fraction = Q_vars[(i, j)].varValue
            allocation_matrix.append({
                "asset_id": i,
                "account_id": j,
                "allocation_fraction": allocation_fraction
            })
    
    total_collateral_cost = value(prob.objective)
    
    # =============================
    # Prepare Output JSON
    # =============================
    
    res = {
        "output": {
            "allocation_matrix": allocation_matrix,
            "total_collateral_cost": total_collateral_cost
        }
    }
    
    return res
