""" Analyze existence of multiple optimal solutions (Table 9). """

import os

import numpy as np

from core.data_processing.data_utils import load_instance
from core.utils.ml_utils import load_edge_predictor_model
from core.ml_models.wrapper import (
    sol_edge_predictor_wrapper,
)
from core.ml_models.wrapper import (
    ml_based_fctp_reduction,
)
from core.fctp_solvers.ip_grb import fctp
from core.fctp_solvers.ip_grb import capacitated_fctp
from core.fctp_solvers.ip_grb import fixed_step_fctp
from core.utils.fctp import FixedStepFCTP
from core.utils.fctp import CapacitatedFCTP


if __name__ == "__main__":
    instance_dir = "data_paper/instances/benchmarking"
    model_dir = "trained_models/sol_edge_predictor"
    model_spec = "gcnn/model_gcnn_features_bipartite_raw_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_20_num_conv_layers_10_num_dense_layers_2"

    datastes = {
        "BASE": "fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.0",
        "B50": "fctp_agarwal-aneja_15_15_B50_Theta0.2_BF1.0",
        "UB10%": "fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.1",
        "UB30%": "fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.3",
        "RBM15": "fctp_roberti_15_15_B20_Theta0.2_BF1.0",
        "EUCL": "fctp_euclidian_15_15_B20_Theta0.2_BF1.0",
        "VFCR0.5": "fctp_agarwal-aneja_15_15_B20_Theta0.5_BF1.0",
        "VFCR0.0": "fctp_agarwal-aneja_15_15_B20_Theta0.0_BF1.0",
        "C-FCTP": "c-fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.0",
        "FS-FCTP": "fs-fctp_agarwal-aneja_15_15_B20_Theta0.2_BF1.0",
    }
    instance_dirs = {k: os.path.join(instance_dir, v) for k, v in datastes.items()}
    model_paths = {
        k: os.path.join(
            model_dir, v, model_spec, "cross_val", "best_checkpoint.pth.tar"
        )
        for k, v in datastes.items()
    }

    for dataset, instance_dir in instance_dirs.items():

        print("+++++++++++++++++++++++++++++++++++++")
        print(f"Evaluating data set {dataset}")
        print("+++++++++++++++++++++++++++++++++++++")

        predictor_model = load_edge_predictor_model(model_paths[dataset])

        instances = os.listdir(instance_dir)
        num_sol_edges = []
        num_opt_solutions = []
        avg_probability = []
        avg_rank = []
        num_edges_needed_to_cover_sol = []
        opt_gaps = []
        index_multi_sols = []
        for idx, instance_name in enumerate(instances):
            instance_id = instance_name.split(".")[0].split("_")[-1]
            # if instance_id != "5":
            #     continue
            print(f"Evaluating instance {instance_id}...")

            # Load instance
            instance_path = os.path.join(instance_dir, instance_name)
            instance = load_instance(instance_path)

            # Step 1: Solve with SolutionPool to find all optimal solutions
            if isinstance(instance, CapacitatedFCTP):
                model, x, _ = capacitated_fctp(
                    instance.supply,
                    instance.demand,
                    instance.var_costs,
                    instance.fix_costs,
                    instance.edge_capacities,
                )
            elif isinstance(instance, FixedStepFCTP):
                model, x, _ = fixed_step_fctp(
                    instance.supply,
                    instance.demand,
                    instance.var_costs,
                    instance.fix_costs,
                    instance.vehicle_capacities,
                )
            else:
                model, x, _ = fctp(
                    instance.supply,
                    instance.demand,
                    instance.var_costs,
                    instance.fix_costs,
                )
            model.setParam("OutputFlag", 0)
            model.setParam("Seed", 0)
            model.setParam("PoolSearchMode", 2)
            model.setParam("PoolSolutions", 5)
            model.setParam("PoolGap", 0)
            model.setParam("TimeLimit", 600)
            model.optimize()
            print(f"Model status: {model.status}")
            opt_val = model.ObjVal
            print(f"Opt val: {opt_val}")
            print(f"Found {model.SolCount} optimal solutions")
            num_opt_solutions.append(model.SolCount)

            # Step 2: Analyze correspondence of predictions and solution(s)
            edge_likelihood = sol_edge_predictor_wrapper(instance, predictor_model)

            num_sol_edges_k = []
            avg_probability_k = []
            avg_rank_k = []
            num_edges_needed_to_cover_sol_k = []
            # Evaluate every solution
            for c in range(model.SolCount):
                num_edges = int(len(model.getVars()) / 2)
                edge_index = np.unravel_index(
                    np.arange(0, num_edges), instance.var_costs.shape
                )
                edge_index = [
                    (edge_index[0][i], edge_index[1][i]) for i in range(num_edges)
                ]

                # Retrieve solution
                if c == 0:
                    x = np.round(model.X).astype(int)[num_edges:]  # only y
                else:
                    model.setParam("SolutionNumber", c)
                    x = np.round(model.Xn).astype(int)[num_edges:]
                edges_x = [edge_index[i] for i, v in enumerate(x) if v > 0]
                num_sol_edges_k.append(len(edges_x))

                # Evaluate support
                support_for_x = [edge_likelihood[e] for e in edges_x]
                print(
                    f"Average probability of solution edges: {np.mean(support_for_x)} (overall mean: {np.mean(edge_likelihood)})"
                )
                avg_probability_k.append(np.mean(support_for_x))

                # Get edge ranking
                ranking = np.unravel_index(
                    np.flip(np.argsort(edge_likelihood.flatten())),
                    edge_likelihood.shape,
                )
                edge_ranks = np.zeros(edge_likelihood.shape, dtype=int)
                for i in range(len(ranking[0])):
                    edge_ranks[ranking[0][i], ranking[1][i]] = i + 1

                ranks_x = [edge_ranks[e] for e in edges_x]
                print(f"Average rank of solution edges: {np.mean(ranks_x)}")
                print(
                    f"Number of edges needed to cover solution (= max rank): {np.max(ranks_x)}"
                )
                avg_rank_k.append(np.mean(ranks_x))
                num_edges_needed_to_cover_sol_k.append(np.max(ranks_x))

            avg_probability.append(avg_probability_k)
            avg_rank.append(avg_rank_k)
            num_edges_needed_to_cover_sol.append(num_edges_needed_to_cover_sol_k)
            num_sol_edges.append(num_sol_edges_k)

            # Compare solutions 1 and 2 (assuming 2 optimal solutions)
            if model.SolCount > 1:
                index_multi_sols.append(idx)

                x1 = np.round(model.X).astype(int)[num_edges:]  # only y
                edges_x1 = [edge_index[i] for i, v in enumerate(x1) if v > 0]

                model.setParam("SolutionNumber", 1)
                x2 = np.round(model.Xn).astype(int)[num_edges:]
                edges_x2 = [edge_index[i] for i, v in enumerate(x2) if v > 0]

                joint_edges = set(edges_x1).intersection(set(edges_x2))
                edges_x1_not_x2 = list(set(edges_x1) - set(edges_x2))
                edges_x2_not_x1 = list(set(edges_x2) - set(edges_x1))
                print(f"Solution 1 and 2 contain {len(joint_edges)} joint edges.")
                print(
                    f"Solution 1 contains {len(edges_x1_not_x2)} edges that solution 2 does not (Total number of edges: {len(edges_x1)})"
                )
                print(
                    f"Solution 2 contains {len(edges_x2_not_x1)} edges that solution 1 does not (Total number of edges: {len(edges_x2)})"
                )

            # Step 3: Evaluate reduce-then-optimize performance
            (
                sol,
                _,
                _,
                _,
                _,
                _,
            ) = ml_based_fctp_reduction(
                instance,
                predictor_model=predictor_model,
                threshold_type="size",
                threshold=0.3,
                decoder="exact",
            )
            sol_val = instance.eval_sol_dict(sol)
            opt_gaps.append(sol_val / opt_val - 1)

        ##################################
        # Summary and Evaluation
        ##################################
        print(
            f"Occurance of multiple solutions: {len(index_multi_sols)} ({len(index_multi_sols)/len(num_sol_edges)*100}%)"
        )

        # Evaluating instances with multiple optimal solutions
        for idx in index_multi_sols:
            print(f"Evaluating {instances[idx].split('.')[0].split('_')[-1]}")
            print(f"Avg. number of solution edges: {num_sol_edges[idx]}")
            print(f"Avg. probability: {avg_probability[idx]}")
            print(f"Avg. rank: {avg_rank[idx]}")
            print(f"Maximum rank: {num_edges_needed_to_cover_sol[idx]}")
            print(f"Optimality gap: {opt_gaps[idx]}")

        # Comparison to instances with only one optimal solution
        print(
            f"Avg. number of solution edges for instances with unique solutions: {np.mean([v for i,v in enumerate(num_sol_edges) if i not in index_multi_sols])}"
        )
        print(
            f"Avg. probability for instances with unique solutions: {np.mean([v for i,v in enumerate(avg_probability) if i not in index_multi_sols])}"
        )
        print(
            f"Avg. rank for instances with unique solutions: {np.mean([v for i,v in enumerate(avg_rank) if i not in index_multi_sols])}"
        )
        print(
            f"Avg. maximum rank for instances with unique solutions: {np.mean([v for i,v in enumerate(num_edges_needed_to_cover_sol) if i not in index_multi_sols])}"
        )
        print(
            f"Avg. optimality gap for instances with unique solutions: {np.mean([v for i,v in enumerate(opt_gaps) if i not in index_multi_sols])}"
        )
