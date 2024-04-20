module EvolutionaryAlgorithm

export evolutionary_algorithm
export ea_wrapper
export Instance
export EAConfig

using JuMP
using Graphs
using Gurobi
using Random

include("./utils.jl")
using .UtilityFunctions

#########################################
# EA objects
#########################################

struct Instance
    supply::Vector{Int}
    demand::Vector{Int}
    var_costs::Matrix{Float64}
    fix_costs::Matrix{Float64}
    edge_mask::Matrix{Bool}
end

Instance(supply, demand, var_costs, fix_costs) = Instance(
    supply,
    demand,
    var_costs,
    fix_costs,
    fill(true, size(var_costs))
)

struct EAConfig
    pop_size::Int
    max_unique_sols::Int
    patience::Int
    mutation_operator::String
    seed::Int
    log_file::String
end

EAConfig(pop_size, max_unique_sols, patience, mutation_operator, seed) = EAConfig(
    pop_size,
    max_unique_sols,
    patience,
    mutation_operator,
    seed,
    ""
)

#########################################
# EA sub-functions
#########################################

""" Convert solution into a basic feasible solution with m+n-1 edges (spanning tree). """
function fill_basis(sol, instance)
    m = length(instance.supply)
    n = length(instance.demand)
    target_size = m + n - 1

    if length(sol) == target_size
        return sol
    end

    # Collect candidate edges
    nbv::Vector{Tuple{Int,Int}} = []
    for i in 1:m, j in 1:n
        if !((i, j) in keys(sol)) && instance.edge_mask[i, j]
            push!(nbv, (i, j))
        end
    end

    # Add candidate edges while making sure that spanning tree property is not violated (no cycles)
    while true
        bfs = copy(sol)
        spanning_tree_graph = SimpleGraph(Edge.([(i, j + m) for (i, j) in keys(sol)]))

        # Add zero-variables into the basis if they do not create a cycle until the target size has been reached
        for k in nbv
            # Tentatively add variable to basis and check whether a cycle is formed
            add_edge!(spanning_tree_graph, k[1], k[2] + m)
            cycle = cycle_basis(spanning_tree_graph)
            if !isempty(cycle)
                rem_edge!(spanning_tree_graph, k[1], k[2] + m)
                continue # If a cycle would result, remove edge and continue with next variable
            else
                bfs[k] = 0 # Add variable to BFS and graph
                if length(bfs) == target_size
                    return bfs
                end
            end
        end

        # If unsuccessful, add NBVs in a different sequence
        shuffle(nbv)
    end
end

""" Decode edge-priorities into a valid FCTP solution. """
function decode(chromosome::Vector{Int}, instance::Instance)
    m = length(instance.supply)
    n = length(instance.demand)

    model = Model(Gurobi.Optimizer)
    set_silent(model)

    @variable(model, x[i in 1:m, j in 1:n] >= 0)
    for i in 1:m, j in 1:n
        if !instance.edge_mask[i, j]
            fix(x[i, j], 0.0; force=true)
        end
    end

    # Fullfill supply and demand constraints
    @constraint(model, [i in 1:m], sum(x[i, :]) <= instance.supply[i])
    @constraint(model, [j in 1:n], sum(x[:, j]) == instance.demand[j])

    # Maximize total "priority"
    @objective(
        model,
        Max,
        sum(chromosome[(i-1)*m+j] * x[i, j] for i in 1:m, j in 1:n),
    )

    optimize!(model)

    sol = Dict([(i, j) => Int(value(x[i, j])) for i in 1:m, j in 1:n if !(isapprox(value(x[i, j]), 0.0; atol=1e-6))])

    return sol
end


""" Create initial population of randomly generated BFS. """
function get_initial_population(instance::Instance, config::EAConfig)
    m = length(instance.supply)
    n = length(instance.demand)

    unique_sols = Set{UInt64}() # Solution hashes
    population = Vector{Dict{Tuple{Int,Int},Int}}() # Basic feasible solutions

    while length(population) < config.pop_size
        candidate = decode(randperm(m * n), instance)
        if !(candidate in unique_sols)
            push!(unique_sols, hash(candidate))
            push!(population, fill_basis(candidate, instance))
        end
    end

    return population, unique_sols
end

""" Find loop containing entering variable. """
function get_loop(sol_vars, var_in)
    m = maximum([i for (i, j) in sol_vars])
    g = SimpleGraph(Edge.([(i, j + m) for (i, j) in sol_vars]))
    add_edge!(g, var_in[1], var_in[2] + m)
    cycle = cycle_basis(g, var_in[1])[1]
    # convert sequence of nodes into sequence of edges and reorder to start with entering edge
    loop = node_cycle_to_edge_loop(cycle, m)
    return reorder_loop(loop, var_in)
end

""" Perform pivot. """
function perform_pivot(
    bfs::Dict{Tuple{Int,Int},Int},
    loop::AbstractVector{Tuple{Int,Int}}
)
    even_cells = loop[1:2:end]
    odd_cells = loop[2:2:end]

    # Find the leaving variable and value
    min_val, min_index = findmin(bfs[p] for p in odd_cells)
    leaving_var = odd_cells[min_index]

    new_bfs = copy(bfs)

    # Update even cells
    for p in even_cells
        if p in keys(new_bfs)
            new_bfs[p] += min_val # update
        else
            new_bfs[p] = min_val # add entering variable
        end
    end

    # Update odd cells
    for p in odd_cells
        new_bfs[p] -= min_val
    end

    # Remove leaving variable
    delete!(new_bfs, leaving_var)
    return new_bfs
end

""" Randomly select a non-basic variable and pivot it into the BFS. """
function eicr_mutation(parent, instance)
    # Randomly select an edge that is not part of the parent
    m = maximum([i for (i, _) in keys(parent)])
    n = maximum([j for (_, j) in keys(parent)])
    var_in = [(i, j) for i in 1:m for j in 1:n if instance.edge_mask[i, j] && !((i, j) in keys(parent))][rand(1:end)]
    # Perform pivot operation
    loop = get_loop(keys(parent), var_in)
    offspring = perform_pivot(parent, loop)
    return offspring
end

""" Select a promising non-basic variable among a few candidates and pivot it into the BFS. """
function nlo_mutation(parent, instance)
    m = maximum([i for (i, _) in keys(parent)])
    n = maximum([j for (_, j) in keys(parent)])

    # Randomly select a node
    node = rand(1:m+n)

    # Identify incident edges
    if node <= m
        i = node
        vars_in = [(i, j) for j in 1:n if instance.edge_mask[i, j] && !((i, j) in keys(parent))]
    else
        j = node - m
        vars_in = [(i, j) for i in 1:m if instance.edge_mask[i, j] && !((i, j) in keys(parent))]
    end

    # Evaluate their potential insertion and select the best
    best_candidate = Dict{Tuple{Int,Int},Int}()
    best_val = Inf64
    for var_in in vars_in
        loop = get_loop(keys(parent), var_in)
        offspring = perform_pivot(parent, loop)
        sol_val = evaluate_sol(offspring, instance.var_costs, instance.fix_costs)
        if sol_val < best_val
            best_val = sol_val
            best_candidate = offspring
        end
    end

    return best_candidate
end

""" Pivot edges from one parent into another. """
function dcx_crossover(parent1, parent2)
    offspring = Dict{Tuple{Int,Int},Int}(parent1)

    # Identify edges which are part of parent2 but not of parent 1
    disjoint_edges = collect(filter(edge -> !(edge in keys(parent1)), keys(parent2)))

    if !isempty(disjoint_edges)
        # Randomly pick half of them for sequential insertion
        vars_in = shuffle(disjoint_edges)[1:floor(Int, 0.5 * length(disjoint_edges))]
        for var_in in vars_in
            if var_in in keys(offspring)
                continue
            end
            # Perform pivot
            loop = get_loop(keys(offspring), var_in)
            offspring = perform_pivot(offspring, loop)
        end
    end

    return offspring
end

""" Randomly select two elements form population and return the better one. """
function tournament_selection(population, pop_perf)
    cand1 = rand(1:length(population))
    cand2 = rand(1:length(population))
    return pop_perf[cand1] < pop_perf[cand2] ? population[cand1] : population[cand2]
end

""" Only return non-zero elements from BFS. """
function strip_bfs(bfs)
    return Dict(filter(kv -> kv[2] > 0, bfs))
end

""" Evolutionary Algorithm. """
function evolutionary_algorithm(instance::Instance, config::EAConfig)
    Random.seed!(config.seed)
    runtime = @elapsed begin
        # Setup
        logging = (isempty(config.log_file) == false)

        if config.mutation_operator == "nlo"
            mutation_fun = nlo_mutation
        else
            mutation_fun = eicr_mutation
        end

        # STEP 1: Create and evaluate initial population
        println("Generating initial population")
        population, unique_sols_visited = get_initial_population(instance, config)
        solutions = [strip_bfs(indiv) for indiv in population] # solutions without zero-edges
        pop_perf = [
            evaluate_sol(sol, instance.var_costs, instance.fix_costs)
            for sol in solutions
        ]
        best_candidate = argmin(pop_perf)
        best_sol = solutions[best_candidate]
        best_sol_value = pop_perf[best_candidate]

        # Run until a certain number of unique solutions has been obtained
        it = 0
        best_it = 0
        start_time = time()
        # Log best solution
        if logging
            write_to_log(config.log_file, "$(0.0),$(best_sol_value)")
        end
        println("Starting iterations")
        while length(unique_sols_visited) < config.max_unique_sols
            # Early stopping
            if it - best_it > config.patience
                println("No improvement since iteration $(best_it) -> Stopping ($(length(unique_sols_visited)) unique solutions visited)")
                break
            end

            it += 1
            if mod(it, 10000) == 0
                println("Iteration $(it) ($(length(unique_sols_visited)) unique solutions visited)")
            end

            # STEP 2: Create offsprings via crossover and mutation
            # Crossover
            parent1 = tournament_selection(population, pop_perf)
            parent2 = tournament_selection(population, pop_perf)
            offspring = dcx_crossover(parent1, parent2)

            # Mutation
            offspring = mutation_fun(offspring, instance)

            if isempty(offspring)
                continue
            end

            # Check if solution is already contained in population solutions
            solution = strip_bfs(offspring)
            if solution in solutions
                continue
            end

            # Evaluate performance
            perf = evaluate_sol(solution, instance.var_costs, instance.fix_costs)
            if perf < best_sol_value
                best_sol_value = perf
                best_sol = solution
                best_it = it
                if logging
                    write_to_log(config.log_file, "$(time() - start_time), $(best_sol_value)")
                end
                println(String("New best solution $(best_sol_value)"))
            end

            # Identify worst individuum in population and replace it by new candidate
            worst_indiv = argmax(pop_perf)
            population[worst_indiv] = offspring
            solutions[worst_indiv] = solution
            pop_perf[worst_indiv] = perf

            # Add new solution to pool of visited solutions
            push!(unique_sols_visited, hash(solution))
        end
        if logging
            write_to_log(config.log_file, "$(time() - start_time), $(best_sol_value)")
        end
    end
    return (best_sol, best_sol_value, runtime)
end


function ea_wrapper(supply, demand, var_costs, fix_costs, edge_mask, ea_config)
    instance = Instance(supply, demand, var_costs, fix_costs, edge_mask)
    config = EAConfig(
        get!(ea_config, "pop_size", 100),
        get!(ea_config, "max_unique_sols", 100000),
        get!(ea_config, "patience", 100000),
        get!(ea_config, "mutation_operator", "eicr"),
        get!(ea_config, "seed", 0),
        get!(ea_config, "log_file", ""),
    )
    sol, sol_val, runtime = evolutionary_algorithm(instance, config)
    return (sol_dict_to_matrix(sol, length(supply), length(demand)), sol_val, runtime)
end

end
