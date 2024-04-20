module TabuSearch

export tabu_search
export ts_wrapper
export Instance
export TabuSearchConfig
export get_alpha

using Random
using Graphs

include("./utils.jl")
using .UtilityFunctions

####################################
# Pivot functions
####################################

""" Find loop containing entering variable. """
function get_loop(bfs_vars, entering_var)
    m = maximum([i for (i, j) in bfs_vars])
    g = SimpleGraph(Edge.([(i, j + m) for (i, j) in bfs_vars]))
    add_edge!(g, entering_var[1], entering_var[2] + m)
    cycle = cycle_basis(g, entering_var[1])[1]
    # convert sequence of nodes into sequence of edges and reorder to start with entering edge
    loop = node_cycle_to_edge_loop(cycle, m)
    return reorder_loop(loop, entering_var)
end

""" Perform pivot. """
function get_pivot(
    bfs::Dict{Tuple{Int,Int},Int},
    loop::AbstractVector{Tuple{Int,Int}},
    fix_costs::AbstractMatrix{Float64}
)
    even_cells = loop[1:2:end]
    odd_cells = loop[2:2:end]

    # Find the minimum value and its indices
    odd_cell_vals = [bfs[k] for k in odd_cells]
    min_val = minimum(odd_cell_vals)
    min_val_indices = findall(x -> x == min_val, odd_cell_vals)

    # If there is only one minimum value, choose it
    if length(min_val_indices) == 1
        leaving_var = odd_cells[min_val_indices[1]]
    else
        # If there are multiple minimum values, choose the one with the lowest fix cost
        min_fix_costs = [fix_costs[i, j] for (i, j) in odd_cells[min_val_indices]]
        min_fix_cost_index = argmin(min_fix_costs)
        leaving_var = odd_cells[min_val_indices[min_fix_cost_index]]
    end

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

    return new_bfs, leaving_var, min_val
end

function get_dual_vars(bfs::Dict{Tuple{Int,Int},Int}, var_costs::AbstractMatrix{Float64})
    m, n = size(var_costs)
    coeff = zeros(Int, m + n, m + n)
    b = zeros(m + n)
    for (r, (i, j)) in enumerate(keys(bfs))
        coeff[r, i] = 1
        coeff[r, m+j] = 1
        b[r] = var_costs[i, j]
    end
    coeff[m+n, 1] = 1
    dual_vals = coeff \ b
    return (dual_vals[1:m], dual_vals[m+1:end])
end

function get_net_change(
    entering_var::Tuple{Int,Int},
    var_costs::AbstractMatrix{Float64},
    fix_costs::AbstractMatrix{Float64},
    bfs::Dict{Tuple{Int,Int},Int},
    u::AbstractVector{Float64},
    v::AbstractVector{Float64}
)
    loop = get_loop(keys(bfs), entering_var)
    new_bfs, leaving_var, min_val = get_pivot(bfs, loop, fix_costs)

    if min_val == 0
        delta = 0.0
    else
        even_cells = loop[1:2:end]
        odd_cells = loop[2:2:end]
        U1 = [k for k in odd_cells if bfs[k] == min_val]
        U2 = [k for k in even_cells[2:end] if bfs[k] == 0]
        delta = (
            min_val
            *
            (var_costs[entering_var...] - u[entering_var[1]] - v[entering_var[2]])
            +
            fix_costs[entering_var...]
            -
            sum([fix_costs[k...] for k in U1])
            +
            sum([fix_costs[k...] for k in U2])
        )
    end

    return delta, new_bfs, leaving_var
end

#########################################
# TS objects
#########################################

struct Instance{A<:AbstractVector{Int},B<:AbstractMatrix{Float64},C<:AbstractMatrix{Bool}}
    supply::A
    demand::A
    var_costs::B
    fix_costs::B
    edge_mask::C
end

Instance(supply, demand, var_costs, fix_costs) = Instance(
    supply,
    demand,
    var_costs,
    fix_costs,
    fill(true, size(var_costs))
)

mutable struct TabuSearchConfig
    tabu_in_range::Tuple{Int,Int}
    tabu_out_range::Tuple{Int,Int}
    tabu_in::Int
    tabu_out::Int
    beta::Float64
    gamma::Float64
    alpha::Float64
    L::Int
    seed::Int
    log_file::String

    function TabuSearchConfig(tabu_in_range, tabu_out_range, beta, gamma, alpha, L, seed, log_file)
        new(
            tabu_in_range,
            tabu_out_range,
            sample_tabu_val(tabu_in_range),
            sample_tabu_val(tabu_out_range),
            beta,
            gamma,
            alpha,
            L,
            seed,
            log_file
        )
    end

end

TabuSearchConfig(tabu_in_range, tabu_out_range, beta, gamma, alpha, L, seed) = TabuSearchConfig(
    tabu_in_range,
    tabu_out_range,
    beta,
    gamma,
    alpha,
    L,
    seed,
    ""
)

mutable struct TabuSearchState
    k::Int
    l::Int
    l_1::Bool
    h::Matrix{Int}
    z::Matrix{Float64}
    t::Matrix{Int}
    # current basis / search position
    i::Int
    z_val::Float64
    i_0::Int
    j_0::Int
    bfs::Dict{Tuple{Int,Int},Int}
    u::Vector{Float64}
    v::Vector{Float64}
    # best solution since last longterm memory process
    z_min::Float64
    k_min::Int
    # best solution found
    z_best::Float64
    best_sol::Dict{Tuple{Int,Int},Int}
    # current process / step
    intermediate_mem_1::Bool
    longterm_mem::Bool
    _next::Function

    function TabuSearchState(m, n, bfs, bfs_val, u, v)
        new(
            0, #k
            0, #l
            false, #l1
            zeros(Int, m, n), #h
            fill(bfs_val, m, n), #z
            fill(typemin(Int), m, n), #t
            0, #i
            bfs_val, #z_val
            0, #i_0
            0, #j_0
            copy(bfs), #bfs
            u, #u
            v, #v
            bfs_val, #z_min
            0, #k_min
            bfs_val, #z_best
            copy(bfs), #best_sol
            false, #intermediate_mem_1
            false, # longterm_mem
            step_1
        )
    end

end

function get_alpha(
    bfs::Dict{Tuple{Int,Int},Int},
    instance::Instance
)
    m = length(instance.supply)
    n = length(instance.demand)
    (u, v) = get_dual_vars(bfs, instance.var_costs)
    deltas = Array{Float64}(undef, 0)
    nbv = [(i, j) for i in collect(1:m) for j in collect(1:n) if !((i, j) in keys(bfs))]
    for entering_var in nbv
        # delta, _, _ = get_net_change(entering_var, var_costs, fix_costs, bfs, u, v)
        delta, _, _ = get_net_change(entering_var, instance.var_costs, instance.fix_costs, bfs, u, v)
        push!(deltas, delta)
    end
    alpha = (1 / (m * n - m - n)) * sum(deltas)
    return alpha
end

function sample_tabu_val(tabu_val_range)
    return rand(tabu_val_range[1]:tabu_val_range[2])
end

function reset_tabu_vals(config::TabuSearchConfig)
    config.tabu_in = sample_tabu_val(config.tabu_in_range)
    config.tabu_out = sample_tabu_val(config.tabu_out_range)
end

#########################################
# TS algorithm
#########################################

function get_candidate_list(state::TabuSearchState, instance::Instance, only_nbv::Bool)
    cand_list = [(state.i, j) for j in 1:length(instance.demand) if instance.edge_mask[state.i, j]]
    if only_nbv
        cand_list = [c for c in cand_list if !(c in keys(state.bfs))]
    end
    return cand_list
end

function check_tabu_status(entering_var::Tuple{Int,Int}, leaving_var::Tuple{Int,Int}, state::TabuSearchState, config::TabuSearchConfig)
    if state.k - state.t[entering_var...] <= config.tabu_in
        return false
    elseif state.k - state.t[leaving_var...] <= config.tabu_out
        return false
    else
        return true
    end
end

function check_aspiration(
    entering_var::Tuple{Int,Int},
    leaving_var::Tuple{Int,Int},
    delta::Float64,
    state::TabuSearchState
)
    if state.z_val + delta > state.z[entering_var...]
        return false
    elseif state.z_val + delta > state.z[leaving_var...]
        return false
    else
        return true
    end
end

function get_intermediate_mem_delta(
    delta::Float64,
    entering_var::Tuple{Int,Int},
    state::TabuSearchState,
    config::TabuSearchConfig
)
    return delta - config.alpha * (state.h[entering_var...] / state.k)
end

function get_longterm_mem_delta(
    delta::Float64,
    entering_var::Tuple{Int,Int},
    state::TabuSearchState,
    config::TabuSearchConfig
)
    return delta - config.alpha * (1 - state.h[entering_var...] / state.k)
end


function step_1(instance::Instance, config::TabuSearchConfig, state::TabuSearchState)
    """Step 1 / 8 / 17"""
    # if state.intermediate_mem_1
    #     println("Performing step 8")
    # elseif state.longterm_mem
    #     println("Performing step 17")
    # else
    #     println("Performing step 1")
    # end
    # Update origin
    state.i += 1
    if state.i > length(instance.supply)
        state.i = 1
    end
    candidates::Array{Tuple{Tuple{Tuple{Int,Int},Tuple{Int,Int},Dict{Tuple{Int,Int},Int},Float64},Float64}} = []
    # For each outgoing NBV, compute delta
    for entering_var in get_candidate_list(state, instance, true)
        delta, new_bfs, leaving_var = get_net_change(
            entering_var, instance.var_costs, instance.fix_costs, state.bfs, state.u, state.v
        )
        # If we are in intermediate memory process 1, compute delta' to intensify the search
        if state.intermediate_mem_1
            delta_prime = get_intermediate_mem_delta(
                delta, entering_var, state, config
            )
            push!(candidates, ((entering_var, leaving_var, new_bfs, delta), delta_prime))
            # If we are in the longterm memory process, compute delta'' to diversify the search
        elseif state.longterm_mem
            delta_prime2 = get_longterm_mem_delta(
                delta, entering_var, state, config
            )
            push!(candidates, ((entering_var, leaving_var, new_bfs, delta), delta_prime2))
        else
            push!(candidates, ((entering_var, leaving_var, new_bfs, delta), delta))
        end
    end
    # Sort candidates by delta / delta' / delta''
    candidates = sort(candidates, by=last)
    state._next = (i, c, s) -> step_2([c[1] for c in candidates], i, c, s)
    return true
end


function check_candidates(
    candidates::AbstractVector{Tuple{Tuple{Int,Int},Tuple{Int,Int},Dict{Tuple{Int,Int},Int},Float64}},
    config::TabuSearchConfig,
    state::TabuSearchState
)
    # Check candidate and return first (=best) candidates that fulfills the tabu (step 3)
    # or the aspiration criterion (step 4)
    for candidate in candidates
        var_in, var_out, _, cand_delta = candidate
        if check_tabu_status(
            var_in, var_out, state, config
        ) || check_aspiration(var_in, var_out, cand_delta, state)
            return candidate
        end
    end
    return nothing
end

function step_2(
    candidates::AbstractVector{Tuple{Tuple{Int,Int},Tuple{Int,Int},Dict{Tuple{Int,Int},Int},Float64}},
    instance::Instance,
    config::TabuSearchConfig,
    state::TabuSearchState
)
    """Step 2-4, Step 9-11, Step 18-20"""
    # if state.intermediate_mem_1
    #     println("Performing step 9-11")
    # elseif state.longterm_mem
    #     println("Performing step 18-20")
    # else
    #     println("Performing step 2-4")
    # end
    # Get best candidate that fulfills the tabu or the aspiration criterion
    candidate = check_candidates(candidates, config, state)
    if candidate !== nothing
        state._next = (i, c, s) -> step_5(candidate, i, c, s)
        # If no candidate fulfills the criterion, continue search with the next origin (-> Step 1)
    elseif state.longterm_mem
        state._next = step_22
    else
        state._next = step_1
    end
    return true
end

function step_5(
    candidate::Tuple{Tuple{Int,Int},Tuple{Int,Int},Dict{Tuple{Int,Int},Int},Float64},
    instance::Instance,
    config::TabuSearchConfig,
    state::TabuSearchState
)
    """Step 5 / 21 / 24"""
    # if state.longterm_mem && state.l_1
    #     println("Performing step 21")
    # elseif state.longterm_mem && !state.l_1
    #     println("Performing step 24")
    # else
    #     println("Performing step 5")
    # end
    var_in, var_out, cand_bfs, cand_delta = candidate
    # Update spanning tree
    state.bfs = cand_bfs
    (state.u, state.v) = get_dual_vars(state.bfs, instance.var_costs)
    # Update h for leaving variable
    state.h[var_out...] += (state.k - max(0, state.t[var_out...]))
    # Update t for entering and leaving variable
    state.t[var_in...] = state.k
    state.t[var_out...] = state.k
    # Update z for entering and leaving variable
    state.z[var_in...] = state.z_val
    state.z[var_out...] = state.z_val
    # Update iteration counter
    state.k += 1
    # Update z
    state.z_val += cand_delta
    ########## moved here from step 6 to not miss better solution
    if state.z_val < state.z_best
        state.z_best = state.z_val
        state.best_sol = copy(state.bfs)
        if isempty(config.log_file) == false
            write_to_log(config.log_file, "$(time()),$(state.z_best)")
        end
        println(String("New best solution $(state.z_best)"))
        println(state.best_sol)
    end
    ##########
    if state.longterm_mem
        state._next = step_22
    else
        state._next = step_6
    end
    return true
end

function step_6(
    instance::Instance,
    config::TabuSearchConfig,
    state::TabuSearchState
)
    # println("Performing step 6")
    if state.z_val < state.z_min
        state.z_min = state.z_val
        state.k_min = state.k
    end
    state._next = step_7
    return true
end

function step_7(
    instance::Instance,
    config::TabuSearchConfig,
    state::TabuSearchState
)
    # println("Performing step 7")
    if mod(state.k, 100) == 0
        println(state.k)
    end
    N = length(instance.supply) * length(instance.supply)
    was_interm_mem_1 = state.intermediate_mem_1
    state.intermediate_mem_1 = false
    # Short term memory process
    if state.k - state.k_min < config.beta * N
        if was_interm_mem_1
            println("Stopping intermediate memory process 1 and starting short term memory process")
        end
        state._next = step_1
        # Intermediate memory process 1
    elseif (
        config.beta * N
        <= state.k - state.k_min
        < (config.beta + config.gamma) * N
    )
        if !was_interm_mem_1
            println("Starting intermediate memory process 1")
        end
        state.intermediate_mem_1 = true
        state._next = step_1 # = Step 8
    # Intermediate memory process 2
    else
        println("Starting intermediate memory process 2")
        state._next = step_12
    end
    return true
end

function set_i0_j0(state::TabuSearchState, instance::Instance)
    state.i_0 = state.i
    for index in length(instance.demand):-1:1
        if instance.edge_mask[state.i_0, index]
            state.j_0 = index
            return
        end
    end
end

function step_12(
    instance::Instance,
    config::TabuSearchConfig,
    state::TabuSearchState
)
    # println("Performing step 12-15")
    # Intermediate memory process 2
    set_i0_j0(state, instance)
    stop = false
    while !stop
        # Step 13
        state.i += 1
        if state.i > length(instance.supply)
            state.i = 1
        end
        # Step 14
        for var_in in get_candidate_list(state, instance, false) # if (self.i, j) not in self.bfs # then var_in can never be == (i_0, j_0) once it has been reset to new bv
            if !(var_in in keys(state.bfs))
                # (a)
                cand_delta, cand_bfs, var_out = get_net_change(
                    var_in, instance.var_costs, instance.fix_costs, state.bfs, state.u, state.v
                )
                # (c) & (d)
                if cand_delta < 0
                    state.i_0, state.j_0 = var_in
                    state.bfs = cand_bfs
                    (state.u, state.v) = get_dual_vars(state.bfs, instance.var_costs)
                    state.h[var_out...] += (state.k - max(0, state.t[var_out...]))
                    state.t[var_in...] = state.k
                    state.t[var_out...] = state.k
                    state.z[var_in...] = state.z_val
                    state.z[var_out...] = state.z_val
                    state.k += 1
                    state.z_val += cand_delta
                    continue
                end
            end
            # (b)
            # if cand_delta >= 0 and var_in == (self.i_0,self.j_0):
            if var_in == (state.i_0, state.j_0)
                println("Stopping intermediate memory process 2 and starting longterm memory process")
                stop = true
                break  # go to step 16
            end
        end
    end
    state._next = step_16
    return true
end

function step_16(
    instance::Instance,
    config::TabuSearchConfig,
    state::TabuSearchState
)
    # println("Performing step 16")
    if state.z_val < state.z_best
        state.z_best = state.z_val
        state.best_sol = copy(state.bfs)
        if isempty(config.log_file) == false
            write_to_log(config.log_file, "$(time()),$(state.z_best)")
        end
        println(String("New best solution $(state.z_best)"))
        println(state.best_sol)
    end
    if state.l >= config.L
        println("Maximum number of search cycles reached -> stopping")
        ###############
        # STOPPING
        ###############
        state._next = terminate
        return true
    end
    state.i_0 = state.i
    state.longterm_mem = true
    # switch
    state.l_1 = !(state.l_1)
    if state.l_1
        # Longterm memory process 1
        println("Starting longterm memory process 1")
        state._next = step_1 # = Step 17
    else
        # Longterm memory process 2
        println("Starting longterm memory process 2")
        state._next = step_23
    end
    return true
end

function step_22(
    instance::Instance,
    config::TabuSearchConfig,
    state::TabuSearchState
)
    """Step 22 / 25"""
    # if state.l_1
    #     println("Performing step 22")
    # else
    #     println("Performing step 25")
    # end
    # back to i_0 -> stop longterm memory process
    if state.i == state.i_0
        println("Stopping longterm memory process and starting short term memory process")
        state.longterm_mem = false
        state.z_min = state.z_val
        state.k_min = state.k
        if !state.l_1
            # increase counter after longterm memory process 2
            state.l += 1
        end
        reset_tabu_vals(config)
        state._next = step_1
        return true
    end
    # otherwise: continue searching within longterm memory process
    if state.l_1
        state._next = step_1 # = Step 17
    else
        state._next = step_23
    end
    return true
end

function step_23(
    instance::Instance,
    config::TabuSearchConfig,
    state::TabuSearchState
)
    # println("Performing step 23")
    state.i += 1
    if state.i > length(instance.supply)
        state.i = 1
    end
    nbv_out = [
        (c, state.t[c...]) for c in get_candidate_list(state, instance, true)
    ]
    # For the reduced problem, it is possible that there are not NBVs for this origin
    if length(nbv_out) == 0
        state._next = step_22
        return true
    end

    # Select arc which has been out of the basis for the longest time (= smallest t)
    var_in = sort(nbv_out, by=last)[1][1]
    delta, new_bfs, var_out = get_net_change(
        var_in, instance.var_costs, instance.fix_costs, state.bfs, state.u, state.v
    )
    candidate = (var_in, var_out, new_bfs, delta)
    state._next = (i, c, s) -> step_5(candidate, i, c, s)
    return true
end

function terminate(
    instance::Instance,
    config::TabuSearchConfig,
    state::TabuSearchState
)
    return false
end

function tabu_search(instance::Instance, config::TabuSearchConfig, bfs::Dict{Tuple{Int,Int},Int})
    Random.seed!(config.seed)
    m = length(instance.supply)
    n = length(instance.demand)
    bfs_val = evaluate_sol(bfs, instance.var_costs, instance.fix_costs)
    (u, v) = get_dual_vars(bfs, instance.var_costs)
    state = TabuSearchState(m, n, bfs, bfs_val, u, v)
    return_val = true
    runtime = @elapsed begin
        while return_val
            return_val = state._next(instance, config, state)
        end
    end
    if isempty(config.log_file) == false
        write_to_log(config.log_file, "$(time()),$(state.z_best)")
    end
    return (state.best_sol, state.z_best, runtime)
end


##################################
# Wrapper
##################################

function ts_wrapper(supply, demand, var_costs, fix_costs, edge_mask, bfs, ts_config)
    bfs = convert(Dict{Tuple{Int,Int},Int}, bfs)
    instance = Instance(supply, demand, var_costs, fix_costs, edge_mask)
    config = TabuSearchConfig(
        get!(ts_config, "tabu_in_range", (7, 10)),
        get!(ts_config, "tabu_out_range", (2, 4)),
        get!(ts_config, "beta", 0.5),
        get!(ts_config, "gamma", 0.5),
        get_alpha(bfs, instance),
        get!(ts_config, "L", 5),
        get!(ts_config, "seed", 0),
        get!(ts_config, "log_file", ""),
    )
    sol, sol_val, runtime = tabu_search(instance, config, bfs)
    return (sol_dict_to_matrix(sol, length(supply), length(demand)), sol_val, runtime)
end

end